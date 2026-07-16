from __future__ import annotations

import argparse
import inspect
import json
import random
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, load_from_disk
from peft import PeftModel

from pipeline.adapter_provenance import write_adapter_provenance
from pipeline.cpt_training import (
    NonFiniteTrainingCallback,
    _build_peft_model,
    _load_model,
    _parameter_counts,
    _peft_config,
    _resolve_training_device,
    _resolve_training_precision,
    cast_trainable_parameters_to_fp32,
    trainable_parameter_dtype_counts,
)
from pipeline.utils import (
    config_without_private_keys,
    copy_file,
    deep_update,
    load_config,
    normalize_path_for_report,
    parse_nullable_int,
    read_json,
    read_text,
    resolve_input_path,
    resolve_training_path,
    save_yaml,
    setup_logging,
    short_error,
    summarize_loss_history,
    utc_now,
    write_json,
    write_text,
)


VALID_SUFFIXES = {".jsonl", ".json"}


def _collect_dpo_files(config: dict[str, Any], config_path: Path) -> list[Path]:
    dpo_cfg = config.get("dpo", {})
    raw_paths = []
    if dpo_cfg.get("input_path"):
        raw_paths.append(dpo_cfg["input_path"])
    raw_paths.extend(dpo_cfg.get("input_paths", []))
    if not raw_paths:
        raw_paths.append("data/dpo/dpo_with_rejected_from_merged.jsonl")

    files: list[Path] = []
    for raw_path in raw_paths:
        path = resolve_input_path(raw_path, config_path).resolve()
        if not path.exists():
            continue
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES:
            files.append(path)
        elif path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and candidate.suffix.lower() in VALID_SUFFIXES:
                    files.append(candidate.resolve())
    return list(dict.fromkeys(files))


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = []
        for line_number, line in enumerate(read_text(path).splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
            records.append(item)
        return records
    payload = json.loads(read_text(path))
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError(f"JSON list must contain objects: {path}")
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return [item for item in payload["data"] if isinstance(item, dict)]
    raise ValueError(f"Unsupported DPO JSON shape: {path}")


def _normalise_dpo_record(item: dict[str, Any], *, source_path: Path, source_index: int) -> dict[str, Any]:
    prompt = str(item.get("prompt") or item.get("instruction") or item.get("question") or "").strip()
    chosen = str(item.get("chosen") or item.get("preferred") or item.get("accept") or "").strip()
    rejected = str(item.get("rejected") or item.get("bad") or item.get("reject") or "").strip()
    if not prompt:
        raise ValueError("missing prompt")
    if not chosen:
        raise ValueError("missing chosen")
    if not rejected:
        raise ValueError("missing rejected")
    if chosen == rejected:
        raise ValueError("chosen and rejected are identical")
    return {
        "id": str(item.get("id") or f"{source_path.stem}-{source_index:06d}"),
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "category": str(item.get("category") or item.get("type") or "general"),
        "origin": str(item.get("origin") or ""),
        "source": str(item.get("source") or ""),
        "source_path": str(source_path),
        "source_index": int(source_index),
    }


def prepare_dpo_dataset(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    logger = setup_logging("prepare_dpo")
    dpo_cfg = config.get("dpo", {})
    output_dir = resolve_training_path(dpo_cfg.get("prepared_dataset_dir"), "outputs/dpo_dataset")
    validation_ratio = float(dpo_cfg.get("validation_ratio", 0.0) or 0.0)
    seed = int(dpo_cfg.get("seed", config.get("training", {}).get("seed", 42)))

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    source_stats: list[dict[str, Any]] = []
    for path in _collect_dpo_files(config, config_path):
        records = _read_json_records(path)
        valid_for_source = 0
        for index, item in enumerate(records):
            try:
                rows.append(_normalise_dpo_record(item, source_path=path, source_index=index))
                valid_for_source += 1
            except Exception as exc:
                skipped.append({"path": str(path), "index": index, "reason": short_error(exc)})
        source_stats.append(
            {
                "path": str(path),
                "display_path": normalize_path_for_report(path),
                "records": len(records),
                "valid_records": valid_for_source,
            }
        )

    if not rows:
        raise RuntimeError(
            "DPO is enabled, but no valid preference pairs were found. "
            "Each row needs non-empty prompt, chosen, and rejected fields."
        )

    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    validation_count = 0
    if validation_ratio > 0 and len(shuffled) > 1:
        validation_count = min(len(shuffled) - 1, max(1, int(round(len(shuffled) * validation_ratio))))
    validation = shuffled[:validation_count]
    train = shuffled[validation_count:]

    dataset = DatasetDict({"train": Dataset.from_list(train)})
    if validation:
        dataset["validation"] = Dataset.from_list(validation)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))

    category_counts = Counter(item["category"] for item in rows)
    report = {
        "status": "ok",
        "source_stats": source_stats,
        "prepared_dataset_dir": str(output_dir),
        "total_valid_pairs": len(rows),
        "train_pairs": len(train),
        "validation_pairs": len(validation),
        "skipped_pairs": skipped,
        "category_counts": dict(sorted(category_counts.items())),
        "created_at_utc": utc_now(),
    }
    write_json(output_dir / "dpo_dataset_report.json", report)
    write_dpo_dataset_report(output_dir / "dpo_dataset_report.md", report)
    save_yaml(output_dir / "config_snapshot.yaml", config_without_private_keys(config))
    logger.info("Prepared DPO dataset at %s: %d train, %d validation", output_dir, len(train), len(validation))
    return report


def write_dpo_dataset_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# DPO Dataset Report",
        "",
        f"Status: **{report.get('status')}**",
        f"Created at UTC: `{report.get('created_at_utc')}`",
        "",
        "## Summary",
        "",
        f"- Valid preference pairs: `{report.get('total_valid_pairs')}`",
        f"- Train pairs: `{report.get('train_pairs')}`",
        f"- Validation pairs: `{report.get('validation_pairs')}`",
        f"- Skipped pairs: `{len(report.get('skipped_pairs', []))}`",
        "",
        "## Categories",
        "",
    ]
    for category, count in report.get("category_counts", {}).items():
        lines.append(f"- `{category}`: `{count}`")
    if report.get("skipped_pairs"):
        lines.extend(["", "## Skipped Pairs", ""])
        for item in report["skipped_pairs"][:100]:
            lines.append(f"- `{item.get('path')}:{item.get('index')}`: {item.get('reason')}")
    write_text(path, "\n".join(lines) + "\n")


def _dpo_training_config(config: dict[str, Any]) -> dict[str, Any]:
    dpo_cfg = config.get("dpo", {})
    inherited = dict(config.get("training", {}))
    overrides = {
        "output_dir": dpo_cfg.get("output_dir", "outputs/dpo_adapter"),
        "max_seq_length": dpo_cfg.get("max_seq_length", min(1024, int(inherited.get("max_seq_length", 2048)))),
        "epochs": dpo_cfg.get("epochs", 1),
        "max_steps": dpo_cfg.get("max_steps", None),
        "per_device_train_batch_size": dpo_cfg.get("per_device_train_batch_size", inherited.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": dpo_cfg.get("per_device_eval_batch_size", inherited.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": dpo_cfg.get("gradient_accumulation_steps", inherited.get("gradient_accumulation_steps", 4)),
        "learning_rate": dpo_cfg.get("learning_rate", 1e-6),
        "weight_decay": dpo_cfg.get("weight_decay", 0.0),
        "warmup_ratio": dpo_cfg.get("warmup_ratio", 0.10),
        "lr_scheduler_type": dpo_cfg.get("lr_scheduler_type", "cosine"),
        "logging_steps": dpo_cfg.get("logging_steps", 1),
        "eval_steps": dpo_cfg.get("eval_steps", 20),
        "save_steps": dpo_cfg.get("save_steps", 50),
        "save_total_limit": dpo_cfg.get("save_total_limit", 2),
        "max_grad_norm": dpo_cfg.get("max_grad_norm", 0.05),
        "optim": dpo_cfg.get("optim", inherited.get("optim", "paged_adamw_8bit")),
        "bf16": dpo_cfg.get("bf16", False),
        "fp16": dpo_cfg.get("fp16", False),
        "torch_dtype": dpo_cfg.get("torch_dtype", inherited.get("torch_dtype", "float16")),
        "abort_on_nonfinite_grad_norm": dpo_cfg.get(
            "abort_on_nonfinite_grad_norm",
            inherited.get("abort_on_nonfinite_grad_norm", False),
        ),
        "logging_nan_inf_filter": dpo_cfg.get(
            "logging_nan_inf_filter",
            inherited.get("logging_nan_inf_filter", False),
        ),
        "resume_from_checkpoint": dpo_cfg.get("resume_from_checkpoint", None),
    }
    inherited.update(overrides)
    return inherited


def _dpo_config_kwargs(config: dict[str, Any], output_dir: Path, has_eval: bool, dpo_config_cls) -> dict[str, Any]:
    dpo_cfg = config.get("dpo", {})
    dpo_training = _dpo_training_config(config)
    bf16, fp16 = _resolve_training_precision(dpo_training)
    max_steps = parse_nullable_int(dpo_training.get("max_steps"))
    loss_type = dpo_cfg.get("loss_type", "sigmoid")
    if isinstance(loss_type, str):
        loss_type = [loss_type]
    args_kwargs = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "num_train_epochs": float(dpo_training.get("epochs", 1)),
        "max_steps": max_steps if max_steps is not None else -1,
        "per_device_train_batch_size": int(dpo_training.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(dpo_training.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(dpo_training.get("gradient_accumulation_steps", 4)),
        "learning_rate": float(dpo_training.get("learning_rate", 1e-6)),
        "weight_decay": float(dpo_training.get("weight_decay", 0.0)),
        "warmup_ratio": float(dpo_training.get("warmup_ratio", 0.10)),
        "lr_scheduler_type": dpo_training.get("lr_scheduler_type", "cosine"),
        "logging_steps": int(dpo_training.get("logging_steps", 1)),
        "save_steps": int(dpo_training.get("save_steps", 50)),
        "save_total_limit": int(dpo_training.get("save_total_limit", 2)),
        "max_grad_norm": float(dpo_training.get("max_grad_norm", 0.05)),
        "optim": dpo_training.get("optim", "paged_adamw_8bit"),
        "bf16": bf16,
        "fp16": fp16,
        "bf16_full_eval": False,
        "fp16_full_eval": False,
        "logging_nan_inf_filter": bool(dpo_training.get("logging_nan_inf_filter", False)),
        "report_to": [],
        "remove_unused_columns": False,
        "save_safetensors": True,
        "gradient_checkpointing": bool(dpo_training.get("gradient_checkpointing", True)),
        "beta": float(dpo_cfg.get("beta", 0.1)),
        "loss_type": loss_type,
        "max_length": dpo_cfg.get("max_length", dpo_training.get("max_seq_length", 1024)),
        "truncation_mode": dpo_cfg.get("truncation_mode", "keep_start"),
        "precompute_ref_log_probs": bool(dpo_cfg.get("precompute_ref_log_probs", False)),
    }
    eval_strategy_key = "eval_strategy" if "eval_strategy" in inspect.signature(dpo_config_cls).parameters else "evaluation_strategy"
    args_kwargs[eval_strategy_key] = "steps" if has_eval else "no"
    if has_eval:
        args_kwargs["eval_steps"] = int(dpo_training.get("eval_steps", 20))
    supported = set(inspect.signature(dpo_config_cls).parameters)
    return {key: value for key, value in args_kwargs.items() if key in supported}


def _drop_reference_adapter_if_present(model: Any, logger: Any) -> None:
    peft_config = getattr(model, "peft_config", {}) or {}
    if "ref" not in peft_config:
        return
    try:
        if hasattr(model, "set_adapter") and "default" in peft_config:
            model.set_adapter("default")
        if hasattr(model, "delete_adapter"):
            model.delete_adapter("ref")
            logger.info("Removed temporary DPO reference adapter before saving.")
    except Exception:
        logger.warning("Could not remove temporary DPO reference adapter before saving.", exc_info=True)


def train_dpo(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    logger = setup_logging("train_dpo")
    try:
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        raise RuntimeError("DPO training requires `trl`. Install dependencies with `pip install -r requirements.txt`.") from exc

    dpo_cfg = config.get("dpo", {})
    dataset_dir = resolve_training_path(dpo_cfg.get("prepared_dataset_dir"), "outputs/dpo_dataset")
    output_dir = resolve_training_path(dpo_cfg.get("output_dir"), "outputs/dpo_adapter")
    base_adapter_dir = resolve_training_path(dpo_cfg.get("base_adapter_dir"), "outputs/fact_sft_adapter")
    require_base_adapter = bool(dpo_cfg.get("require_base_adapter", True))
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Prepared DPO dataset directory not found: {dataset_dir}")
    if require_base_adapter and not base_adapter_dir.exists():
        raise FileNotFoundError(f"DPO requires the base adapter first, but it was not found: {base_adapter_dir}")

    dataset = load_from_disk(str(dataset_dir))
    if "train" not in dataset or len(dataset["train"]) == 0:
        raise RuntimeError(f"Prepared DPO dataset has no train split: {dataset_dir}")
    has_eval = "validation" in dataset and len(dataset["validation"]) > 0

    dpo_training = _dpo_training_config(config)
    active_config = deep_update(config_without_private_keys(config), {"training": dpo_training})
    model = _load_model(active_config, logger)
    if base_adapter_dir.exists():
        logger.info("Continuing DPO from PEFT adapter: %s", base_adapter_dir)
        model = PeftModel.from_pretrained(model, str(base_adapter_dir), is_trainable=True)
    else:
        logger.info("No DPO base adapter found; creating a fresh PEFT adapter.")
        model = _build_peft_model(model, active_config)
    trainable_cast = cast_trainable_parameters_to_fp32(model, logger)
    trainable_dtypes_before_trainer = trainable_parameter_dtype_counts(model)
    logger.info("DPO precision resolved: bf16=%s fp16=%s", *_resolve_training_precision(dpo_training))
    logger.info("Trainable parameter dtype counts before DPOTrainer: %s", trainable_dtypes_before_trainer)
    param_counts = _parameter_counts(model)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    tokenizer = load_tokenizer_for_dpo(config)
    bf16, fp16 = _resolve_training_precision(dpo_training)
    dpo_args = DPOConfig(**_dpo_config_kwargs(config, output_dir, has_eval, DPOConfig))
    trainer_kwargs = {
        "model": model,
        "ref_model": None,
        "args": dpo_args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"] if has_eval else None,
        "processing_class": tokenizer,
        "callbacks": [
            NonFiniteTrainingCallback(
                abort_on_nonfinite_grad_norm=bool(dpo_training.get("abort_on_nonfinite_grad_norm", False)),
                logger=logger,
            )
        ],
    }
    trainer_params = inspect.signature(DPOTrainer).parameters
    trainer_kwargs = {key: value for key, value in trainer_kwargs.items() if key in trainer_params}
    trainer = DPOTrainer(**trainer_kwargs)
    post_trainer_cast = cast_trainable_parameters_to_fp32(trainer.model, logger)
    trainable_dtypes_after_trainer = trainable_parameter_dtype_counts(trainer.model)
    logger.info(
        "DPO Trainer args precision: bf16=%s fp16=%s bf16_full_eval=%s fp16_full_eval=%s",
        getattr(dpo_args, "bf16", None),
        getattr(dpo_args, "fp16", None),
        getattr(dpo_args, "bf16_full_eval", None),
        getattr(dpo_args, "fp16_full_eval", None),
    )
    logger.info("Trainable parameter dtype counts after DPOTrainer: %s", trainable_dtypes_after_trainer)
    resume = dpo_training.get("resume_from_checkpoint")
    logger.info("Starting DPO training. beta=%s, loss_type=%s", dpo_cfg.get("beta", 0.1), dpo_cfg.get("loss_type", "sigmoid"))
    train_result = trainer.train(resume_from_checkpoint=resume if resume else None)

    _drop_reference_adapter_if_present(trainer.model, logger)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    trainer.save_state()

    save_yaml(output_dir / "config_snapshot.yaml", config_without_private_keys(config))
    save_yaml(output_dir / "original_config.yaml", config_without_private_keys(config))
    if (base_adapter_dir / "fact_sft_training_metadata.json").exists():
        copy_file(base_adapter_dir / "fact_sft_training_metadata.json", output_dir / "base_fact_sft_training_metadata.json")
    elif (base_adapter_dir / "training_metadata.json").exists():
        copy_file(base_adapter_dir / "training_metadata.json", output_dir / "base_training_metadata.json")
    write_json(output_dir / "training_args.json", dpo_args.to_dict())
    log_history = trainer.state.log_history
    dataset_report = read_json(dataset_dir / "dpo_dataset_report.json", default={})
    metadata = {
        "status": "completed",
        "base_model_name_or_path": config["base_model_name_or_path"],
        "base_adapter_dir": str(base_adapter_dir) if base_adapter_dir.exists() else "",
        "adapter_output_dir": str(output_dir),
        "dataset_dir": str(dataset_dir),
        "dataset": dataset_report,
        "training": dpo_training,
        "dpo": {
            "beta": float(dpo_cfg.get("beta", 0.1)),
            "loss_type": dpo_cfg.get("loss_type", "sigmoid"),
            "reference_adapter_mode": "trl_peft_ref_adapter",
        },
        "peft": _peft_config(config),
        "precision": {"bf16": bf16, "fp16": fp16},
        "trainable_parameter_cast": trainable_cast,
        "post_trainer_trainable_parameter_cast": post_trainer_cast,
        "trainable_parameter_dtypes_before_trainer": trainable_dtypes_before_trainer,
        "trainable_parameter_dtypes_after_trainer": trainable_dtypes_after_trainer,
        "device": _resolve_training_device(dpo_training),
        "parameter_counts": param_counts,
        "train_result": train_result.metrics,
        "loss_summary": {
            "loss": summarize_loss_history(log_history, "loss"),
            "eval_loss": summarize_loss_history(log_history, "eval_loss"),
        },
        "has_validation": has_eval,
        "created_at_utc": utc_now(),
    }
    write_json(output_dir / "dpo_training_metadata.json", metadata)
    write_adapter_provenance(
        output_dir,
        stage="dpo",
        created_at_utc=metadata["created_at_utc"],
        base_adapter_dir=base_adapter_dir if base_adapter_dir.exists() else None,
    )
    write_dpo_training_report(resolve_training_path("outputs/reports/dpo_report.md", "outputs/reports/dpo_report.md"), metadata)
    return metadata


def load_tokenizer_for_dpo(config: dict[str, Any]):
    from pipeline.modeling import load_tokenizer

    return load_tokenizer(
        config["base_model_name_or_path"],
        trust_remote_code=bool(config.get("trust_remote_code", True)),
    )


def write_dpo_training_report(path: Path, metadata: dict[str, Any]) -> None:
    dataset = metadata.get("dataset", {})
    params = metadata.get("parameter_counts", {})
    loss = metadata.get("loss_summary", {})
    lines = [
        "# DPO Training Report",
        "",
        f"Status: **{metadata.get('status')}**",
        f"Created at UTC: `{metadata.get('created_at_utc')}`",
        "",
        "## Dataset",
        "",
        f"- Train pairs: `{dataset.get('train_pairs', 'N/A')}`",
        f"- Validation pairs: `{dataset.get('validation_pairs', 'N/A')}`",
        f"- Categories: `{dataset.get('category_counts', {})}`",
        "",
        "## Training",
        "",
        f"- Base model: `{metadata.get('base_model_name_or_path')}`",
        f"- Base adapter: `{metadata.get('base_adapter_dir')}`",
        f"- Output adapter: `{metadata.get('adapter_output_dir')}`",
        f"- beta: `{metadata.get('dpo', {}).get('beta')}`",
        f"- loss_type: `{metadata.get('dpo', {}).get('loss_type')}`",
        f"- learning_rate: `{metadata.get('training', {}).get('learning_rate')}`",
        f"- epoch / max_steps: `{metadata.get('training', {}).get('epochs')}` / `{metadata.get('training', {}).get('max_steps')}`",
        f"- gradient accumulation: `{metadata.get('training', {}).get('gradient_accumulation_steps')}`",
        f"- bf16 / fp16: `{metadata.get('precision', {}).get('bf16')}` / `{metadata.get('precision', {}).get('fp16')}`",
        "",
        "## Parameters",
        "",
        f"- Trainable params: `{params.get('trainable_params')}`",
        f"- Total params: `{params.get('total_params')}`",
        "",
        "## Loss Summary",
        "",
        f"- Train loss: `{loss.get('loss', {})}`",
        f"- Eval loss: `{loss.get('eval_loss', {})}`",
        "",
    ]
    write_text(path, "\n".join(lines))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and train the optional post-SFT DPO PEFT stage.")
    parser.add_argument("--config", default="configs/domain_post_training.yaml")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--train_only", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--input_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--base_adapter_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    logger = setup_logging("train_dpo", args.verbose)
    config, config_path = load_config(args.config)
    dpo_cfg = config.setdefault("dpo", {})
    dpo_cfg["enabled"] = True
    if args.max_steps is not None:
        dpo_cfg["max_steps"] = args.max_steps
    if args.learning_rate is not None:
        dpo_cfg["learning_rate"] = args.learning_rate
    if args.beta is not None:
        dpo_cfg["beta"] = args.beta
    if args.input_path:
        dpo_cfg["input_path"] = args.input_path
    if args.output_dir:
        dpo_cfg["output_dir"] = args.output_dir
    if args.base_adapter_dir:
        dpo_cfg["base_adapter_dir"] = args.base_adapter_dir
    if args.device:
        config.setdefault("training", {})["device"] = args.device
    try:
        if not args.train_only:
            prepare_dpo_dataset(config, config_path)
        if not args.prepare_only:
            train_dpo(config, config_path)
    except RuntimeError as exc:
        message = short_error(exc)
        if "out of memory" in str(exc).lower():
            message += "; CUDA OOM: lower dpo.max_length, batch size, or LoRA rank."
        logger.error(message)
        logger.debug(traceback.format_exc())
        return 9
    except Exception as exc:
        logger.error(short_error(exc))
        logger.debug(traceback.format_exc())
        return 9
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
