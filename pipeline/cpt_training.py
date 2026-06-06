from __future__ import annotations

import argparse
import inspect
import json
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from pipeline.modeling import configure_generation_tokens, disable_cache, load_tokenizer, load_transformers_model
from pipeline.utils import (
    copy_file,
    format_number,
    load_config,
    parse_nullable_int,
    read_json,
    resolve_bool_auto,
    resolve_precision_flags,
    resolve_training_path,
    save_yaml,
    setup_logging,
    short_error,
    summarize_loss_history,
    torch_dtype_from_config,
    utc_now,
    write_json,
    write_text,
)


@dataclass
class FullTokenLossCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids = [feature["input_ids"] for feature in features]
        attention_mask = [feature.get("attention_mask", [1] * len(feature["input_ids"])) for feature in features]
        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )
        labels = torch.full_like(batch["input_ids"], -100)
        for row_index, feature in enumerate(features):
            label_ids = feature.get("labels", feature["input_ids"])
            label_tensor = torch.tensor(label_ids, dtype=torch.long)
            labels[row_index, : label_tensor.numel()] = label_tensor
        batch["labels"] = labels
        return batch


class NonFiniteTrainingCallback(TrainerCallback):
    def __init__(self, *, abort_on_nonfinite_grad_norm: bool = True, logger: Any | None = None) -> None:
        self.abort_on_nonfinite_grad_norm = abort_on_nonfinite_grad_norm
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        for key in ("loss", "grad_norm"):
            if not logs or key not in logs:
                continue
            try:
                value = float(logs[key])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                if key == "grad_norm" and not self.abort_on_nonfinite_grad_norm:
                    if self.logger:
                        self.logger.warning(
                            "Non-finite grad_norm detected at step %s: grad_norm=%s. "
                            "Continuing because training.abort_on_nonfinite_grad_norm=false; "
                            "loss remains the hard-stop metric.",
                            state.global_step,
                            logs[key],
                        )
                    continue
                raise RuntimeError(
                    f"Non-finite training metric detected at step {state.global_step}: {key}={logs[key]}. "
                    "Stop this run and restart from a clean adapter output directory with safer precision/LR settings."
                )


def _apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.base_model_name_or_path:
        config["base_model_name_or_path"] = args.base_model_name_or_path
    training = config.setdefault("training", {})
    if args.max_seq_length is not None:
        training["max_seq_length"] = args.max_seq_length
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.max_steps is not None:
        training["max_steps"] = args.max_steps
    if args.learning_rate is not None:
        training["learning_rate"] = args.learning_rate
    if args.load_in_4bit:
        training["load_in_4bit"] = True
    if args.device:
        training["device"] = args.device
    if args.output_dir:
        training["output_dir"] = args.output_dir
    if args.resume_from_checkpoint:
        training["resume_from_checkpoint"] = args.resume_from_checkpoint
    return config


def _load_tokenizer(config: dict[str, Any]):
    return load_tokenizer(
        config["base_model_name_or_path"],
        trust_remote_code=bool(config.get("trust_remote_code", True)),
    )


def _resolve_training_device(training_cfg: dict[str, Any]) -> str | None:
    requested = str(training_cfg.get("device", "cuda")).strip().lower()
    if requested in {"", "none"}:
        return None
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Training device is set to CUDA, but torch.cuda.is_available() is false.")
    return requested


def _resolve_training_precision(training_cfg: dict[str, Any]) -> tuple[bool, bool]:
    bf16, fp16 = resolve_precision_flags(training_cfg, torch)
    if _resolve_training_device(training_cfg) == "cpu":
        return False, False
    return bf16, fp16


def _load_model(config: dict[str, Any], logger):
    training_cfg = config.get("training", {})
    base_model = config["base_model_name_or_path"]
    trust_remote_code = bool(config.get("trust_remote_code", True))
    load_in_4bit = bool(training_cfg.get("load_in_4bit", False))
    load_in_8bit = bool(training_cfg.get("load_in_8bit", False))
    device = _resolve_training_device(training_cfg)
    if load_in_4bit and load_in_8bit:
        raise ValueError("Only one of load_in_4bit/load_in_8bit can be true.")

    dtype = torch_dtype_from_config(training_cfg.get("torch_dtype", "auto"), torch)
    model_kwargs: dict[str, Any] = {}
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype

    if load_in_4bit or load_in_8bit:
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        logger.info("Loading quantized base model. DeepSpeed is intentionally not used.")
        device_map: str | dict[str, str | int] = "auto"
        if device and device.startswith("cuda"):
            device_map = {"": device}
        elif device == "cpu":
            device_map = {"": "cpu"}
        model = load_transformers_model(
            base_model,
            trust_remote_code=trust_remote_code,
            logger=logger,
            quantization_config=quant_config,
            device_map=device_map,
            **model_kwargs,
        )
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", True)),
        )
    else:
        model = load_transformers_model(base_model, trust_remote_code=trust_remote_code, logger=logger, **model_kwargs)
        if device:
            logger.info("Moving base model to training device: %s", device)
            model.to(torch.device(device))

    tokenizer = _load_tokenizer(config)
    configure_generation_tokens(model, tokenizer)
    if bool(training_cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        disable_cache(model)
    return model


def _peft_config(config: dict[str, Any]) -> dict[str, Any]:
    if "peft" in config:
        return dict(config.get("peft") or {})
    return dict(config.get("dora") or {})


def _build_peft_model(model, config: dict[str, Any]):
    peft_cfg = _peft_config(config)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(peft_cfg.get("r", 16)),
        lora_alpha=int(peft_cfg.get("lora_alpha", 16)),
        lora_dropout=float(peft_cfg.get("lora_dropout", 0.0)),
        target_modules=peft_cfg.get("target_modules", "all-linear"),
        bias=peft_cfg.get("bias", "none"),
        use_dora=bool(peft_cfg.get("use_dora", False)),
        use_rslora=bool(peft_cfg.get("use_rslora", False)),
    )
    return get_peft_model(model, peft_config)


def _parameter_counts(model) -> dict[str, Any]:
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    ratio = trainable / total if total else 0.0
    return {"trainable_params": trainable, "total_params": total, "trainable_ratio": ratio}


def cast_trainable_parameters_to_fp32(model, logger=None) -> dict[str, int]:
    converted_tensors = 0
    converted_params = 0
    for _, param in model.named_parameters():
        if not param.requires_grad or not param.is_floating_point() or param.dtype == torch.float32:
            continue
        converted_tensors += 1
        converted_params += param.numel()
        param.data = param.data.float()
        if param.grad is not None:
            param.grad.data = param.grad.data.float()
    if converted_tensors and logger:
        logger.info(
            "Cast %d trainable floating tensors (%d parameters) to fp32 for stable AMP training.",
            converted_tensors,
            converted_params,
        )
    return {"converted_tensors": converted_tensors, "converted_params": converted_params}


def trainable_parameter_dtype_counts(model) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        key = str(param.dtype)
        counts[key] = counts.get(key, 0) + param.numel()
    return dict(sorted(counts.items()))


def _training_args_kwargs(config: dict[str, Any], output_dir: Path, has_eval: bool) -> dict[str, Any]:
    training_cfg = config.get("training", {})
    bf16, fp16 = _resolve_training_precision(training_cfg)
    max_steps = parse_nullable_int(training_cfg.get("max_steps"))
    args_kwargs = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "num_train_epochs": float(training_cfg.get("epochs", 3)),
        "max_steps": max_steps if max_steps is not None else -1,
        "per_device_train_batch_size": int(training_cfg.get("per_device_train_batch_size", 1)),
        "per_device_eval_batch_size": int(training_cfg.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(training_cfg.get("gradient_accumulation_steps", 16)),
        "learning_rate": float(training_cfg.get("learning_rate", 1e-4)),
        "weight_decay": float(training_cfg.get("weight_decay", 0.0)),
        "warmup_ratio": float(training_cfg.get("warmup_ratio", 0.03)),
        "lr_scheduler_type": training_cfg.get("lr_scheduler_type", "cosine"),
        "logging_steps": int(training_cfg.get("logging_steps", 10)),
        "save_steps": int(training_cfg.get("save_steps", 200)),
        "save_total_limit": int(training_cfg.get("save_total_limit", 3)),
        "max_grad_norm": float(training_cfg.get("max_grad_norm", 1.0)),
        "optim": training_cfg.get("optim", "adamw_torch"),
        "bf16": bf16,
        "fp16": fp16,
        "logging_nan_inf_filter": bool(training_cfg.get("logging_nan_inf_filter", False)),
        "report_to": [],
        "remove_unused_columns": False,
        "save_safetensors": True,
    }
    if has_eval:
        args_kwargs["eval_steps"] = int(training_cfg.get("eval_steps", 100))
        eval_strategy_key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments).parameters else "evaluation_strategy"
        args_kwargs[eval_strategy_key] = "steps"
    else:
        eval_strategy_key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments).parameters else "evaluation_strategy"
        args_kwargs[eval_strategy_key] = "no"
    supported = set(inspect.signature(TrainingArguments).parameters)
    return {key: value for key, value in args_kwargs.items() if key in supported}


def write_training_report(report_path: Path, metadata: dict[str, Any]) -> None:
    dataset = metadata.get("dataset", {})
    params = metadata.get("parameter_counts", {})
    loss = metadata.get("loss_summary", {})
    lines = [
        "# PEFT CPT Training Report",
        "",
        f"Status: **{metadata.get('status')}**",
        f"Created at UTC: `{metadata.get('created_at_utc')}`",
        "",
        "## Configuration",
        "",
        f"- Base model: `{metadata.get('base_model_name_or_path')}`",
        f"- CPT document paths: {', '.join('`' + p + '`' for p in dataset.get('source_paths', [])) or 'N/A'}",
        f"- CPT document characters: `{dataset.get('total_chars', 'N/A')}`",
        f"- CPT document tokens: `{dataset.get('total_tokens', 'N/A')}`",
        f"- CPT sample unit: `{dataset.get('sample_unit', 'N/A')}`",
        f"- max_seq_length: `{dataset.get('max_seq_length', 'N/A')}`",
        f"- max_sample_tokens: `{dataset.get('max_sample_tokens', 'N/A')}`",
        f"- Training samples: `{dataset.get('train_samples', 'N/A')}`",
        f"- Validation samples: `{dataset.get('validation_samples', 'N/A')}`",
        f"- Max train sample tokens: `{dataset.get('max_train_sample_tokens', 'N/A')}`",
        f"- Truncated train samples: `{dataset.get('truncated_train_samples', 'N/A')}`",
        f"- Validation mode: `{dataset.get('validation_mode', 'N/A')}`",
        f"- Device: `{metadata.get('device', 'N/A')}`",
        f"- Mandatory coverage rate: `{format_number(dataset.get('mandatory_coverage_rate'))}`",
        f"- Train mandatory samples: `{dataset.get('train_mandatory_chunks', 'N/A')}`",
        f"- Train replay samples: `{dataset.get('train_replay_chunks', 'N/A')}`",
        f"- PEFT rank: `{metadata.get('peft', {}).get('r')}`",
        f"- PEFT alpha: `{metadata.get('peft', {}).get('lora_alpha')}`",
        f"- PEFT dropout: `{metadata.get('peft', {}).get('lora_dropout')}`",
        f"- target_modules: `{metadata.get('peft', {}).get('target_modules')}`",
        f"- learning_rate: `{metadata.get('training', {}).get('learning_rate')}`",
        f"- epoch / max_steps: `{metadata.get('training', {}).get('epochs')}` / `{metadata.get('training', {}).get('max_steps')}`",
        f"- batch size: `{metadata.get('training', {}).get('per_device_train_batch_size')}`",
        f"- gradient accumulation: `{metadata.get('training', {}).get('gradient_accumulation_steps')}`",
        f"- 4bit: `{metadata.get('training', {}).get('load_in_4bit')}`",
        f"- bf16 / fp16: `{metadata.get('precision', {}).get('bf16')}` / `{metadata.get('precision', {}).get('fp16')}`",
        "",
        "## Parameters",
        "",
        f"- Trainable params: `{params.get('trainable_params')}`",
        f"- Total params: `{params.get('total_params')}`",
        f"- Trainable ratio: `{format_number(params.get('trainable_ratio'))}`",
        "",
        "## Loss Summary",
        "",
        f"- Train loss: `{loss.get('loss', {})}`",
        f"- Eval loss: `{loss.get('eval_loss', {})}`",
        f"- Validation enabled: `{metadata.get('has_validation')}`",
        f"- Evaluation note: `{metadata.get('evaluation_note')}`",
        "",
        "## Outputs",
        "",
        f"- Adapter output path: `{metadata.get('adapter_output_dir')}`",
        f"- Merged model output path: `{metadata.get('merged_output_dir')}`",
        f"- Safety evaluation passed: `{metadata.get('safety_eval_passed', 'pending')}`",
        f"- Merged model load test passed: `{metadata.get('merged_model_load_test_passed', 'pending')}`",
        "",
    ]
    write_text(report_path, "\n".join(lines))


def train(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    logger = setup_logging("train_cpt_lora")
    training_cfg = config.get("training", {})
    dataset_dir = resolve_training_path(config.get("corpus", {}).get("prepared_dataset_dir"), "outputs/cpt_dataset")
    output_dir = resolve_training_path(training_cfg.get("output_dir"), "outputs/lora_adapter")
    merged_output_dir = resolve_training_path(training_cfg.get("merged_output_dir"), "outputs/merged_model")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Prepared dataset directory not found: {dataset_dir}")

    dataset = load_from_disk(str(dataset_dir))
    if "train" not in dataset or len(dataset["train"]) == 0:
        raise RuntimeError(f"Prepared dataset has no train split: {dataset_dir}")
    has_eval = "validation" in dataset and len(dataset["validation"]) > 0
    dataset_info = read_json(dataset_dir / "dataset_info.json", default={})
    if not has_eval:
        logger.info("Prepared dataset has no validation split. Trainer evaluation is disabled for full CPT coverage mode.")

    tokenizer = _load_tokenizer(config)
    model = _load_model(config, logger)
    model = _build_peft_model(model, config)
    trainable_cast = cast_trainable_parameters_to_fp32(model, logger)
    param_counts = _parameter_counts(model)
    logger.info(
        "Trainable params: %s / %s (%.6f)",
        param_counts["trainable_params"],
        param_counts["total_params"],
        param_counts["trainable_ratio"],
    )
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    bf16, fp16 = _resolve_training_precision(training_cfg)
    args_kwargs = _training_args_kwargs(config, output_dir, has_eval)
    training_args = TrainingArguments(**args_kwargs)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"] if has_eval else None,
        "data_collator": FullTokenLossCollator(tokenizer),
        "callbacks": [
            NonFiniteTrainingCallback(
                abort_on_nonfinite_grad_norm=bool(training_cfg.get("abort_on_nonfinite_grad_norm", True)),
                logger=logger,
            )
        ],
    }
    trainer_params = inspect.signature(Trainer).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    resume = training_cfg.get("resume_from_checkpoint")
    active_peft_cfg = _peft_config(config)
    logger.info("Starting PEFT CPT training. full_token_loss=True, chat_template_used=False")
    train_result = trainer.train(resume_from_checkpoint=resume if resume else None)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    trainer.save_state()

    config_snapshot = {k: v for k, v in config.items() if k != "_config_path"}
    save_yaml(output_dir / "config_snapshot.yaml", config_snapshot)
    copy_file(config_path, output_dir / "original_config.yaml")
    training_args_json = training_args.to_dict()
    write_json(output_dir / "training_args.json", training_args_json)
    log_history = trainer.state.log_history
    loss_summary = {
        "loss": summarize_loss_history(log_history, "loss"),
        "eval_loss": summarize_loss_history(log_history, "eval_loss"),
    }
    metadata = {
        "status": "completed",
        "base_model_name_or_path": config["base_model_name_or_path"],
        "dataset_dir": str(dataset_dir),
        "dataset": dataset_info,
        "training": training_cfg,
        "peft": active_peft_cfg,
        "dora": active_peft_cfg,
        "precision": {"bf16": bf16, "fp16": fp16},
        "trainable_parameter_cast": trainable_cast,
        "device": _resolve_training_device(training_cfg),
        "parameter_counts": param_counts,
        "train_result": train_result.metrics,
        "loss_summary": loss_summary,
        "has_validation": has_eval,
        "evaluation_note": "Validation enabled." if has_eval else "CPT full-coverage training is enabled; validation is disabled.",
        "adapter_output_dir": str(output_dir),
        "merged_output_dir": str(merged_output_dir),
        "full_token_loss": True,
        "assistant_only_loss": False,
        "chat_template_used": False,
        "created_at_utc": utc_now(),
    }
    write_json(output_dir / "training_metadata.json", metadata)
    write_training_report(resolve_training_path("outputs/reports/training_report.md", "outputs/reports/training_report.md"), metadata)
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a PEFT LoRA adapter for full-token CPT causal LM.")
    parser.add_argument("--config", default="configs/domain_post_training.yaml")
    parser.add_argument("--base_model_name_or_path", default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--device", default=None, help="Training device. Defaults to training.device.")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    logger = setup_logging("train_cpt_lora", args.verbose)
    config, config_path = load_config(args.config)
    config = _apply_cli_overrides(config, args)
    try:
        train(config, config_path)
    except RuntimeError as exc:
        message = short_error(exc)
        if "out of memory" in str(exc).lower():
            message += (
                ". CUDA OOM: enable load_in_4bit, lower LoRA rank, or inspect "
                "outputs/cpt_dataset/coverage_report.md for longest whole-document samples."
            )
        logger.error(message)
        logger.debug(traceback.format_exc())
        return 4
    except Exception as exc:
        logger.error(short_error(exc))
        logger.debug(traceback.format_exc())
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
