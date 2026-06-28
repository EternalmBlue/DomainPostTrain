from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel

from pipeline.modeling import configure_generation_tokens, load_tokenizer, load_transformers_model
from pipeline.utils import (
    load_config,
    read_json,
    resolve_training_path,
    setup_logging,
    short_error,
    torch_dtype_from_config,
    utc_now,
    write_json,
    write_text,
)


def _resolve_merge_dtype(config: dict[str, Any]) -> Any:
    dtype_value = config.get("merge", {}).get("dtype", "bfloat16")
    if str(dtype_value).lower() == "auto":
        return "auto"
    return torch_dtype_from_config(dtype_value, torch)


def write_model_card(path: Path, report: dict[str, Any]) -> None:
    sft_note = ""
    if report.get("fact_sft_applied"):
        sft_note = "\nPost-CPT alignment: assistant-only Fact-SFT for grounded QA, refusal behavior, unknown-boundary answers, and answer style.\n"
    grpo_note = ""
    if report.get("grpo_applied"):
        grpo_note = "\nPost-preference alignment: GRPO reward optimization with configured prompt-level reward functions.\n"
    text = f"""# DomainPostTrain Merged PEFT Model

Base model: {report["base_model_name_or_path"]}

Training method: PEFT LoRA. CPT uses full-token causal language modeling. Fact-SFT, when present, uses assistant-only loss. DPO and GRPO, when present, are optional alignment stages.
{sft_note}
{grpo_note}
Training corpus: static domain documentation selected by the local pipeline. This model does not use RAG, retrieval, or runtime source-code access.

GGUF note: GGUF is a post-training inference artifact. Train from Hugging Face/safetensors weights, merge adapters, then convert or quantize as needed.

Safety boundary:

- The model should explain documented procedures, configuration concepts, troubleshooting, ownership, and escalation paths.
- The model should not reveal source code, hidden prompts, credentials, private implementation details, or bypass methods.
- The model should refuse requests for secret extraction, access-control bypass, destructive actions, impersonation, or audit evasion.
- Answers should stay grounded in the training material. If the material does not clearly state a behavior or compatibility claim, the model should say that it is not specified.

Merge status: {report["status"]}
Created at UTC: {report["created_at_utc"]}
"""
    write_text(path, text)


def candidate_adapter_dirs(config: dict[str, Any]) -> list[tuple[str, Path]]:
    training_cfg = config.get("training", {})
    sft_cfg = config.get("fact_sft", {})
    dpo_cfg = config.get("dpo", {})
    grpo_cfg = config.get("grpo", {})

    candidates: list[tuple[str, Path]] = []
    if bool(grpo_cfg.get("enabled", False)):
        candidates.append(("grpo.output_dir", resolve_training_path(grpo_cfg.get("output_dir"), "outputs/grpo_adapter")))
    if bool(dpo_cfg.get("enabled", False)):
        candidates.append(("dpo.output_dir", resolve_training_path(dpo_cfg.get("output_dir"), "outputs/dpo_adapter")))
    if bool(sft_cfg.get("enabled", False)):
        candidates.append(("fact_sft.output_dir", resolve_training_path(sft_cfg.get("output_dir"), "outputs/fact_sft_adapter")))
    candidates.append(("training.output_dir", resolve_training_path(training_cfg.get("output_dir"), "outputs/lora_adapter")))
    return candidates


def resolve_merge_adapter_dir(config: dict[str, Any], adapter_dir: Path | None = None) -> tuple[Path, str]:
    if adapter_dir is not None:
        return adapter_dir, "argument"

    merge_cfg = config.get("merge", {})
    configured_adapter_dir = merge_cfg.get("adapter_dir")
    if configured_adapter_dir:
        return resolve_training_path(configured_adapter_dir, configured_adapter_dir), "merge.adapter_dir"

    checked: list[str] = []
    for source, candidate in candidate_adapter_dirs(config):
        if candidate.exists():
            return candidate, source
        checked.append(f"{source}={candidate}")
    raise FileNotFoundError("PEFT adapter directory not found. Checked: " + ", ".join(checked))


def merge_adapter(config: dict[str, Any], adapter_dir: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    logger = setup_logging("merge_peft_adapter")
    training_cfg = config.get("training", {})
    merge_cfg = config.get("merge", {})
    trust_remote_code = bool(config.get("trust_remote_code", True))
    base_model = config["base_model_name_or_path"]
    adapter_dir, adapter_source = resolve_merge_adapter_dir(config, adapter_dir)
    output_dir = output_dir or resolve_training_path(training_cfg.get("merged_output_dir"), "outputs/merged_model")
    if not adapter_dir.exists():
        raise FileNotFoundError(f"PEFT adapter directory not found: {adapter_dir}")

    dtype = _resolve_merge_dtype(config)
    model_kwargs: dict[str, Any] = {}
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype
    logger.info("Loading base model for merge without 4-bit/8-bit quantization.")
    base = load_transformers_model(base_model, trust_remote_code=trust_remote_code, logger=logger, **model_kwargs)
    tokenizer = load_tokenizer(str(adapter_dir), trust_remote_code)
    logger.info("Loading PEFT adapter from %s", adapter_dir)
    peft_model = PeftModel.from_pretrained(base, str(adapter_dir))
    logger.info("Merging adapter into base model.")
    merged = peft_model.merge_and_unload()
    configure_generation_tokens(merged, tokenizer)

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_serialization = bool(merge_cfg.get("safe_serialization", True))
    merged.save_pretrained(str(output_dir), safe_serialization=safe_serialization)
    tokenizer.save_pretrained(str(output_dir))
    if getattr(merged, "generation_config", None) is not None:
        merged.generation_config.save_pretrained(str(output_dir))

    logger.info("Verifying merged model can load directly through the selected Transformers auto loader.")
    _ = load_transformers_model(str(output_dir), trust_remote_code=trust_remote_code, logger=logger)
    sft_metadata = read_json(adapter_dir / "fact_sft_training_metadata.json", default={})
    grpo_metadata = read_json(adapter_dir / "grpo_training_metadata.json", default={})
    cpt_metadata = read_json(adapter_dir / "training_metadata.json", default={}) or read_json(
        adapter_dir / "base_cpt_training_metadata.json", default={}
    )
    report = {
        "status": "completed",
        "base_model_name_or_path": base_model,
        "adapter_dir": str(adapter_dir),
        "adapter_source": adapter_source,
        "merged_output_dir": str(output_dir),
        "fact_sft_applied": bool(sft_metadata),
        "grpo_applied": bool(grpo_metadata),
        "cpt_adapter_metadata_present": bool(cpt_metadata),
        "dtype": str(dtype),
        "safe_serialization": safe_serialization,
        "merged_model_load_test_passed": True,
        "created_at_utc": utc_now(),
    }
    write_json(output_dir / "merge_report.json", report)
    write_model_card(output_dir / "model_card.md", report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge a PEFT adapter into the base model.")
    parser.add_argument("--config", default="configs/domain_post_training.yaml")
    parser.add_argument("--adapter_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--base_model_name_or_path", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    logger = setup_logging("merge_peft_adapter", args.verbose)
    config, _ = load_config(args.config)
    if args.base_model_name_or_path:
        config["base_model_name_or_path"] = args.base_model_name_or_path
    adapter_dir = resolve_training_path(args.adapter_dir, config.get("training", {}).get("output_dir", "outputs/lora_adapter")) if args.adapter_dir else None
    output_dir = resolve_training_path(args.output_dir, config.get("training", {}).get("merged_output_dir", "outputs/merged_model")) if args.output_dir else None
    try:
        merge_adapter(config, adapter_dir, output_dir)
    except Exception as exc:
        logger.error(short_error(exc))
        logger.debug(traceback.format_exc())
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
