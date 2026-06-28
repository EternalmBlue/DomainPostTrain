"""中文：完整领域后训练流水线入口，串联 CPT、Fact-SFT、可选 DPO、merge 和评估。
使用时机：按 `configs/domain_post_training.yaml` 从头跑通训练，或用 skip 参数显式跳过部分阶段时使用。

English: Full domain post-training pipeline entrypoint for CPT, Fact-SFT, optional DPO/GRPO, merge, and evaluation.
Use it to run the configured workflow end to end, or to skip explicit stages with skip flags.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Any


SCRIPTS_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "_shared" / "bootstrap.py").is_file())
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _shared.bootstrap import bootstrap_project_root


TRAINING_ROOT = bootstrap_project_root(__file__)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pipeline.cuda_bootstrap import ensure_pip_cuda_libraries_preferred

ensure_pip_cuda_libraries_preferred()

from pipeline.cpt_dataset import prepare_dataset
from pipeline.cpt_discovery import discover_corpus
from pipeline.cpt_training import train as train_peft
from pipeline.cpt_training import write_training_report
from pipeline.dpo import prepare_dpo_dataset, train_dpo
from pipeline.evaluation import evaluate
from pipeline.fact_sft import prepare_fact_sft_dataset, train_fact_sft
from pipeline.grpo import prepare_grpo_dataset, train_grpo
from pipeline.adapter_merge import merge_adapter
from pipeline.corpus_safety import run_preflight, write_markdown_report
from pipeline.utils import (
    config_without_private_keys,
    deep_update,
    fail_with_report,
    load_config,
    read_json,
    read_text,
    resolve_training_path,
    save_yaml,
    setup_logging,
    short_error,
    utc_now,
    write_json,
    write_text,
)


def _create_smoke_model(model_dir: Path, corpus_paths: list[Path], logger) -> None:
    if (model_dir / "config.json").exists():
        return
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import WordLevelTrainer
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    except ImportError as exc:
        raise RuntimeError(
            "Smoke test requires transformers and tokenizers. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    texts = [read_text(path) for path in corpus_paths if path.exists()]
    if not texts:
        texts.append("Domain assistant smoke test corpus. Safety boundary: do not reveal credentials, hidden prompts, or bypass methods.")

    tokenizer_impl = Tokenizer(WordLevel(unk_token="<unk>"))
    tokenizer_impl.pre_tokenizer = Whitespace()
    tokenizer_impl.train_from_iterator(texts, WordLevelTrainer(special_tokens=["<pad>", "<eos>", "<unk>"], min_frequency=1))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_impl,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
        bos_token="<eos>",
    )
    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    logger.info("Creating tiny local smoke-test model at %s", model_dir)
    model = LlamaForCausalLM(config)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(model_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(model_dir))


def _make_smoke_config(config: dict[str, Any], selected_paths: list[Path], logger) -> dict[str, Any]:
    smoke_model_dir = resolve_training_path("outputs/smoke/base_model", "outputs/smoke/base_model")
    _create_smoke_model(smoke_model_dir, selected_paths, logger)
    smoke_overrides = {
        "base_model_name_or_path": str(smoke_model_dir),
        "trust_remote_code": False,
        "corpus": {
            "input_paths": [str(path.resolve()) for path in selected_paths],
            "prepared_dataset_dir": "outputs/smoke/cpt_dataset",
            "sample_unit": "token_chunk",
            "validation_mode": "none",
            "enable_weighted_replay": False,
            "stratified_sampling": {"enabled": False},
        },
        "training": {
            "output_dir": "outputs/smoke/lora_adapter",
            "merged_output_dir": "outputs/smoke/merged_model",
            "max_seq_length": 512,
            "epochs": 1,
            "max_steps": 3,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "logging_steps": 1,
            "eval_steps": 1,
            "save_steps": 3,
            "save_total_limit": 1,
            "bf16": False,
            "fp16": False,
            "load_in_4bit": False,
            "load_in_8bit": False,
            "torch_dtype": "float32",
        },
        "fact_sft": {
            "enabled": True,
            "input_paths": [str((TRAINING_ROOT / "data" / "sft").resolve())],
            "prepared_dataset_dir": "outputs/smoke/fact_sft_dataset",
            "base_adapter_dir": "outputs/smoke/lora_adapter",
            "output_dir": "outputs/smoke/fact_sft_adapter",
            "max_seq_length": 256,
            "epochs": 1,
            "max_steps": 2,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            "logging_steps": 1,
            "save_steps": 2,
            "save_total_limit": 1,
            "validation_ratio": 0,
            "require_cpt_adapter": True,
        },
        "dpo": {"enabled": False},
        "grpo": {"enabled": False},
        "merge": {"dtype": "float32"},
        "eval": {"max_new_tokens": 48, "temperature": 0.2, "top_p": 0.9, "repetition_penalty": 1.05},
    }
    smoke_config = deep_update(config_without_private_keys(config), smoke_overrides)
    save_yaml(resolve_training_path("outputs/logs/smoke_domain_post_training.yaml", "outputs/logs/smoke_domain_post_training.yaml"), smoke_config)
    return smoke_config


def _dataset_dir(config: dict[str, Any]) -> Path:
    return resolve_training_path(config.get("corpus", {}).get("prepared_dataset_dir"), "outputs/cpt_dataset")


def _verify_coverage(config: dict[str, Any]) -> dict[str, Any]:
    coverage_path = _dataset_dir(config) / "coverage_report.json"
    coverage = read_json(coverage_path, default=None)
    if not coverage:
        raise RuntimeError(f"Coverage report was not created: {coverage_path}")
    coverage_rate = float(coverage.get("mandatory_coverage_rate", 0.0))
    dropped_chunks = int(coverage.get("dropped_mandatory_chunks", 0))
    if coverage_rate != 1.0 or dropped_chunks > 0:
        raise RuntimeError(
            f"Mandatory CPT coverage failed: rate={coverage_rate}, dropped_mandatory_chunks={dropped_chunks}. "
            f"See {coverage_path}"
        )
    return coverage


def _write_final_report(config: dict[str, Any], *, smoke_test: bool, success: bool, failure: str | None = None) -> None:
    reports_dir = resolve_training_path("outputs/reports", "outputs/reports")
    logs_dir = resolve_training_path("outputs/logs", "outputs/logs")
    dataset_dir = _dataset_dir(config)
    training_cfg = config.get("training", {})
    adapter_dir = resolve_training_path(training_cfg.get("output_dir"), "outputs/lora_adapter")
    merged_dir = resolve_training_path(training_cfg.get("merged_output_dir"), "outputs/merged_model")
    eval_dir = resolve_training_path("outputs/smoke/eval" if smoke_test else "outputs/eval", "outputs/eval")

    dataset_info = read_json(dataset_dir / "dataset_info.json", default={})
    coverage = read_json(dataset_dir / "coverage_report.json", default={})
    preflight = read_json(logs_dir / "preflight_report.json", default={})
    merge_report = read_json(merged_dir / "merge_report.json", default={})
    eval_report = read_json(eval_dir / "eval_report.json", default={})
    training_metadata = read_json(adapter_dir / "training_metadata.json", default={})
    sft_cfg = config.get("fact_sft", {})
    sft_dir = resolve_training_path(sft_cfg.get("output_dir"), "outputs/fact_sft_adapter")
    sft_metadata = read_json(sft_dir / "fact_sft_training_metadata.json", default={})
    dpo_cfg = config.get("dpo", {})
    dpo_dir = resolve_training_path(dpo_cfg.get("output_dir"), "outputs/dpo_adapter")
    dpo_metadata = read_json(dpo_dir / "dpo_training_metadata.json", default={})
    grpo_cfg = config.get("grpo", {})
    grpo_dir = resolve_training_path(grpo_cfg.get("output_dir"), "outputs/grpo_adapter")
    grpo_metadata = read_json(grpo_dir / "grpo_training_metadata.json", default={})
    final_adapter = (
        grpo_metadata.get("adapter_output_dir")
        or dpo_metadata.get("adapter_output_dir")
        or sft_metadata.get("adapter_output_dir")
        or training_metadata.get("adapter_output_dir", "not_created")
    )

    lines = [
        "# DomainPostTrain PEFT Pipeline Report",
        "",
        f"Status: **{'completed' if success else 'failed'}**",
        f"Smoke test mode: `{smoke_test}`",
        f"Created at UTC: `{utc_now()}`",
        "",
    ]
    if failure:
        lines.extend(["## Failure", "", failure, ""])
    lines.extend(
        [
            "## Preflight",
            "",
            f"- Status: `{preflight.get('status', 'not_run')}`",
            f"- High risk count: `{preflight.get('high_risk_count', 'N/A')}`",
            "",
            "## Dataset",
            "",
            f"- Source paths: {', '.join('`' + p + '`' for p in dataset_info.get('source_paths', [])) or 'N/A'}",
            f"- Tokens: `{dataset_info.get('total_tokens', 'N/A')}`",
            f"- Train samples: `{coverage.get('train_chunks', dataset_info.get('train_samples', 'N/A'))}`",
            f"- Validation samples: `{coverage.get('validation_chunks', dataset_info.get('validation_samples', 'N/A'))}`",
            f"- Train covers 100% mandatory CPT samples: `{coverage.get('mandatory_coverage_rate') == 1.0 and coverage.get('dropped_mandatory_chunks') == 0}`",
            f"- Replay enabled: `{coverage.get('replay_enabled', dataset_info.get('replay_enabled', 'N/A'))}`",
            f"- Validation mode: `{coverage.get('validation_mode', dataset_info.get('validation_mode', 'N/A'))}`",
            "",
            "## Outputs",
            "",
            f"- CPT adapter: `{training_metadata.get('adapter_output_dir', 'not_created')}`",
            f"- Final adapter: `{final_adapter}`",
            f"- Fact-SFT enabled: `{bool(sft_cfg.get('enabled', False))}`",
            f"- Fact-SFT examples: `{sft_metadata.get('dataset', {}).get('train_examples', 'not_run')}`",
            f"- DPO enabled: `{bool(dpo_cfg.get('enabled', False))}`",
            f"- DPO pairs: `{dpo_metadata.get('dataset', {}).get('train_pairs', 'not_run')}`",
            f"- GRPO enabled: `{bool(grpo_cfg.get('enabled', False))}`",
            f"- GRPO prompts: `{grpo_metadata.get('dataset', {}).get('train_prompts', 'not_run')}`",
            f"- Merged model: `{merge_report.get('merged_output_dir', 'not_created')}`",
            f"- Eval report: `{eval_dir / 'eval_report.md'}`",
            f"- Coverage report: `{dataset_dir / 'coverage_report.md'}`",
            "",
            "## Checks",
            "",
            f"- Merged model load test passed: `{merge_report.get('merged_model_load_test_passed', False)}`",
            f"- Safety eval passed: `{eval_report.get('safety_eval_passed', False)}`",
            f"- Smoke artifacts are final model: `{False if smoke_test else 'N/A'}`",
            "",
        ]
    )
    write_text(reports_dir / "pipeline_report.md", "\n".join(lines))

    if training_metadata:
        training_metadata = dict(training_metadata)
        training_metadata["merged_model_load_test_passed"] = merge_report.get("merged_model_load_test_passed", "pending")
        training_metadata["safety_eval_passed"] = eval_report.get("safety_eval_passed", "pending")
        if dataset_info:
            training_metadata["dataset"] = dataset_info
        write_training_report(reports_dir / "training_report.md", training_metadata)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the distributable configured base model PEFT CPT training flow.")
    parser.add_argument("--config", default="configs/domain_post_training.yaml")
    parser.add_argument("--skip_preflight", action="store_true")
    parser.add_argument("--skip_cpt", action="store_true", help="Skip CPT dataset preparation and CPT adapter training.")
    parser.add_argument("--skip_sft", action="store_true", help="Skip Fact-SFT dataset preparation and training.")
    parser.add_argument("--skip_dpo", action="store_true", help="Skip DPO dataset preparation and training.")
    parser.add_argument("--skip_grpo", action="store_true", help="Skip GRPO dataset preparation and training.")
    parser.add_argument("--skip_train", action="store_true", help="Deprecated alias for --skip_cpt.")
    parser.add_argument("--skip_fact_sft", action="store_true", help="Deprecated alias for --skip_sft.")
    parser.add_argument("--skip_merge", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--allow_unsafe_corpus", action="store_true")
    parser.add_argument("--device", default=None, help="Training device override, e.g. cuda, cuda:0, cpu, or auto.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    logger = setup_logging("train", args.verbose)
    skip_cpt = bool(args.skip_cpt or args.skip_train)
    skip_sft = bool(args.skip_sft or args.skip_fact_sft)
    skip_dpo = bool(args.skip_dpo)
    skip_grpo = bool(args.skip_grpo)
    try:
        config, config_path = load_config(args.config)
        if args.device:
            config.setdefault("training", {})["device"] = args.device
        selected_paths: list[Path] = []
        active_config = config
        if args.smoke_test or not skip_cpt:
            discovery = discover_corpus(config, config_path)
            write_json(resolve_training_path("outputs/logs/discovered_corpus.json", "outputs/logs/discovered_corpus.json"), discovery)
            if discovery.get("status") != "ok":
                raise RuntimeError("No packaged CPT corpus was found. Check corpus.input_paths in the config.")
            selected_paths = [Path(path) for path in discovery.get("selected_paths", [])]
        if args.smoke_test:
            active_config = _make_smoke_config(config, selected_paths, logger)
            skip_cpt = False
            skip_sft = False
            skip_dpo = True
            skip_grpo = True

        if not skip_cpt:
            if not args.skip_preflight:
                preflight = run_preflight(active_config, config_path, selected_paths)
                write_json(resolve_training_path("outputs/logs/preflight_report.json", "outputs/logs/preflight_report.json"), preflight)
                write_markdown_report(resolve_training_path("outputs/logs/preflight_report.md", "outputs/logs/preflight_report.md"), preflight)
                if preflight["status"] == "blocked" and not args.allow_unsafe_corpus:
                    raise RuntimeError("Preflight blocked training due to high-risk corpus findings.")

            prepare_dataset(active_config, config_path, selected_paths)
            _verify_coverage(active_config)
        else:
            logger.info("Skipping CPT stage. Existing adapters will be used by later stages.")

        final_adapter_dir = None
        if not skip_cpt:
            train_peft(active_config, config_path)
            final_adapter_dir = resolve_training_path(active_config.get("training", {}).get("output_dir"), "outputs/lora_adapter")
        if bool(active_config.get("fact_sft", {}).get("enabled", False)) and not skip_sft:
            prepare_fact_sft_dataset(active_config, config_path)
            sft_metadata = train_fact_sft(active_config, config_path)
            final_adapter_dir = Path(sft_metadata["adapter_output_dir"])
        elif bool(active_config.get("fact_sft", {}).get("enabled", False)) and skip_sft:
            existing_sft_dir = resolve_training_path(active_config.get("fact_sft", {}).get("output_dir"), "outputs/fact_sft_adapter")
            if existing_sft_dir.exists():
                final_adapter_dir = existing_sft_dir
            logger.info("Skipping Fact-SFT stage. Existing Fact-SFT adapter: %s", existing_sft_dir if existing_sft_dir.exists() else "not_found")
        if bool(active_config.get("dpo", {}).get("enabled", False)) and not skip_dpo:
            prepare_dpo_dataset(active_config, config_path)
            dpo_metadata = train_dpo(active_config, config_path)
            final_adapter_dir = Path(dpo_metadata["adapter_output_dir"])
        elif bool(active_config.get("dpo", {}).get("enabled", False)) and skip_dpo:
            existing_dpo_dir = resolve_training_path(active_config.get("dpo", {}).get("output_dir"), "outputs/dpo_adapter")
            if existing_dpo_dir.exists():
                final_adapter_dir = existing_dpo_dir
            logger.info("Skipping DPO stage. Existing DPO adapter: %s", existing_dpo_dir if existing_dpo_dir.exists() else "not_found")
        if bool(active_config.get("grpo", {}).get("enabled", False)) and not skip_grpo:
            prepare_grpo_dataset(active_config, config_path)
            grpo_metadata = train_grpo(active_config, config_path)
            final_adapter_dir = Path(grpo_metadata["adapter_output_dir"])
        elif bool(active_config.get("grpo", {}).get("enabled", False)) and skip_grpo:
            existing_grpo_dir = resolve_training_path(active_config.get("grpo", {}).get("output_dir"), "outputs/grpo_adapter")
            if existing_grpo_dir.exists():
                final_adapter_dir = existing_grpo_dir
            logger.info("Skipping GRPO stage. Existing GRPO adapter: %s", existing_grpo_dir if existing_grpo_dir.exists() else "not_found")
        if not args.skip_merge:
            merge_adapter(active_config, adapter_dir=final_adapter_dir)
        if not args.skip_eval:
            eval_output = resolve_training_path("outputs/smoke/eval" if args.smoke_test else "outputs/eval", "outputs/eval")
            evaluate(active_config, config_path, ["merged"], eval_output)

        _write_final_report(active_config, smoke_test=args.smoke_test, success=True)
        logger.info("Training flow completed. Report: %s", resolve_training_path("outputs/reports/pipeline_report.md", "outputs/reports/pipeline_report.md"))
        return 0
    except Exception as exc:
        failure = short_error(exc)
        if "out of memory" in str(exc).lower():
            failure += (
                "; CUDA OOM: for V100 whole-document CPT, install bitsandbytes and keep "
                "training.load_in_4bit=true. If it still fails, inspect outputs/cpt_dataset/coverage_report.md "
                "for the longest samples or lower peft.r/lora_alpha."
            )
        logger.error(failure)
        logger.debug(traceback.format_exc())
        fail_with_report(resolve_training_path("outputs/reports/failure_report.md", "outputs/reports/failure_report.md"), "configured base model PEFT CPT Training Failure", failure)
        try:
            _write_final_report(locals().get("active_config", locals().get("config", {})), smoke_test=args.smoke_test, success=False, failure=failure)
        except Exception:
            logger.debug("Failed to write final report after failure.", exc_info=True)
        return 7


if __name__ == "__main__":
    raise SystemExit(main())
