"""中文：用已 merge 的 Hugging Face 模型为 DPO JSONL 补全 rejected 回答。
使用时机：已有 prompt/chosen 偏好草稿，想用当前模型生成较弱 rejected 候选再人工审核时使用。

English: Fill rejected answers in a DPO JSONL file with generations from an already merged Hugging Face model.
Use it when you have prompt/chosen preference drafts and want weaker rejected candidates for human review.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPTS_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "_shared" / "bootstrap.py").is_file())
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _shared.bootstrap import bootstrap_project_root


TRAINING_ROOT = bootstrap_project_root(__file__)

from pipeline.cuda_bootstrap import ensure_pip_cuda_libraries_preferred

ensure_pip_cuda_libraries_preferred()

import torch
from transformers.utils import logging as hf_logging

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from run_inference import (  # noqa: E402
    DEFAULT_SYSTEM_PROMPT,
    _build_prompt,
    _clean_answer,
    _decode_generated,
    _load_config,
    _resolve_path,
    _select_device,
    _special_token_bans,
    _torch_dtype,
)
from pipeline.modeling import configure_generation_tokens, load_tokenizer, load_transformers_model  # noqa: E402


DEFAULT_INPUT = "data/dpo/dpo_placeholders_1000.jsonl"
DEFAULT_OUTPUT = "data/dpo/dpo_with_rejected_from_merged.jsonl"
ONLINE_OUTPUT_RULE_SOURCE_SUFFIX = "+infer._build_prompt(output_rule,enable_thinking=false)"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}.")
            rows.append(row)
    return rows


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
    tmp_path.replace(path)


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _status_for(row: dict[str, Any]) -> str:
    has_chosen = bool(str(row.get("chosen", "")).strip())
    has_rejected = bool(str(row.get("rejected", "")).strip())
    if has_chosen and has_rejected:
        return "ready_for_dpo"
    if has_rejected:
        return "needs_chosen"
    return "needs_chosen_rejected"


def _resolve_system_prompt(config: dict[str, Any], override: str | None, *, allow_default: bool) -> tuple[str, str]:
    if override:
        return override, f"argument{ONLINE_OUTPUT_RULE_SOURCE_SUFFIX}"
    configured = str(config.get("fact_sft", {}).get("system_prompt") or "").strip()
    if configured:
        return configured, f"config.fact_sft.system_prompt{ONLINE_OUTPUT_RULE_SOURCE_SUFFIX}"
    if allow_default:
        return DEFAULT_SYSTEM_PROMPT, f"infer.DEFAULT_SYSTEM_PROMPT{ONLINE_OUTPUT_RULE_SOURCE_SUFFIX}"
    raise RuntimeError(
        "Config does not define fact_sft.system_prompt. "
        "This script is intended to mimic the online Flask inference prompt; "
        "set fact_sft.system_prompt in the config or pass --allow_default_system_prompt explicitly."
    )


def _load_rows_for_run(input_path: Path, output_path: Path, *, in_place: bool, restart: bool) -> tuple[list[dict[str, Any]], str]:
    if not input_path.exists():
        raise FileNotFoundError(f"DPO input JSONL not found: {input_path}")
    if not in_place and output_path.exists() and not restart:
        return _read_jsonl(output_path), "output_resume"
    return _read_jsonl(input_path), "input"


def _generate_answers(
    *,
    model,
    tokenizer,
    input_device,
    texts: list[str],
    system_prompt: str,
    raw_prompt: bool,
    max_new_tokens: int,
    min_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    no_empty_retry: bool,
    clean_answer: bool,
) -> list[tuple[str, str]]:
    if not texts:
        return []

    prompts = [_build_prompt(tokenizer, text, raw_prompt, system_prompt) for text in texts]
    previous_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(input_device)
    finally:
        tokenizer.padding_side = previous_padding_side

    do_sample = temperature > 0
    generate_kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": max(0, int(no_repeat_ngram_size)),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if min_new_tokens > 0:
        generate_kwargs["min_new_tokens"] = int(min_new_tokens)

    with torch.inference_mode():
        output_ids = model.generate(**generate_kwargs)
    prompt_width = inputs["input_ids"].shape[-1]
    raw_answers = [_decode_generated(tokenizer, generated) for generated in output_ids[:, prompt_width:]]

    retry_indices = [
        index for index, raw_answer in enumerate(raw_answers) if not raw_answer and not no_empty_retry and max_new_tokens > 1
    ]
    if retry_indices:
        retry_prompts = [prompts[index] for index in retry_indices]
        previous_padding_side = getattr(tokenizer, "padding_side", "right")
        tokenizer.padding_side = "left"
        try:
            retry_inputs = tokenizer(retry_prompts, return_tensors="pt", padding=True).to(input_device)
        finally:
            tokenizer.padding_side = previous_padding_side
        retry_kwargs: dict[str, Any] = {
            **retry_inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else None,
            "top_p": top_p if do_sample else None,
            "repetition_penalty": min(repetition_penalty, 1.05),
            "no_repeat_ngram_size": 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "bad_words_ids": _special_token_bans(tokenizer),
            "use_cache": True,
        }
        retry_kwargs["min_new_tokens"] = min(max(8, int(min_new_tokens)), max_new_tokens)
        with torch.inference_mode():
            output_ids = model.generate(**retry_kwargs)
        retry_prompt_width = retry_inputs["input_ids"].shape[-1]
        retry_raw_answers = [_decode_generated(tokenizer, generated) for generated in output_ids[:, retry_prompt_width:]]
        for original_index, retry_raw_answer in zip(retry_indices, retry_raw_answers, strict=True):
            raw_answers[original_index] = retry_raw_answer

    answers: list[tuple[str, str]] = []
    for raw_answer in raw_answers:
        answer = _clean_answer(raw_answer) if clean_answer else raw_answer.strip()
        answers.append((answer.strip(), raw_answer.strip()))
    return answers


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fill DPO JSONL rejected fields with answers generated by an already merged HF model."
    )
    parser.add_argument("--config", default="configs/domain_post_training.yaml")
    parser.add_argument("--model_path", default="outputs/merged_model")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input DPO placeholder JSONL.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSONL. Ignored when --in_place is set.")
    parser.add_argument("--in_place", action="store_true", help="Overwrite the input JSONL atomically as rows are filled.")
    parser.add_argument("--restart", action="store_true", help="Ignore an existing output file and restart from --input.")
    parser.add_argument("--overwrite_rejected", action="store_true", help="Regenerate rows even when rejected is already filled.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows to generate in this run.")
    parser.add_argument("--start_index", type=int, default=0, help="Zero-based row index to start from.")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of DPO prompts to generate in one model batch.")
    parser.add_argument("--save_every", type=int, default=1, help="Save progress every N generated rows.")
    parser.add_argument("--report_path", default="data/dpo/dpo_rejected_fill_report.json")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, cpu, or auto.")
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--min_new_tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=None)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=6)
    parser.add_argument("--raw_prompt", action="store_true", help="Debug only: bypass system/user chat prompting.")
    parser.add_argument("--system_prompt", default=None, help="Override the config system prompt. Debug only.")
    parser.add_argument(
        "--allow_default_system_prompt",
        action="store_true",
        help="Allow fallback to infer.DEFAULT_SYSTEM_PROMPT if the config has no fact_sft.system_prompt.",
    )
    parser.add_argument("--no_clean_answer", action="store_true")
    parser.add_argument(
        "--no_raw_fallback_on_empty_clean",
        action="store_true",
        help=(
            "When cleaned output is empty but raw model output is non-empty, keep the row failed instead of "
            "using raw output as rejected."
        ),
    )
    parser.add_argument("--no_empty_retry", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true", default=None)
    parser.add_argument("--no_trust_remote_code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--dry_run", action="store_true", help="Validate paths and row counts without loading the model.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if not args.verbose:
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()

    config = _load_config(args.config)
    eval_cfg = config.get("eval", {})
    input_path = _resolve_path(args.input, DEFAULT_INPUT)
    output_path = input_path if args.in_place else _resolve_path(args.output, DEFAULT_OUTPUT)
    report_path = _resolve_path(args.report_path, "data/dpo/dpo_rejected_fill_report.json")
    rows, row_source = _load_rows_for_run(input_path, output_path, in_place=args.in_place, restart=args.restart)

    total_rows = len(rows)
    already_filled = sum(1 for row in rows if str(row.get("rejected", "")).strip())
    candidates = [
        index
        for index, row in enumerate(rows)
        if index >= max(0, args.start_index) and (args.overwrite_rejected or not str(row.get("rejected", "")).strip())
    ]
    if args.limit is not None:
        candidates = candidates[: max(0, args.limit)]
    system_prompt, system_prompt_source = _resolve_system_prompt(
        config,
        args.system_prompt,
        allow_default=bool(args.allow_default_system_prompt),
    )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "row_source": row_source,
                    "total_rows": total_rows,
                    "already_filled_rejected": already_filled,
                    "rows_to_generate": len(candidates),
                    "batch_size": max(1, int(args.batch_size)),
                    "system_prompt_source": system_prompt_source,
                    "raw_prompt": bool(args.raw_prompt),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    model_path = _resolve_path(args.model_path, "outputs/merged_model")
    if not model_path.exists():
        raise FileNotFoundError(f"Merged model directory not found: {model_path}")

    trust_remote_code = bool(config.get("trust_remote_code", True)) if args.trust_remote_code is None else args.trust_remote_code
    max_new_tokens = int(args.max_new_tokens or min(int(eval_cfg.get("max_new_tokens", 512)), 256))
    top_p = float(args.top_p if args.top_p is not None else eval_cfg.get("top_p", 0.9))
    repetition_penalty = float(
        args.repetition_penalty if args.repetition_penalty is not None else eval_cfg.get("repetition_penalty", 1.05)
    )

    device = _select_device(args.device)
    dtype = _torch_dtype(args.dtype)
    model_kwargs: dict[str, Any] = {}
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype
    if device == "auto":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = {"": device}

    tokenizer = load_tokenizer(str(model_path), trust_remote_code)
    model = load_transformers_model(str(model_path), trust_remote_code=trust_remote_code, **model_kwargs)
    configure_generation_tokens(model, tokenizer)
    model.eval()
    input_device = next(model.parameters()).device

    generated_count = 0
    failed_count = 0
    raw_fallback_count = 0
    skipped_empty_prompt = 0
    last_saved_generated = 0
    batch_size = max(1, int(args.batch_size))
    started_at = time.time()
    position = 0
    while position < len(candidates):
        batch_indices: list[int] = []
        batch_prompts: list[str] = []
        while position < len(candidates) and len(batch_indices) < batch_size:
            row_index = candidates[position]
            position += 1
            row = rows[row_index]
            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                row["status"] = "generation_failed_empty_prompt"
                skipped_empty_prompt += 1
                failed_count += 1
                continue
            batch_indices.append(row_index)
            batch_prompts.append(prompt)

        if not batch_indices:
            if position == len(candidates):
                _write_jsonl_atomic(output_path, rows)
            continue

        batch_results = _generate_answers(
            model=model,
            tokenizer=tokenizer,
            input_device=input_device,
            texts=batch_prompts,
            system_prompt=system_prompt,
            raw_prompt=bool(args.raw_prompt),
            max_new_tokens=max_new_tokens,
            min_new_tokens=int(args.min_new_tokens),
            temperature=float(args.temperature),
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=int(args.no_repeat_ngram_size),
            no_empty_retry=bool(args.no_empty_retry),
            clean_answer=not bool(args.no_clean_answer),
        )

        for row_index, (answer, raw_answer) in zip(batch_indices, batch_results, strict=True):
            row = rows[row_index]
            used_raw_fallback = False
            if not answer and raw_answer and not args.no_raw_fallback_on_empty_clean and not args.no_clean_answer:
                answer = raw_answer.strip()
                used_raw_fallback = True

            if answer:
                row["rejected"] = answer
                row["status"] = _status_for(row)
                row["rejected_source"] = "raw_fallback_after_empty_clean" if used_raw_fallback else "merged_model_generation"
                row["rejected_model_path"] = str(model_path)
                row["rejected_system_prompt_source"] = system_prompt_source
                row["rejected_prompt_mode"] = "raw" if args.raw_prompt else "chat_template_or_fallback"
                row["rejected_clean_answer_empty"] = used_raw_fallback
                row["rejected_generated_at_utc"] = _utc_now()
                generated_count += 1
                if used_raw_fallback:
                    raw_fallback_count += 1
            else:
                row["status"] = "generation_failed_empty_answer"
                row["rejected_raw_empty"] = raw_answer == ""
                if raw_answer:
                    row["rejected_raw_preview"] = raw_answer[:500]
                failed_count += 1

        save_every = max(1, int(args.save_every))
        if generated_count - last_saved_generated >= save_every or position == len(candidates):
            _write_jsonl_atomic(output_path, rows)
            last_saved_generated = generated_count
            elapsed = max(time.time() - started_at, 1e-6)
            generated_per_second = generated_count / elapsed
            print(
                f"[{position}/{len(candidates)}] row={batch_indices[-1] + 1} batch={len(batch_indices)} "
                f"generated={generated_count} failed={failed_count} raw_fallback={raw_fallback_count} "
                f"speed={generated_per_second:.3f}/s elapsed={elapsed:.1f}s"
            )

    _write_jsonl_atomic(output_path, rows)
    report = {
        "status": "completed" if failed_count == 0 else "completed_with_failures",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "row_source": row_source,
        "model_path": str(model_path),
        "system_prompt_source": system_prompt_source,
        "raw_prompt": bool(args.raw_prompt),
        "total_rows": total_rows,
        "already_filled_rejected_at_start": already_filled,
        "rows_selected": len(candidates),
        "batch_size": batch_size,
        "generated_rejected": generated_count,
        "raw_fallback_after_empty_clean": raw_fallback_count,
        "failed": failed_count,
        "skipped_empty_prompt": skipped_empty_prompt,
        "remaining_empty_rejected": sum(1 for row in rows if not str(row.get("rejected", "")).strip()),
        "created_at_utc": _utc_now(),
    }
    _write_report(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if failed_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
