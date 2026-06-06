from __future__ import annotations

import argparse
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from pipeline.cpt_discovery import discover_corpus
from pipeline.corpus_safety import has_safety_boundary
from pipeline.utils import (
    SAFETY_PREAMBLE,
    clean_text_noise,
    load_config,
    normalize_path_for_report,
    parse_nullable_int,
    read_text,
    resolve_model_name_or_path,
    resolve_input_path,
    resolve_training_path,
    save_yaml,
    setup_logging,
    utc_now,
    write_json,
    write_text,
)


VALID_SUFFIXES = {".md", ".markdown", ".txt"}
DEFAULT_EXCLUDES = [
    "training_audit_not_for_training",
    ".git",
    "src",
    "build",
    "target",
    "outputs",
    "reports",
]
COVERAGE_SENTENCE = "CPT train dataset includes all mandatory corpus samples. No mandatory sample was removed for validation."
SUPPORTED_SAMPLE_UNITS = {"document", "section", "token_chunk"}

CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("model_spec", ["model_spec", "model specification", "assistant specification", "role", "answer style"]),
    ("commands", ["command", "commands", "cli", "workflow"]),
    ("permissions", ["permission", "permissions", "access", "approval"]),
    ("configuration", ["config", "configuration", "settings", "schema", "contract"]),
    ("database_reload", ["database", "storage", "cache", "reload", "restart"]),
    ("troubleshooting", ["faq", "troubleshooting", "issue", "incident", "error", "runbook"]),
    ("safety", ["safety", "boundary", "secret", "credential", "bypass", "audit"]),
    ("unknowns", ["unknown", "unknowns", "not specified", "non-guarantees", "non_guarantees"]),
    ("runtime_behavior", ["behavior", "listener", "event", "workflow", "routing"]),
    ("public_api", ["api", "developer", "public api", "integration"]),
]


def _load_tokenizer(base_model_name_or_path: str, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_model_name_or_path(base_model_name_or_path),
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({"eos_token": "</s>", "pad_token": "</s>"})
        else:
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _is_excluded(path: Path, exclude_fragments: list[str]) -> bool:
    resolved = str(path.resolve()).replace("\\", "/").lower()
    parts = [part.lower() for part in path.resolve().parts]
    for fragment in exclude_fragments:
        needle = fragment.replace("\\", "/").strip("/").lower()
        if not needle:
            continue
        if "/" in needle and needle in resolved:
            return True
        if needle in parts:
            return True
        if needle in resolved and needle in {"training_audit_not_for_training"}:
            return True
    return False


def _collect_source_files(paths: list[Path], exclude_fragments: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = raw_path.resolve()
        if not path.exists() or _is_excluded(path, exclude_fragments):
            continue
        if path.is_file():
            if path.suffix.lower() in VALID_SUFFIXES and path not in files:
                files.append(path)
            continue
        for candidate in sorted(path.rglob("*")):
            if (
                candidate.is_file()
                and candidate.suffix.lower() in VALID_SUFFIXES
                and not _is_excluded(candidate, exclude_fragments)
                and candidate.resolve() not in files
            ):
                files.append(candidate.resolve())
    return files


def _selected_input_paths(config: dict[str, Any], config_path: Path, input_paths: list[Path] | None) -> list[Path]:
    if input_paths:
        return input_paths
    corpus_cfg = config.get("corpus", {})
    configured_paths = [resolve_input_path(path, config_path) for path in corpus_cfg.get("input_paths", [])]
    if configured_paths:
        return configured_paths
    discovered = discover_corpus(config, config_path)
    return [Path(path) for path in discovered.get("selected_paths", [])]


def _split_markdown_sections(text: str, default_heading: str, enabled: bool) -> list[dict[str, str]]:
    if not enabled:
        return [{"heading": default_heading, "text": text}]
    sections: list[dict[str, str]] = []
    heading = default_heading
    buffer: list[str] = []
    heading_re = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
    for line in text.splitlines(keepends=True):
        match = heading_re.match(line.rstrip("\n"))
        if match:
            if buffer:
                section_text = "".join(buffer).strip()
                if section_text:
                    sections.append({"heading": heading, "text": section_text + "\n"})
            heading = match.group(2).strip()
            buffer = [line]
        else:
            buffer.append(line)
    if buffer:
        section_text = "".join(buffer).strip()
        if section_text:
            sections.append({"heading": heading, "text": section_text + "\n"})
    return sections or [{"heading": default_heading, "text": text}]


def _resolve_sample_unit(corpus_cfg: dict[str, Any]) -> str:
    raw = str(corpus_cfg.get("sample_unit", "token_chunk")).strip().lower().replace("-", "_")
    aliases = {
        "chunk": "token_chunk",
        "token_chunks": "token_chunk",
        "token": "token_chunk",
        "doc": "document",
        "file": "document",
        "full_text": "document",
        "fulltext": "document",
        "markdown_section": "section",
    }
    sample_unit = aliases.get(raw, raw)
    if sample_unit not in SUPPORTED_SAMPLE_UNITS:
        supported = ", ".join(sorted(SUPPORTED_SAMPLE_UNITS))
        raise ValueError(f"Unsupported corpus.sample_unit={corpus_cfg.get('sample_unit')!r}. Use one of: {supported}.")
    return sample_unit


def _stratified_sampling_cfg(corpus_cfg: dict[str, Any]) -> dict[str, Any]:
    raw_cfg = corpus_cfg.get("stratified_sampling", {}) or {}
    if not isinstance(raw_cfg, dict):
        raise ValueError("corpus.stratified_sampling must be a mapping when provided.")
    return dict(raw_cfg)


def _stratified_sampling_enabled(corpus_cfg: dict[str, Any]) -> bool:
    sampling_cfg = _stratified_sampling_cfg(corpus_cfg)
    return bool(sampling_cfg.get("enabled", corpus_cfg.get("enable_weighted_replay", False)))


def _classify_section(source_path: Path | str, heading: str) -> str:
    text = f"{source_path} {heading}".lower()
    for category, needles in CATEGORY_RULES:
        if any(needle.lower() in text for needle in needles):
            return category
    return "main_textbook"


def _chunk_token_ids(token_ids: list[int], max_seq_length: int, min_chunk_tokens: int) -> list[list[int]]:
    if not token_ids:
        return []
    chunks = [token_ids[start : start + max_seq_length] for start in range(0, len(token_ids), max_seq_length)]
    chunks = [chunk for chunk in chunks if chunk]
    if len(chunks) > 1 and len(chunks[-1]) < min_chunk_tokens:
        last = chunks[-1]
        previous = chunks[-2]
        if len(previous) + len(last) <= max_seq_length:
            chunks[-2] = previous + last
            chunks.pop()
        else:
            needed = min_chunk_tokens - len(last)
            if needed > 0 and len(previous) - needed >= min_chunk_tokens:
                moved = previous[-needed:]
                chunks[-2] = previous[:-needed]
                chunks[-1] = moved + last
    return chunks


def _build_sections(
    source_files: list[Path],
    tokenizer: Any,
    corpus_cfg: dict[str, Any],
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool, bool]:
    sample_unit = _resolve_sample_unit(corpus_cfg)
    split_sections = bool(corpus_cfg.get("split_markdown_sections", True)) and sample_unit != "document"
    append_safety_preamble = bool(corpus_cfg.get("append_safety_preamble", True))
    sections: list[dict[str, Any]] = []
    source_stats: list[dict[str, Any]] = []
    combined_texts: list[str] = []

    for source_file in source_files:
        text = clean_text_noise(read_text(source_file))
        combined_texts.append(text)
        file_sections = _split_markdown_sections(text, source_file.stem, split_sections)
        file_token_total = 0
        for section_index, section in enumerate(file_sections):
            token_ids = list(tokenizer(section["text"], add_special_tokens=False)["input_ids"])
            file_token_total += len(token_ids)
            sections.append(
                {
                    "source_path": str(source_file),
                    "display_path": normalize_path_for_report(source_file),
                    "section_heading": section["heading"],
                    "section_index": section_index,
                    "category": _classify_section(source_file, section["heading"]),
                    "text": section["text"],
                    "char_count": len(section["text"]),
                    "token_ids": token_ids,
                    "token_count": len(token_ids),
                }
            )
        source_stats.append(
            {
                "path": str(source_file),
                "display_path": normalize_path_for_report(source_file),
                "chars": len(text),
                "lines": len(text.splitlines()),
                "section_count": len(file_sections),
                "token_count": file_token_total,
                "chunk_count": 0,
            }
        )

    safety_boundary_present = has_safety_boundary("\n\n".join(combined_texts))
    preamble_added = False
    if append_safety_preamble and not safety_boundary_present:
        preamble_tokens = list(tokenizer(SAFETY_PREAMBLE + "\n", add_special_tokens=False)["input_ids"])
        sections.insert(
            0,
            {
                "source_path": "__safety_preamble__",
                "display_path": "__safety_preamble__",
                "section_heading": "Safety Boundary Preamble",
                "section_index": 0,
                "category": "safety",
                "text": SAFETY_PREAMBLE + "\n",
                "char_count": len(SAFETY_PREAMBLE),
                "token_ids": preamble_tokens,
                "token_count": len(preamble_tokens),
            },
        )
        preamble_added = True
        warnings.append("Safety preamble was prepended because no explicit safety boundary was detected.")
    return sections, source_stats, safety_boundary_present, preamble_added


def _make_example(
    *,
    example_id: str,
    token_ids: list[int],
    original_token_count: int | None,
    was_truncated: bool,
    sample_overflow_strategy: str,
    source_path: str,
    section_heading: str,
    category: str,
    is_mandatory: bool,
    is_replay: bool,
    source_chunk_index: int,
    mandatory_chunk_id: str,
    replay_of_example_id: str,
    validation_copy_of: str,
) -> dict[str, Any]:
    return {
        "example_id": example_id,
        "input_ids": list(token_ids),
        "attention_mask": [1] * len(token_ids),
        "labels": list(token_ids),
        "source_path": source_path,
        "section_heading": section_heading,
        "category": category,
        "is_mandatory": is_mandatory,
        "is_replay": is_replay,
        "token_count": len(token_ids),
        "original_token_count": int(original_token_count if original_token_count is not None else len(token_ids)),
        "was_truncated": was_truncated,
        "truncated_token_count": max(0, int(original_token_count if original_token_count is not None else len(token_ids)) - len(token_ids)),
        "sample_overflow_strategy": sample_overflow_strategy,
        "source_chunk_index": source_chunk_index,
        "mandatory_chunk_id": mandatory_chunk_id,
        "replay_of_example_id": replay_of_example_id,
        "validation_copy_of": validation_copy_of,
    }


def _fit_whole_sample_tokens(
    *,
    token_ids: list[int],
    max_sample_tokens: int | None,
    strategy: str,
    head_ratio: float,
    sample_label: str,
    warnings: list[str],
) -> tuple[list[int], bool, str]:
    if max_sample_tokens is None or max_sample_tokens <= 0 or len(token_ids) <= max_sample_tokens:
        return token_ids, False, "none"

    normalized = strategy.strip().lower().replace("-", "_")
    if normalized in {"", "none", "keep"}:
        warnings.append(
            f"Whole-document sample `{sample_label}` has {len(token_ids)} tokens above max_sample_tokens={max_sample_tokens}; it was kept whole."
        )
        return token_ids, False, "none"
    if normalized in {"error", "fail"}:
        raise ValueError(f"Whole-document sample `{sample_label}` has {len(token_ids)} tokens above max_sample_tokens={max_sample_tokens}.")
    if normalized in {"truncate", "head", "truncate_tail"}:
        fitted = token_ids[:max_sample_tokens]
    elif normalized in {"tail", "truncate_head"}:
        fitted = token_ids[-max_sample_tokens:]
    elif normalized in {"head_tail", "head+tail"}:
        if max_sample_tokens <= 2:
            fitted = token_ids[:max_sample_tokens]
        else:
            ratio = min(0.9, max(0.1, head_ratio))
            head_count = min(max_sample_tokens - 1, max(1, int(round(max_sample_tokens * ratio))))
            tail_count = max_sample_tokens - head_count
            fitted = token_ids[:head_count] + token_ids[-tail_count:]
    else:
        raise ValueError(f"Unsupported corpus.sample_overflow_strategy={strategy!r}.")

    warnings.append(
        f"Whole-document sample `{sample_label}` was reduced from {len(token_ids)} to {len(fitted)} tokens "
        f"with sample_overflow_strategy={normalized}."
    )
    return fitted, True, normalized


def _build_mandatory_chunks(
    sections: list[dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int,
    min_chunk_tokens: int,
    source_stats: list[dict[str, Any]],
    sample_unit: str,
    max_sample_tokens: int | None,
    sample_overflow_strategy: str,
    sample_head_ratio: float,
    warnings: list[str],
) -> list[dict[str, Any]]:
    mandatory_chunks: list[dict[str, Any]] = []
    chunk_counts_by_source: Counter[str] = Counter()
    for section in sections:
        section_token_ids = list(section["token_ids"])
        if tokenizer.eos_token_id is not None:
            section_token_ids.append(int(tokenizer.eos_token_id))
        if sample_unit == "token_chunk":
            section_chunks = [
                {
                    "token_ids": chunk_ids,
                    "original_token_count": len(chunk_ids),
                    "was_truncated": False,
                    "sample_overflow_strategy": "token_chunk",
                }
                for chunk_ids in _chunk_token_ids(section_token_ids, max_seq_length, min_chunk_tokens)
            ]
        else:
            section_chunks = []
            if section_token_ids:
                sample_label = f"{section['display_path']} / {section['section_heading']}"
                fitted_ids, was_truncated, overflow_strategy = _fit_whole_sample_tokens(
                    token_ids=section_token_ids,
                    max_sample_tokens=max_sample_tokens,
                    strategy=sample_overflow_strategy,
                    head_ratio=sample_head_ratio,
                    sample_label=sample_label,
                    warnings=warnings,
                )
                section_chunks.append(
                    {
                        "token_ids": fitted_ids,
                        "original_token_count": len(section_token_ids),
                        "was_truncated": was_truncated,
                        "sample_overflow_strategy": overflow_strategy,
                    }
                )
            if section_chunks and len(section_token_ids) > max_seq_length and max_sample_tokens is None:
                warnings.append(
                    f"{sample_unit} sample `{section['display_path']}` / `{section['section_heading']}` has "
                    f"{len(section_token_ids)} tokens, above max_seq_length={max_seq_length}; it was kept whole."
                )
        for section_chunk_index, chunk in enumerate(section_chunks):
            source_chunk_index = chunk_counts_by_source[section["source_path"]]
            mandatory_id = f"mandatory-{len(mandatory_chunks):06d}"
            mandatory_chunks.append(
                _make_example(
                    example_id=mandatory_id,
                    token_ids=chunk["token_ids"],
                    original_token_count=chunk["original_token_count"],
                    was_truncated=chunk["was_truncated"],
                    sample_overflow_strategy=chunk["sample_overflow_strategy"],
                    source_path=section["source_path"],
                    section_heading=section["section_heading"],
                    category=section["category"],
                    is_mandatory=True,
                    is_replay=False,
                    source_chunk_index=source_chunk_index,
                    mandatory_chunk_id=mandatory_id,
                    replay_of_example_id="",
                    validation_copy_of="",
                )
            )
            chunk_counts_by_source[section["source_path"]] += 1

    for item in source_stats:
        item["chunk_count"] = chunk_counts_by_source.get(item["path"], 0)
    return mandatory_chunks


def _normalize_ratios(raw_ratios: dict[str, Any], warnings: list[str]) -> dict[str, float]:
    parsed = {key: max(0.0, float(value)) for key, value in raw_ratios.items()}
    total = sum(parsed.values())
    if total <= 0:
        warnings.append("Replay ratios sum to zero; weighted replay will not add samples.")
        return {}
    if not math.isclose(total, 1.0):
        warnings.append(f"Replay ratios sum to {total:.6g}; ratios were normalized.")
    return {key: value / total for key, value in parsed.items() if value > 0}


def _weighted_replay(
    mandatory_chunks: list[dict[str, Any]],
    corpus_cfg: dict[str, Any],
    seed: int,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sampling_cfg = _stratified_sampling_cfg(corpus_cfg)
    raw_ratios = dict(sampling_cfg.get("replay_ratios", corpus_cfg.get("replay_ratios", {})))
    max_replay_factor = {
        key: int(value)
        for key, value in dict(sampling_cfg.get("max_replay_factor", corpus_cfg.get("max_replay_factor", {}))).items()
    }
    sample_repeats = {
        key: max(1, int(value))
        for key, value in dict(sampling_cfg.get("sample_repeats", corpus_cfg.get("sample_repeats", {}))).items()
    }
    enabled = _stratified_sampling_enabled(corpus_cfg)
    mode = str(sampling_cfg.get("mode", "sample_repeats" if sample_repeats else "token_budget")).strip().lower()
    normalized_ratios = _normalize_ratios(raw_ratios, warnings) if mode in {"token_budget", "weighted_tokens"} else {}
    mix_report = {
        "enable_weighted_replay": enabled,
        "stratified_sampling_enabled": enabled,
        "stratified_sampling_mode": mode,
        "total_train_tokens": sampling_cfg.get("total_train_tokens", corpus_cfg.get("total_train_tokens")),
        "mandatory_tokens": sum(item["token_count"] for item in mandatory_chunks),
        "replay_ratios_raw": raw_ratios,
        "replay_ratios_normalized": normalized_ratios,
        "max_replay_factor": max_replay_factor,
        "sample_repeats": sample_repeats,
        "category_targets": {},
        "warnings": warnings,
    }

    if not enabled:
        return [], mix_report
    chunks_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in mandatory_chunks:
        chunks_by_category[chunk["category"]].append(chunk)

    if mode in {"sample_repeats", "fixed_repeats", "repeats"}:
        replay_chunks: list[dict[str, Any]] = []
        all_categories = sorted(set(chunks_by_category) | set(sample_repeats))
        for category in all_categories:
            available = chunks_by_category.get(category, [])
            configured_repeats = sample_repeats.get(category, sample_repeats.get("default", 1))
            replay_copies_per_sample = max(0, configured_repeats - 1)
            if category in max_replay_factor:
                capped = min(replay_copies_per_sample, max(0, max_replay_factor[category]))
                if capped < replay_copies_per_sample:
                    warnings.append(
                        f"Category `{category}` sample_repeats={configured_repeats} was capped by max_replay_factor={max_replay_factor[category]}."
                    )
                replay_copies_per_sample = capped
            if not available:
                if configured_repeats > 1:
                    warnings.append(f"Replay category `{category}` has no mandatory samples and was skipped.")
                mix_report["category_targets"][category] = {
                    "available_mandatory_chunks": 0,
                    "configured_sample_repeats": configured_repeats,
                    "replay_copies_per_sample": 0,
                    "target_replay_chunks": 0,
                    "actual_replay_chunks": 0,
                    "cap": 0,
                }
                continue
            for source in available:
                for _copy_index in range(replay_copies_per_sample):
                    replay_id = f"replay-{len(replay_chunks):06d}"
                    replay_chunks.append(
                        _make_example(
                            example_id=replay_id,
                            token_ids=source["input_ids"],
                            original_token_count=source.get("original_token_count", source["token_count"]),
                            was_truncated=bool(source.get("was_truncated", False)),
                            sample_overflow_strategy=source.get("sample_overflow_strategy", "none"),
                            source_path=source["source_path"],
                            section_heading=source["section_heading"],
                            category=source["category"],
                            is_mandatory=False,
                            is_replay=True,
                            source_chunk_index=source["source_chunk_index"],
                            mandatory_chunk_id=source["mandatory_chunk_id"],
                            replay_of_example_id=source["example_id"],
                            validation_copy_of="",
                        )
                    )
            mix_report["category_targets"][category] = {
                "available_mandatory_chunks": len(available),
                "configured_sample_repeats": configured_repeats,
                "replay_copies_per_sample": replay_copies_per_sample,
                "target_replay_chunks": len(available) * replay_copies_per_sample,
                "actual_replay_chunks": len(available) * replay_copies_per_sample,
                "cap": max_replay_factor.get(category),
            }
        mix_report["actual_replay_chunks"] = len(replay_chunks)
        mix_report["actual_replay_tokens"] = sum(item["token_count"] for item in replay_chunks)
        return replay_chunks, mix_report

    if mode not in {"token_budget", "weighted_tokens"}:
        raise ValueError("corpus.stratified_sampling.mode must be `sample_repeats` or `token_budget`.")

    total_train_tokens = mix_report["total_train_tokens"]
    if total_train_tokens is None:
        warnings.append("Weighted replay is enabled but total_train_tokens is null; no replay samples were added.")
        return [], mix_report
    target_total_tokens = int(total_train_tokens)
    mandatory_tokens = int(mix_report["mandatory_tokens"])
    if target_total_tokens <= mandatory_tokens:
        warnings.append(
            "total_train_tokens is smaller than or equal to mandatory token total; no downsampling was performed and replay was skipped."
        )
        mix_report["target_total_tokens_below_mandatory"] = True
        return [], mix_report

    remaining_tokens = target_total_tokens - mandatory_tokens
    rng = random.Random(seed)
    replay_chunks: list[dict[str, Any]] = []

    for category, ratio in normalized_ratios.items():
        available = chunks_by_category.get(category, [])
        if not available:
            warnings.append(f"Replay category `{category}` has no mandatory samples and was skipped.")
            mix_report["category_targets"][category] = {
                "available_mandatory_chunks": 0,
                "target_replay_chunks": 0,
                "actual_replay_chunks": 0,
                "cap": 0,
            }
            continue
        mean_tokens = sum(item["token_count"] for item in available) / len(available)
        target_replay_chunks = max(1, int(round((remaining_tokens * ratio) / max(1.0, mean_tokens))))
        cap = max_replay_factor.get(category, 1) * len(available)
        actual_count = min(target_replay_chunks, cap)
        for _ in range(actual_count):
            source = rng.choice(available)
            replay_id = f"replay-{len(replay_chunks):06d}"
            replay_chunks.append(
                _make_example(
                    example_id=replay_id,
                    token_ids=source["input_ids"],
                    original_token_count=source.get("original_token_count", source["token_count"]),
                    was_truncated=bool(source.get("was_truncated", False)),
                    sample_overflow_strategy=source.get("sample_overflow_strategy", "none"),
                    source_path=source["source_path"],
                    section_heading=source["section_heading"],
                    category=source["category"],
                    is_mandatory=False,
                    is_replay=True,
                    source_chunk_index=source["source_chunk_index"],
                    mandatory_chunk_id=source["mandatory_chunk_id"],
                    replay_of_example_id=source["example_id"],
                    validation_copy_of="",
                )
            )
        mix_report["category_targets"][category] = {
            "available_mandatory_chunks": len(available),
            "target_replay_chunks": target_replay_chunks,
            "actual_replay_chunks": actual_count,
            "cap": cap,
        }

    mix_report["actual_replay_chunks"] = len(replay_chunks)
    mix_report["actual_replay_tokens"] = sum(item["token_count"] for item in replay_chunks)
    return replay_chunks, mix_report


def _build_validation_chunks(
    *,
    validation_mode: str,
    train_chunks: list[dict[str, Any]],
    config: dict[str, Any],
    config_path: Path,
    tokenizer: Any,
    max_seq_length: int,
    min_chunk_tokens: int,
    sample_unit: str,
    max_sample_tokens: int | None,
    sample_overflow_strategy: str,
    sample_head_ratio: float,
    seed: int,
    warnings: list[str],
) -> list[dict[str, Any]]:
    corpus_cfg = config.get("corpus", {})
    training_cfg = config.get("training", {})
    validation_mode = validation_mode.lower()
    if validation_mode == "none":
        if float(training_cfg.get("train_val_split", 0) or 0) > 0:
            warnings.append(
                "training.train_val_split is no longer used to remove mandatory samples when validation_mode=none."
            )
        return []
    if validation_mode == "copy_from_train":
        ratio = float(corpus_cfg.get("validation_copy_ratio", training_cfg.get("train_val_split", 0.03)) or 0)
        if ratio <= 0 or not train_chunks:
            return []
        copy_count = max(1, int(round(len(train_chunks) * ratio)))
        copy_count = min(copy_count, len(train_chunks))
        rng = random.Random(seed)
        selected = rng.sample(train_chunks, copy_count) if copy_count < len(train_chunks) else list(train_chunks)
        validation_chunks = []
        for index, source in enumerate(selected):
            validation_chunks.append(
                _make_example(
                    example_id=f"validation-copy-{index:06d}",
                    token_ids=source["input_ids"],
                    original_token_count=source.get("original_token_count", source["token_count"]),
                    was_truncated=bool(source.get("was_truncated", False)),
                    sample_overflow_strategy=source.get("sample_overflow_strategy", "none"),
                    source_path=source["source_path"],
                    section_heading=source["section_heading"],
                    category=source["category"],
                    is_mandatory=False,
                    is_replay=False,
                    source_chunk_index=source["source_chunk_index"],
                    mandatory_chunk_id=source["mandatory_chunk_id"],
                    replay_of_example_id=source.get("replay_of_example_id", ""),
                    validation_copy_of=source["example_id"],
                )
            )
        warnings.append("Validation is copy_from_train; it is a training-stability reference, not a strict heldout set.")
        return validation_chunks
    if validation_mode == "separate_sources":
        raw_sources = corpus_cfg.get("validation_sources", []) or []
        if not raw_sources:
            warnings.append("validation_mode=separate_sources but validation_sources is empty; validation was disabled.")
            return []
        exclude_fragments = list(dict.fromkeys(DEFAULT_EXCLUDES + list(corpus_cfg.get("exclude_paths", []))))
        validation_paths = [resolve_input_path(path, config_path) for path in raw_sources]
        source_files = _collect_source_files(validation_paths, exclude_fragments)
        if not source_files:
            warnings.append("validation_sources did not resolve to readable .md/.txt files; validation was disabled.")
            return []
        validation_corpus_cfg = dict(corpus_cfg)
        validation_corpus_cfg["append_safety_preamble"] = False
        sections, source_stats, _, _ = _build_sections(source_files, tokenizer, validation_corpus_cfg, warnings)
        validation_chunks = _build_mandatory_chunks(
            sections,
            tokenizer,
            max_seq_length,
            min_chunk_tokens,
            source_stats,
            sample_unit,
            max_sample_tokens,
            sample_overflow_strategy,
            sample_head_ratio,
            warnings,
        )
        for index, chunk in enumerate(validation_chunks):
            chunk["example_id"] = f"validation-source-{index:06d}"
            chunk["is_mandatory"] = False
            chunk["mandatory_chunk_id"] = ""
        return validation_chunks
    raise ValueError(f"Unsupported corpus.validation_mode: {validation_mode}")


def _category_stats(
    mandatory_chunks: list[dict[str, Any]],
    replay_chunks: list[dict[str, Any]],
    validation_chunks: list[dict[str, Any]],
    train_chunks: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    categories = sorted(
        set([item["category"] for item in mandatory_chunks + replay_chunks + validation_chunks + train_chunks])
        | {
            "main_textbook",
            "model_spec",
            "commands",
            "permissions",
            "configuration",
            "runtime_behavior",
            "database_reload",
            "troubleshooting",
            "safety",
            "unknowns",
            "public_api",
        }
    )
    stats: dict[str, dict[str, int]] = {}
    for category in categories:
        mandatory = [item for item in mandatory_chunks if item["category"] == category]
        replay = [item for item in replay_chunks if item["category"] == category]
        train = [item for item in train_chunks if item["category"] == category]
        validation = [item for item in validation_chunks if item["category"] == category]
        stats[category] = {
            "mandatory_chunks": len(mandatory),
            "mandatory_tokens": sum(item["token_count"] for item in mandatory),
            "replay_chunks": len(replay),
            "train_chunks": len(train),
            "validation_chunks": len(validation),
        }
    return stats


def _write_coverage_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# CPT Dataset Coverage Report",
        "",
        f"Status: **{report['status']}**",
        "",
        COVERAGE_SENTENCE if report["mandatory_coverage_rate"] == 1.0 and report["dropped_mandatory_chunks"] == 0 else "Mandatory coverage check failed.",
        "",
        "## Summary",
        "",
        f"- Strategy: `{report['strategy']}`",
        f"- Sample unit: `{report.get('sample_unit', 'token_chunk')}`",
        f"- Max sample tokens: `{report.get('max_sample_tokens', 'N/A')}`",
        f"- Sample overflow strategy: `{report.get('sample_overflow_strategy', 'N/A')}`",
        f"- Total mandatory samples: `{report['total_mandatory_chunks']}`",
        f"- Total mandatory tokens: `{report['total_mandatory_tokens']}`",
        f"- Train mandatory samples: `{report['train_mandatory_chunks']}`",
        f"- Train replay samples: `{report['train_replay_chunks']}`",
        f"- Validation samples: `{report['validation_chunks']}`",
        f"- Max train sample tokens: `{report.get('max_train_sample_tokens', 'N/A')}`",
        f"- Max original train sample tokens: `{report.get('max_original_train_sample_tokens', 'N/A')}`",
        f"- Truncated mandatory samples: `{report.get('truncated_mandatory_samples', 'N/A')}`",
        f"- Truncated train samples: `{report.get('truncated_train_samples', 'N/A')}`",
        f"- Truncated token count: `{report.get('truncated_token_count', 'N/A')}`",
        f"- Mandatory coverage rate: `{report['mandatory_coverage_rate']:.2%}`",
        f"- Dropped mandatory samples: `{report['dropped_mandatory_chunks']}`",
        f"- Dropped token count: `{report['dropped_token_count']}`",
        f"- Replay enabled: `{report['replay_enabled']}`",
        f"- Validation enabled: `{report['validation_enabled']}`",
        f"- Validation mode: `{report['validation_mode']}`",
        "",
        "## Sources",
        "",
    ]
    for source in report["sources"]:
        lines.append(
            f"- `{source['display_path']}`: {source['chars']} chars, {source['token_count']} tokens, "
            f"{source['section_count']} sections, {source['chunk_count']} samples"
        )
    lines.extend(["", "## Categories", ""])
    for category, stats in report["categories"].items():
        lines.append(
            f"- `{category}`: mandatory={stats['mandatory_chunks']} samples/{stats['mandatory_tokens']} tokens, "
            f"replay={stats['replay_chunks']}, train={stats['train_chunks']}, validation={stats['validation_chunks']}"
        )
    longest_samples = report.get("longest_train_samples", [])
    if longest_samples:
        lines.extend(["", "## Longest Train Samples", ""])
        for item in longest_samples:
            lines.append(
                f"- `{item['example_id']}`: {item['token_count']}/{item.get('original_token_count', item['token_count'])} tokens, "
                f"truncated=`{item.get('was_truncated', False)}`, category=`{item['category']}`, "
                f"source=`{item['source_path']}`, replay_of=`{item.get('replay_of_example_id') or 'N/A'}`"
            )
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
    lines.append("")
    write_text(path, "\n".join(lines))


def _write_mix_markdown(path: Path, mix_report: dict[str, Any]) -> None:
    lines = [
        "# CPT Dataset Mix Report",
        "",
        f"Weighted replay enabled: `{mix_report.get('enable_weighted_replay')}`",
        f"Stratified sampling mode: `{mix_report.get('stratified_sampling_mode')}`",
        f"Target total train tokens: `{mix_report.get('total_train_tokens')}`",
        f"Mandatory tokens: `{mix_report.get('mandatory_tokens')}`",
        "",
        "## Replay Ratios",
        "",
        f"- Raw: `{mix_report.get('replay_ratios_raw', {})}`",
        f"- Normalized: `{mix_report.get('replay_ratios_normalized', {})}`",
        f"- Sample repeats: `{mix_report.get('sample_repeats', {})}`",
        f"- Max replay factor: `{mix_report.get('max_replay_factor', {})}`",
        "",
        "## Category Replay Targets",
        "",
    ]
    targets = mix_report.get("category_targets", {})
    if not targets:
        lines.append("No replay targets were used.")
    else:
        for category, item in targets.items():
            lines.append(
                f"- `{category}`: target={item.get('target_replay_chunks')}, actual={item.get('actual_replay_chunks')}, "
                f"available={item.get('available_mandatory_chunks')}, repeats={item.get('configured_sample_repeats')}, "
                f"copies={item.get('replay_copies_per_sample')}, cap={item.get('cap')}"
            )
    if mix_report.get("target_total_tokens_below_mandatory"):
        lines.extend(["", "No downsampling was performed because mandatory tokens already exceed the requested budget."])
    if mix_report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in mix_report["warnings"]:
            lines.append(f"- {warning}")
    lines.append("")
    write_text(path, "\n".join(lines))


def _verify_full_coverage(mandatory_chunks: list[dict[str, Any]], train_chunks: list[dict[str, Any]]) -> tuple[float, list[str], int]:
    mandatory_ids = {item["mandatory_chunk_id"] for item in mandatory_chunks}
    train_mandatory_ids = {item["mandatory_chunk_id"] for item in train_chunks if item.get("is_mandatory")}
    dropped_ids = sorted(mandatory_ids - train_mandatory_ids)
    dropped_tokens = sum(item["token_count"] for item in mandatory_chunks if item["mandatory_chunk_id"] in dropped_ids)
    coverage_rate = len(train_mandatory_ids & mandatory_ids) / max(1, len(mandatory_ids))
    return coverage_rate, dropped_ids, dropped_tokens


def _longest_sample_summary(samples: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    longest = sorted(samples, key=lambda item: int(item.get("token_count", 0)), reverse=True)[:limit]
    return [
        {
            "example_id": item["example_id"],
            "token_count": int(item["token_count"]),
            "original_token_count": int(item.get("original_token_count", item["token_count"])),
            "was_truncated": bool(item.get("was_truncated", False)),
            "category": item["category"],
            "source_path": item["source_path"],
            "section_heading": item["section_heading"],
            "is_mandatory": bool(item.get("is_mandatory")),
            "is_replay": bool(item.get("is_replay")),
            "replay_of_example_id": item.get("replay_of_example_id", ""),
        }
        for item in longest
    ]


def prepare_dataset(
    config: dict[str, Any],
    config_path: Path,
    input_paths: list[Path] | None = None,
    output_dir: Path | None = None,
    max_seq_length: int | None = None,
) -> dict[str, Any]:
    logger = setup_logging("prepare_cpt_dataset")
    corpus_cfg = config.get("corpus", {})
    training_cfg = config.get("training", {})
    selected_paths = _selected_input_paths(config, config_path, input_paths)
    if not selected_paths:
        raise RuntimeError("No CPT corpus files found. Check corpus.input_paths in the config.")
    exclude_fragments = list(dict.fromkeys(DEFAULT_EXCLUDES + list(corpus_cfg.get("exclude_paths", []))))
    source_files = _collect_source_files(selected_paths, exclude_fragments)
    if not source_files:
        raise RuntimeError("No readable .md/.txt CPT source files remained after exclusions.")

    output_dir = output_dir or resolve_training_path(corpus_cfg.get("prepared_dataset_dir"), "outputs/cpt_dataset")
    max_seq_length = int(max_seq_length or training_cfg.get("max_seq_length", 2048))
    min_chunk_tokens = max(1, int(corpus_cfg.get("min_chunk_tokens", 128)))
    sample_unit = _resolve_sample_unit(corpus_cfg)
    max_sample_tokens = parse_nullable_int(corpus_cfg.get("max_sample_tokens"))
    if sample_unit == "token_chunk":
        max_sample_tokens = None
    sample_overflow_strategy = str(corpus_cfg.get("sample_overflow_strategy", "head_tail"))
    sample_head_ratio = float(corpus_cfg.get("sample_head_ratio", 0.70))
    if max_seq_length <= 0:
        raise ValueError(f"Invalid max_seq_length={max_seq_length}")

    trust_remote_code = bool(config.get("trust_remote_code", True))
    base_model = config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError("base_model_name_or_path is required.")
    tokenizer = _load_tokenizer(base_model, trust_remote_code)

    warnings: list[str] = []
    sections, source_stats, safety_boundary_present, preamble_added = _build_sections(source_files, tokenizer, corpus_cfg, warnings)
    mandatory_chunks = _build_mandatory_chunks(
        sections,
        tokenizer,
        max_seq_length,
        min_chunk_tokens,
        source_stats,
        sample_unit,
        max_sample_tokens,
        sample_overflow_strategy,
        sample_head_ratio,
        warnings,
    )
    if not mandatory_chunks:
        raise RuntimeError("No mandatory training samples were created from the CPT corpus.")

    replay_chunks, mix_report = _weighted_replay(mandatory_chunks, corpus_cfg, int(training_cfg.get("seed", 42)), warnings)
    train_chunks = mandatory_chunks + replay_chunks
    validation_mode = str(corpus_cfg.get("validation_mode", "none")).lower()
    validation_chunks = _build_validation_chunks(
        validation_mode=validation_mode,
        train_chunks=train_chunks,
        config=config,
        config_path=config_path,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        min_chunk_tokens=min_chunk_tokens,
        sample_unit=sample_unit,
        max_sample_tokens=max_sample_tokens,
        sample_overflow_strategy=sample_overflow_strategy,
        sample_head_ratio=sample_head_ratio,
        seed=int(training_cfg.get("seed", 42)),
        warnings=warnings,
    )

    coverage_rate, dropped_ids, dropped_tokens = _verify_full_coverage(mandatory_chunks, train_chunks)
    if coverage_rate != 1.0 or dropped_ids:
        raise RuntimeError("Mandatory CPT coverage failed before saving dataset; no training output was written.")

    dataset_dict = DatasetDict({"train": Dataset.from_list(train_chunks)})
    if validation_chunks:
        dataset_dict["validation"] = Dataset.from_list(validation_chunks)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))

    category_stats = _category_stats(mandatory_chunks, replay_chunks, validation_chunks, train_chunks)
    total_chars = sum(item["chars"] for item in source_stats) + (len(SAFETY_PREAMBLE) if preamble_added else 0)
    total_mandatory_tokens = sum(item["token_count"] for item in mandatory_chunks)
    coverage_report = {
        "status": "ok",
        "strategy": corpus_cfg.get("strategy", "guaranteed_coverage_then_optional_replay"),
        "sample_unit": sample_unit,
        "max_sample_tokens": max_sample_tokens,
        "sample_overflow_strategy": sample_overflow_strategy if max_sample_tokens is not None else "none",
        "sample_head_ratio": sample_head_ratio,
        "coverage_sentence": COVERAGE_SENTENCE,
        "sources": source_stats,
        "source_paths": [str(path) for path in source_files],
        "section_count": len(sections),
        "total_chars": total_chars,
        "total_mandatory_chunks": len(mandatory_chunks),
        "total_mandatory_tokens": total_mandatory_tokens,
        "train_mandatory_chunks": sum(1 for item in train_chunks if item["is_mandatory"]),
        "train_replay_chunks": sum(1 for item in train_chunks if item["is_replay"]),
        "train_chunks": len(train_chunks),
        "validation_chunks": len(validation_chunks),
        "max_train_sample_tokens": max(item["token_count"] for item in train_chunks),
        "max_original_train_sample_tokens": max(item["original_token_count"] for item in train_chunks),
        "longest_train_samples": _longest_sample_summary(train_chunks),
        "truncated_train_samples": sum(1 for item in train_chunks if item.get("was_truncated")),
        "truncated_mandatory_samples": sum(1 for item in mandatory_chunks if item.get("was_truncated")),
        "truncated_token_count": sum(item.get("truncated_token_count", 0) for item in train_chunks),
        "mandatory_coverage_rate": coverage_rate,
        "dropped_mandatory_chunks": len(dropped_ids),
        "dropped_mandatory_chunk_ids": dropped_ids,
        "dropped_token_count": dropped_tokens,
        "categories": category_stats,
        "replay_enabled": _stratified_sampling_enabled(corpus_cfg),
        "validation_enabled": bool(validation_chunks),
        "validation_mode": validation_mode if validation_chunks or validation_mode != "separate_sources" else "none",
        "validation_is_copy_from_train": validation_mode == "copy_from_train" and bool(validation_chunks),
        "warnings": list(dict.fromkeys(warnings)),
        "created_at_utc": utc_now(),
    }
    mix_report["warnings"] = list(dict.fromkeys(warnings))
    mix_report["actual_train_chunks"] = len(train_chunks)
    mix_report["actual_train_tokens"] = sum(item["token_count"] for item in train_chunks)

    write_json(output_dir / "coverage_report.json", coverage_report)
    _write_coverage_markdown(output_dir / "coverage_report.md", coverage_report)
    write_json(output_dir / "mix_report.json", mix_report)
    _write_mix_markdown(output_dir / "mix_report.md", mix_report)

    info = {
        "status": "ok",
        "base_model_name_or_path": base_model,
        "strategy": coverage_report["strategy"],
        "source_paths": [str(path) for path in source_files],
        "source_stats": source_stats,
        "total_chars": total_chars,
        "total_tokens": total_mandatory_tokens,
        "max_seq_length": max_seq_length,
        "min_chunk_tokens": min_chunk_tokens,
        "sample_unit": sample_unit,
        "max_sample_tokens": max_sample_tokens,
        "sample_overflow_strategy": coverage_report["sample_overflow_strategy"],
        "sample_head_ratio": sample_head_ratio,
        "num_chunks": len(mandatory_chunks),
        "mandatory_chunks": len(mandatory_chunks),
        "mandatory_tokens": total_mandatory_tokens,
        "train_samples": len(train_chunks),
        "train_mandatory_chunks": coverage_report["train_mandatory_chunks"],
        "train_replay_chunks": coverage_report["train_replay_chunks"],
        "validation_samples": len(validation_chunks),
        "max_train_sample_tokens": coverage_report["max_train_sample_tokens"],
        "max_original_train_sample_tokens": coverage_report["max_original_train_sample_tokens"],
        "longest_train_samples": coverage_report["longest_train_samples"],
        "truncated_train_samples": coverage_report["truncated_train_samples"],
        "truncated_mandatory_samples": coverage_report["truncated_mandatory_samples"],
        "truncated_token_count": coverage_report["truncated_token_count"],
        "mandatory_coverage_rate": coverage_rate,
        "dropped_mandatory_chunks": len(dropped_ids),
        "dropped_token_count": dropped_tokens,
        "validation_mode": coverage_report["validation_mode"],
        "replay_enabled": coverage_report["replay_enabled"],
        "preamble_added": preamble_added,
        "safety_boundary_present": safety_boundary_present,
        "full_token_loss": True,
        "chat_template_used": False,
        "assistant_only_loss": False,
        "coverage_report": str(output_dir / "coverage_report.md"),
        "mix_report": str(output_dir / "mix_report.md"),
        "warnings": coverage_report["warnings"],
        "created_at_utc": utc_now(),
    }
    write_json(output_dir / "dataset_info.json", info)
    save_yaml(output_dir / "config_snapshot.yaml", {k: v for k, v in config.items() if k != "_config_path"})
    logger.info(
        "Prepared CPT dataset with full coverage at %s: %d train, %d validation, %.2f%% mandatory coverage",
        output_dir,
        info["train_samples"],
        info["validation_samples"],
        info["mandatory_coverage_rate"] * 100,
    )
    return info


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a full-token causal LM CPT dataset with guaranteed train coverage.")
    parser.add_argument("--config", default="configs/domain_post_training.yaml", help="Path to domain_post_training.yaml.")
    parser.add_argument("--input_paths", nargs="*", default=None, help="Explicit corpus files or directories.")
    parser.add_argument("--output_dir", default=None, help="Prepared Hugging Face dataset directory.")
    parser.add_argument("--max_seq_length", type=int, default=None, help="Override max sequence length.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    logger = setup_logging("prepare_cpt_dataset", args.verbose)
    config, config_path = load_config(args.config)
    input_paths = [resolve_input_path(p, config_path) for p in args.input_paths] if args.input_paths else None
    output_dir = resolve_training_path(args.output_dir, config.get("corpus", {}).get("prepared_dataset_dir", "outputs/cpt_dataset")) if args.output_dir else None
    info = prepare_dataset(config, config_path, input_paths, output_dir, args.max_seq_length)
    if info["total_chars"] < int(config.get("corpus", {}).get("min_text_chars", 1000)):
        logger.warning("Corpus is short; overfitting risk is high.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
