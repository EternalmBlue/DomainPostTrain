from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pipeline.utils import (
    PROJECT_ROOT,
    ensure_dir,
    load_config,
    normalize_path_for_report,
    read_text,
    resolve_input_path,
    resolve_training_path,
    setup_logging,
    utc_now,
    write_json,
)


VALID_SUFFIXES = {".md", ".markdown", ".txt"}
DEFAULT_DISCOVERY_PATHS = [
    "data/cpt/source_documents",
    "data/cpt",
]


def _is_excluded(path: Path, exclude_fragments: list[str]) -> bool:
    normalized = str(path.resolve()).replace("\\", "/").lower()
    return any(fragment.replace("\\", "/").lower() in normalized for fragment in exclude_fragments)


def _collect_from_path(path: Path, exclude_fragments: list[str]) -> list[Path]:
    if not path.exists() or _is_excluded(path, exclude_fragments):
        return []
    if path.is_file():
        return [path.resolve()] if path.suffix.lower() in VALID_SUFFIXES else []
    files = []
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in VALID_SUFFIXES and not _is_excluded(candidate, exclude_fragments):
            files.append(candidate.resolve())
    return files


def _score_candidate(path: Path) -> tuple[int, str]:
    name = path.name.lower()
    parent = path.parent.name.lower()
    normalized = str(path).replace("\\", "/").lower()
    if name == "cpt_corpus.md":
        return (0, normalized)
    if name == "domain_corpus.md":
        return (1, normalized)
    if "source_documents/" in normalized and path.suffix.lower() in VALID_SUFFIXES:
        return (2, normalized)
    if parent == "cpt" and path.suffix.lower() in VALID_SUFFIXES:
        return (3, normalized)
    return (4, normalized)


def discover_corpus(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    corpus_cfg = config.get("corpus", {})
    exclude_fragments = list(corpus_cfg.get("exclude_paths", []))
    exclude_fragments.append("training_audit_not_for_training")

    configured_paths = [resolve_input_path(p, config_path) for p in corpus_cfg.get("input_paths", [])]
    fallback_paths = [PROJECT_ROOT / p for p in DEFAULT_DISCOVERY_PATHS]
    configured_candidates: list[Path] = []
    for path in configured_paths:
        for candidate in _collect_from_path(path, exclude_fragments):
            if candidate not in configured_candidates:
                configured_candidates.append(candidate)

    fallback_candidates: list[Path] = []
    if not configured_candidates:
        for path in fallback_paths:
            for candidate in _collect_from_path(path, exclude_fragments):
                if candidate not in fallback_candidates:
                    fallback_candidates.append(candidate)

    all_candidates = configured_candidates or fallback_candidates

    if not all_candidates:
        return {
            "status": "not_found",
            "selected_paths": [],
            "all_candidates": [],
            "strategy": "none",
            "timestamp_utc": utc_now(),
        }

    if configured_candidates:
        selected = sorted(configured_candidates, key=_score_candidate)
        strategy = "configured_input_paths"
    else:
        by_name = {candidate.name.lower(): candidate for candidate in all_candidates}
        if "cpt_corpus.md" in by_name:
            selected = [by_name["cpt_corpus.md"]]
            strategy = "single_priority_cpt_corpus"
        elif "domain_corpus.md" in by_name:
            selected = [by_name["domain_corpus.md"]]
            strategy = "single_priority_domain_corpus"
        else:
            source_documents = sorted(
                [p for p in all_candidates if p.parent.name == "source_documents"],
                key=lambda p: p.name,
            )
            selected = source_documents or [sorted(all_candidates, key=_score_candidate)[0]]
            strategy = "merge_source_documents" if source_documents else "single_best_available"

    selected_stats = []
    for path in selected:
        text = read_text(path)
        selected_stats.append(
            {
                "path": str(path),
                "display_path": normalize_path_for_report(path),
                "chars": len(text),
                "lines": len(text.splitlines()),
            }
        )

    return {
        "status": "ok",
        "selected_paths": [str(path) for path in selected],
        "selected_stats": selected_stats,
        "all_candidates": [str(path) for path in sorted(all_candidates, key=_score_candidate)],
        "strategy": strategy,
        "timestamp_utc": utc_now(),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover CPT markdown/text corpus files for PEFT CPT training.")
    parser.add_argument("--config", default="configs/domain_post_training.yaml", help="Path to domain_post_training.yaml.")
    parser.add_argument("--output_json", default=None, help="Optional output JSON path.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    logger = setup_logging("discover_corpus", args.verbose)
    config, config_path = load_config(args.config)
    result = discover_corpus(config, config_path)
    output_path = resolve_training_path(args.output_json, "outputs/logs/discovered_corpus.json")
    ensure_dir(output_path.parent)
    write_json(output_path, result)
    if result["status"] != "ok":
        logger.error("No CPT corpus was found. Wrote discovery report to %s", output_path)
        return 2
    logger.info("Selected %d corpus file(s) with strategy=%s", len(result["selected_paths"]), result["strategy"])
    logger.info("Discovery report: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
