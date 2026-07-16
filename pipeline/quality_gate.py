from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any


QUALITY_GATE_CATEGORIES = ("domain_knowledge", "safety_boundary", "base_regression")
DEFAULT_MINIMUM_PASS_RATE = {
    "domain_knowledge": 1.0,
    "safety_boundary": 1.0,
    "base_regression": 1.0,
}

_QUALITY_GATE_KEYS = {"enabled", "fail_pipeline", "minimum_pass_rate"}


def _require_boolean(value: Any, field_path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_path} must be a boolean.")
    return value


def _require_rate(value: Any, field_path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_path} must be a finite number between 0 and 1; booleans are not allowed.")
    rate = float(value)
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError(f"{field_path} must be a finite number between 0 and 1; booleans are not allowed.")
    return rate


def validate_quality_gate_configuration(eval_config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and normalize the public ``eval.quality_gate`` configuration."""

    if eval_config is None:
        eval_config = {}
    if not isinstance(eval_config, Mapping):
        raise ValueError("eval must be a mapping.")

    raw_gate = eval_config.get("quality_gate", {})
    if raw_gate is None:
        raw_gate = {}
    if not isinstance(raw_gate, Mapping):
        raise ValueError("eval.quality_gate must be a mapping.")

    unknown_keys = sorted(str(key) for key in raw_gate if key not in _QUALITY_GATE_KEYS)
    if unknown_keys:
        raise ValueError(f"eval.quality_gate contains unsupported keys: {', '.join(unknown_keys)}.")

    enabled = _require_boolean(raw_gate.get("enabled", True), "eval.quality_gate.enabled")
    fail_pipeline = _require_boolean(raw_gate.get("fail_pipeline", True), "eval.quality_gate.fail_pipeline")

    raw_rates = raw_gate.get("minimum_pass_rate", DEFAULT_MINIMUM_PASS_RATE)
    if not isinstance(raw_rates, Mapping):
        raise ValueError("eval.quality_gate.minimum_pass_rate must be a mapping.")

    unknown_categories = sorted(str(category) for category in raw_rates if category not in QUALITY_GATE_CATEGORIES)
    if unknown_categories:
        allowed = ", ".join(QUALITY_GATE_CATEGORIES)
        raise ValueError(
            "eval.quality_gate.minimum_pass_rate contains unsupported categories: "
            f"{', '.join(unknown_categories)}. Allowed: {allowed}."
        )
    if enabled and not raw_rates:
        raise ValueError("eval.quality_gate.minimum_pass_rate must contain at least one category when the gate is enabled.")

    minimum_pass_rate = {
        category: _require_rate(value, f"eval.quality_gate.minimum_pass_rate.{category}")
        for category, value in raw_rates.items()
    }
    return {
        "enabled": enabled,
        "fail_pipeline": fail_pipeline,
        "minimum_pass_rate": minimum_pass_rate,
    }


def _section_pass_rate(section: Mapping[str, Any]) -> tuple[float | None, int | None, int | None]:
    passed = section.get("passed")
    total = section.get("total")
    if (
        isinstance(passed, int)
        and not isinstance(passed, bool)
        and isinstance(total, int)
        and not isinstance(total, bool)
        and total > 0
        and 0 <= passed <= total
    ):
        return passed / total, passed, total

    raw_rate = section.get("pass_rate")
    if isinstance(raw_rate, bool) or not isinstance(raw_rate, Real):
        return None, None, None
    rate = float(raw_rate)
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        return None, None, None
    return rate, None, None


def _target_name(target_report: Mapping[str, Any]) -> str | None:
    raw_name = target_report.get("target")
    if isinstance(raw_name, str) and raw_name.strip():
        return raw_name.strip()
    return None


def evaluate_quality_gate(
    eval_config: Mapping[str, Any] | None,
    target_reports: Sequence[Mapping[str, Any]],
    *,
    evaluation_performed: bool = True,
    expected_targets: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Assess evaluation summaries without copying prompts, completions, paths, or credentials."""

    gate = validate_quality_gate_configuration(eval_config)
    result: dict[str, Any] = {
        **gate,
        "status": "not_evaluated",
        "should_fail_pipeline": False,
        "evaluated_target_count": 0,
        "targets": [],
        "failed_checks": [],
    }
    if not evaluation_performed:
        return result
    if not gate["enabled"]:
        result["status"] = "not_enforced"
        return result

    if isinstance(target_reports, (str, bytes)) or not isinstance(target_reports, Sequence):
        raise ValueError("target_reports must be a sequence of evaluation target mappings.")
    if expected_targets is not None and (
        isinstance(expected_targets, (str, bytes)) or not isinstance(expected_targets, Sequence)
    ):
        raise ValueError("expected_targets must be a sequence of target names.")

    reports_by_name: dict[str, Mapping[str, Any]] = {}
    duplicate_targets: set[str] = set()
    unnamed_targets: set[str] = set()
    invalid_target_entries = 0
    for index, report in enumerate(target_reports):
        if not isinstance(report, Mapping):
            invalid_target_entries += 1
            continue
        name = _target_name(report)
        if name is None:
            name = f"invalid_target_{index + 1}"
            unnamed_targets.add(name)
        if name in reports_by_name:
            duplicate_targets.add(name)
        else:
            reports_by_name[name] = report

    names = list(expected_targets) if expected_targets is not None else list(reports_by_name)
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("expected_targets must contain only non-empty target names.")
    names = list(dict.fromkeys(name.strip() for name in names))
    if not names:
        result["failed_checks"].append({"target": None, "category": None, "reason": "no_targets_evaluated"})

    for name in names:
        report = reports_by_name.get(name)
        target_result = {"target": name, "status": "failed", "categories": []}
        if report is None:
            target_result["categories"] = [
                {
                    "category": category,
                    "minimum_pass_rate": threshold,
                    "pass_rate": None,
                    "passed": None,
                    "total": None,
                    "status": "failed",
                    "reason": "target_not_evaluated",
                }
                for category, threshold in gate["minimum_pass_rate"].items()
            ]
        else:
            result["evaluated_target_count"] += 1
            sections_by_category: dict[str, Mapping[str, Any]] = {}
            duplicate_categories: set[str] = set()
            raw_sections = report.get("sections", [])
            if isinstance(raw_sections, Sequence) and not isinstance(raw_sections, (str, bytes)):
                for section in raw_sections:
                    if not isinstance(section, Mapping):
                        continue
                    category = section.get("category")
                    if category in sections_by_category:
                        duplicate_categories.add(str(category))
                    elif isinstance(category, str):
                        sections_by_category[category] = section

            category_results = []
            for category, threshold in gate["minimum_pass_rate"].items():
                section = sections_by_category.get(category)
                if section is None:
                    rate, passed, total = None, None, None
                    reason = "missing_category"
                elif category in duplicate_categories:
                    rate, passed, total = None, None, None
                    reason = "duplicate_category"
                else:
                    rate, passed, total = _section_pass_rate(section)
                    if rate is None:
                        reason = "invalid_section"
                    elif rate < threshold:
                        reason = "below_minimum_pass_rate"
                    else:
                        reason = "threshold_met"
                category_results.append(
                    {
                        "category": category,
                        "minimum_pass_rate": threshold,
                        "pass_rate": rate,
                        "passed": passed,
                        "total": total,
                        "status": "passed" if reason == "threshold_met" else "failed",
                        "reason": reason,
                    }
                )
            target_result["categories"] = category_results

        target_result["status"] = (
            "passed" if target_result["categories"] and all(item["status"] == "passed" for item in target_result["categories"]) else "failed"
        )
        result["targets"].append(target_result)
        for category_result in target_result["categories"]:
            if category_result["status"] == "failed":
                result["failed_checks"].append(
                    {
                        "target": name,
                        "category": category_result["category"],
                        "reason": category_result["reason"],
                    }
                )

    if duplicate_targets:
        for name in sorted(duplicate_targets):
            result["failed_checks"].append({"target": name, "category": None, "reason": "duplicate_target"})
    for name in sorted(unnamed_targets):
        result["failed_checks"].append({"target": name, "category": None, "reason": "missing_target_name"})
    for _ in range(invalid_target_entries):
        result["failed_checks"].append({"target": None, "category": None, "reason": "invalid_target_report"})

    result["status"] = "failed" if result["failed_checks"] else "passed"
    result["should_fail_pipeline"] = bool(result["status"] == "failed" and gate["fail_pipeline"])
    return result
