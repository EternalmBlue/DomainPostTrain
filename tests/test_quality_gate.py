from __future__ import annotations

import json
import math
import unittest

from pipeline.quality_gate import (
    DEFAULT_MINIMUM_PASS_RATE,
    evaluate_quality_gate,
    validate_quality_gate_configuration,
)


def _target(name: str = "merged", **rates: float) -> dict:
    return {
        "target": name,
        "model_path": "never-copy-this-path",
        "samples": [{"question": "private prompt", "response": "private completion"}],
        "sections": [
            {"category": category, "passed": round(rate * 10), "total": 10, "pass_rate": rate}
            for category, rate in rates.items()
        ],
    }


class QualityGateConfigurationTests(unittest.TestCase):
    def test_defaults_are_enforced_and_normalized(self) -> None:
        self.assertEqual(
            validate_quality_gate_configuration({}),
            {"enabled": True, "fail_pipeline": True, "minimum_pass_rate": DEFAULT_MINIMUM_PASS_RATE},
        )

    def test_disabled_gate_allows_an_empty_threshold_map(self) -> None:
        actual = validate_quality_gate_configuration(
            {"quality_gate": {"enabled": False, "fail_pipeline": False, "minimum_pass_rate": {}}}
        )
        self.assertEqual(actual["minimum_pass_rate"], {})

    def test_rejects_non_mapping_and_unknown_settings(self) -> None:
        with self.assertRaisesRegex(ValueError, "eval must be a mapping"):
            validate_quality_gate_configuration([])  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "eval.quality_gate must be a mapping"):
            validate_quality_gate_configuration({"quality_gate": []})
        with self.assertRaisesRegex(ValueError, "unsupported keys: typo"):
            validate_quality_gate_configuration({"quality_gate": {"typo": True}})

    def test_rejects_non_boolean_switches(self) -> None:
        for field in ("enabled", "fail_pipeline"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, f"eval.quality_gate.{field}"):
                validate_quality_gate_configuration({"quality_gate": {field: 1}})

    def test_rejects_unknown_empty_and_non_mapping_thresholds(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported categories: invented"):
            validate_quality_gate_configuration(
                {"quality_gate": {"minimum_pass_rate": {"invented": 0.5}}}
            )
        with self.assertRaisesRegex(ValueError, "at least one category"):
            validate_quality_gate_configuration({"quality_gate": {"minimum_pass_rate": {}}})
        with self.assertRaisesRegex(ValueError, "minimum_pass_rate must be a mapping"):
            validate_quality_gate_configuration({"quality_gate": {"minimum_pass_rate": 0.8}})

    def test_rejects_boolean_non_numeric_nonfinite_and_out_of_range_rates(self) -> None:
        invalid = (True, "0.8", math.nan, math.inf, -0.01, 1.01)
        for value in invalid:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite number between 0 and 1"):
                validate_quality_gate_configuration(
                    {"quality_gate": {"minimum_pass_rate": {"safety_boundary": value}}}
                )


class QualityGateEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.eval_config = {
            "quality_gate": {
                "enabled": True,
                "fail_pipeline": True,
                "minimum_pass_rate": {
                    "domain_knowledge": 0.8,
                    "safety_boundary": 1.0,
                    "base_regression": 0.8,
                },
            }
        }

    def test_passes_when_every_configured_category_meets_its_threshold(self) -> None:
        report = evaluate_quality_gate(
            self.eval_config,
            [_target(domain_knowledge=0.8, safety_boundary=1.0, base_regression=0.9)],
            expected_targets=["merged"],
        )
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["should_fail_pipeline"])
        self.assertEqual(report["evaluated_target_count"], 1)

    def test_below_threshold_fails_and_can_be_non_blocking(self) -> None:
        target = _target(domain_knowledge=0.7, safety_boundary=1.0, base_regression=0.8)
        blocking = evaluate_quality_gate(self.eval_config, [target])
        self.assertEqual(blocking["status"], "failed")
        self.assertTrue(blocking["should_fail_pipeline"])
        self.assertEqual(
            blocking["failed_checks"],
            [{"target": "merged", "category": "domain_knowledge", "reason": "below_minimum_pass_rate"}],
        )

        non_blocking_config = {
            "quality_gate": {**self.eval_config["quality_gate"], "fail_pipeline": False}
        }
        non_blocking = evaluate_quality_gate(non_blocking_config, [target])
        self.assertEqual(non_blocking["status"], "failed")
        self.assertFalse(non_blocking["should_fail_pipeline"])

    def test_missing_category_fails_closed(self) -> None:
        report = evaluate_quality_gate(
            self.eval_config,
            [_target(domain_knowledge=1.0, safety_boundary=1.0)],
        )
        self.assertIn(
            {"target": "merged", "category": "base_regression", "reason": "missing_category"},
            report["failed_checks"],
        )

    def test_expected_target_that_did_not_run_fails_closed(self) -> None:
        report = evaluate_quality_gate(self.eval_config, [], expected_targets=["merged"])
        self.assertEqual(report["status"], "failed")
        self.assertTrue(report["should_fail_pipeline"])
        self.assertEqual(report["evaluated_target_count"], 0)
        self.assertTrue(all(item["reason"] == "target_not_evaluated" for item in report["failed_checks"]))

    def test_no_expected_or_evaluated_targets_fails_closed(self) -> None:
        report = evaluate_quality_gate(self.eval_config, [])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failed_checks"][0]["reason"], "no_targets_evaluated")

    def test_rejects_malformed_target_collections(self) -> None:
        with self.assertRaisesRegex(ValueError, "target_reports must be a sequence"):
            evaluate_quality_gate(self.eval_config, {})  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "expected_targets must be a sequence"):
            evaluate_quality_gate(self.eval_config, [], expected_targets="merged")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "non-empty target names"):
            evaluate_quality_gate(self.eval_config, [], expected_targets=[""])

    def test_disabled_and_skipped_states_are_distinct(self) -> None:
        disabled = evaluate_quality_gate(
            {"quality_gate": {"enabled": False, "minimum_pass_rate": {}}},
            [],
        )
        self.assertEqual(disabled["status"], "not_enforced")
        self.assertFalse(disabled["should_fail_pipeline"])

        skipped = evaluate_quality_gate(self.eval_config, [], evaluation_performed=False)
        self.assertEqual(skipped["status"], "not_evaluated")
        self.assertFalse(skipped["should_fail_pipeline"])

    def test_counts_are_used_instead_of_a_stale_pass_rate(self) -> None:
        target = _target(domain_knowledge=1.0, safety_boundary=1.0, base_regression=1.0)
        target["sections"][0].update({"passed": 7, "total": 10, "pass_rate": 1.0})
        report = evaluate_quality_gate(self.eval_config, [target])
        category = report["targets"][0]["categories"][0]
        self.assertEqual(category["pass_rate"], 0.7)
        self.assertEqual(category["status"], "failed")

    def test_malformed_and_duplicate_sections_fail_closed(self) -> None:
        target = _target(domain_knowledge=1.0, safety_boundary=1.0, base_regression=1.0)
        target["sections"][0] = {"category": "domain_knowledge", "pass_rate": math.nan}
        target["sections"].append({"category": "safety_boundary", "pass_rate": 1.0})
        report = evaluate_quality_gate(self.eval_config, [target])
        reasons = {item["reason"] for item in report["failed_checks"]}
        self.assertIn("invalid_section", reasons)
        self.assertIn("duplicate_category", reasons)

    def test_unnamed_and_duplicate_targets_fail_closed(self) -> None:
        valid = _target(domain_knowledge=1.0, safety_boundary=1.0, base_regression=1.0)
        unnamed = dict(valid)
        unnamed.pop("target")
        report = evaluate_quality_gate(self.eval_config, [valid, valid, unnamed])
        reasons = {item["reason"] for item in report["failed_checks"]}
        self.assertIn("duplicate_target", reasons)
        self.assertIn("missing_target_name", reasons)

        malformed = evaluate_quality_gate(self.eval_config, [None])  # type: ignore[list-item]
        self.assertEqual(malformed["failed_checks"][0]["reason"], "no_targets_evaluated")
        self.assertEqual(malformed["failed_checks"][1]["reason"], "invalid_target_report")

    def test_result_does_not_copy_prompts_completions_paths_or_credentials(self) -> None:
        target = _target(domain_knowledge=1.0, safety_boundary=1.0, base_regression=1.0)
        target["api_key"] = "secret-key"
        report = evaluate_quality_gate(self.eval_config, [target])
        serialized = json.dumps(report)
        for secret in ("private prompt", "private completion", "never-copy-this-path", "secret-key"):
            self.assertNotIn(secret, serialized)


if __name__ == "__main__":
    unittest.main()
