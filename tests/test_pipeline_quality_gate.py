from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.training import train_pipeline


class PipelineQualityGateTests(unittest.TestCase):
    def _run(self, *, gate_status: str, should_fail: bool, skip_eval: bool = False) -> tuple[int, list[dict]]:
        config = {
            "base_model_name_or_path": "base",
            "training": {"output_dir": "cpt", "merged_output_dir": "merged"},
            "fact_sft": {"enabled": False},
            "dpo": {"enabled": False},
            "grpo": {"enabled": False},
            "eval": {"quality_gate": {"enabled": True, "fail_pipeline": True}},
        }
        reports: list[dict] = []
        args = [
            "train_pipeline.py",
            "--config",
            "config.yaml",
            "--skip_cpt",
            "--skip_sft",
            "--skip_dpo",
            "--skip_grpo",
            "--skip_merge",
        ]
        if skip_eval:
            args.append("--skip_eval")
        evaluation = {
            "status": "completed",
            "quality_gate": {"status": gate_status, "should_fail_pipeline": should_fail},
        }

        with (
            patch("sys.argv", args),
            patch("scripts.training.train_pipeline.load_config", return_value=(config, Path("config.yaml"))),
            patch("scripts.training.train_pipeline.evaluate", return_value=evaluation),
            patch(
                "scripts.training.train_pipeline._write_final_report",
                side_effect=lambda *_args, **kwargs: reports.append(kwargs),
            ),
        ):
            return train_pipeline.main(), reports

    def test_gate_failure_keeps_report_and_returns_exit_8(self) -> None:
        exit_code, reports = self._run(gate_status="failed", should_fail=True)
        self.assertEqual(exit_code, 8)
        self.assertEqual(len(reports), 1)
        self.assertTrue(reports[0]["success"])
        self.assertTrue(reports[0]["evaluation_performed"])

    def test_gate_pass_returns_zero(self) -> None:
        exit_code, reports = self._run(gate_status="passed", should_fail=False)
        self.assertEqual(exit_code, 0)
        self.assertTrue(reports[0]["success"])

    def test_skipped_evaluation_is_nonblocking_and_reported(self) -> None:
        exit_code, reports = self._run(gate_status="not_evaluated", should_fail=False, skip_eval=True)
        self.assertEqual(exit_code, 0)
        self.assertFalse(reports[0]["evaluation_performed"])

    def test_skipped_evaluation_does_not_reuse_a_stale_eval_report(self) -> None:
        config = {
            "training": {"output_dir": "cpt", "merged_output_dir": "merged"},
            "fact_sft": {"enabled": False},
            "dpo": {"enabled": False},
            "grpo": {"enabled": False},
        }
        written: list[str] = []

        def read_report(_path: Path, default=None):
            return default

        with (
            patch("scripts.training.train_pipeline.resolve_training_path", side_effect=lambda raw, default: Path(raw or default)),
            patch("scripts.training.train_pipeline.read_json", side_effect=read_report) as read_json,
            patch("scripts.training.train_pipeline.write_text", side_effect=lambda _path, text: written.append(text)),
        ):
            train_pipeline._write_final_report(
                config,
                smoke_test=False,
                success=True,
                evaluation_performed=False,
            )

        self.assertFalse(any(call.args[0].name == "eval_report.json" for call in read_json.call_args_list))
        self.assertIn("Quality gate status: `not_evaluated`", written[0])
        self.assertIn("Eval report: `not_run`", written[0])
        self.assertIn("Safety eval passed: `not_run`", written[0])
        self.assertIn("Release ready: `False`", written[0])

    def test_nonblocking_gate_failure_returns_zero_but_is_not_release_ready(self) -> None:
        config = {
            "training": {"output_dir": "cpt", "merged_output_dir": "merged"},
            "fact_sft": {"enabled": False},
            "dpo": {"enabled": False},
            "grpo": {"enabled": False},
        }
        eval_report = {
            "status": "completed",
            "safety_eval_passed": False,
            "quality_gate": {"status": "failed", "should_fail_pipeline": False},
        }
        written: list[str] = []
        with (
            patch("scripts.training.train_pipeline.resolve_training_path", side_effect=lambda raw, default: Path(raw or default)),
            patch("scripts.training.train_pipeline.read_json", side_effect=lambda _path, default=None: default),
            patch("scripts.training.train_pipeline.write_text", side_effect=lambda _path, text: written.append(text)),
        ):
            train_pipeline._write_final_report(
                config,
                smoke_test=False,
                success=True,
                evaluation_performed=True,
                eval_report=eval_report,
            )

        self.assertIn("Status: **completed_with_quality_warnings**", written[0])
        self.assertIn("Release ready: `False`", written[0])

    def test_invalid_gate_configuration_fails_before_pipeline_stages(self) -> None:
        config = {
            "training": {"output_dir": "cpt", "merged_output_dir": "merged"},
            "fact_sft": {"enabled": False},
            "dpo": {"enabled": False},
            "grpo": {"enabled": False},
            "eval": {"quality_gate": {"typo": True}},
        }
        args = [
            "train_pipeline.py",
            "--config",
            "config.yaml",
            "--skip_cpt",
            "--skip_sft",
            "--skip_dpo",
            "--skip_grpo",
            "--skip_merge",
            "--skip_eval",
        ]
        with (
            patch("sys.argv", args),
            patch("scripts.training.train_pipeline.load_config", return_value=(config, Path("config.yaml"))),
            patch("scripts.training.train_pipeline.merge_adapter") as merge,
            patch("scripts.training.train_pipeline.fail_with_report"),
            patch("scripts.training.train_pipeline._write_final_report"),
        ):
            self.assertEqual(train_pipeline.main(), 7)
        merge.assert_not_called()


if __name__ == "__main__":
    unittest.main()
