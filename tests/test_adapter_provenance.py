from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pipeline.adapter_provenance import (
    PROVENANCE_FILENAME,
    PROVENANCE_SCHEMA_VERSION,
    inspect_peft_adapter,
    resolve_adapter_provenance,
    validate_adapter_provenance_manifest,
    write_adapter_provenance,
)


class AdapterProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _write_legacy_stage(self, directory: Path, stage: str, base: Path | None = None) -> None:
        payload: dict[str, object] = {
            "status": "completed",
            "adapter_output_dir": str(directory),
            "created_at_utc": "2026-07-16T12:00:00+00:00",
        }
        filenames = {
            "cpt": "training_metadata.json",
            "fact_sft": "fact_sft_training_metadata.json",
            "dpo": "dpo_training_metadata.json",
            "grpo": "grpo_training_metadata.json",
        }
        if stage == "cpt":
            payload.update({"full_token_loss": True, "assistant_only_loss": False})
        elif stage == "fact_sft":
            payload.update({"full_token_loss": False, "assistant_only_loss": True})
        elif stage == "dpo":
            payload["dpo"] = {"beta": 0.1}
        elif stage == "grpo":
            payload["grpo"] = {"num_generations": 4}
        else:
            raise AssertionError(f"Unsupported test stage: {stage}")
        if base is not None:
            payload["base_adapter_dir"] = str(base)
        self._write_json(directory / filenames[stage], payload)

    def test_manifest_is_self_contained_across_full_chain(self) -> None:
        cpt = self.root / "cpt"
        fact_sft = self.root / "fact_sft"
        dpo = self.root / "dpo"
        grpo = self.root / "grpo"
        timestamp = "2026-07-16T12:00:00+00:00"

        write_adapter_provenance(cpt, stage="cpt", created_at_utc=timestamp)
        write_adapter_provenance(fact_sft, stage="fact_sft", created_at_utc=timestamp, base_adapter_dir=cpt)
        write_adapter_provenance(dpo, stage="dpo", created_at_utc=timestamp, base_adapter_dir=fact_sft)
        manifest = write_adapter_provenance(grpo, stage="grpo", created_at_utc=timestamp, base_adapter_dir=dpo)

        self.assertEqual(manifest["schema_version"], PROVENANCE_SCHEMA_VERSION)
        self.assertTrue(manifest["provenance_complete"])
        self.assertEqual([item["stage"] for item in manifest["stages"]], ["cpt", "fact_sft", "dpo", "grpo"])
        self.assertNotIn("base_adapter_dir", json.dumps(manifest))

        resolution = resolve_adapter_provenance(grpo)
        self.assertEqual(resolution.source, "manifest")
        self.assertTrue(resolution.complete)
        self.assertEqual(resolution.stages, ("cpt", "fact_sft", "dpo", "grpo"))

    def test_manifest_preserves_incomplete_base_provenance(self) -> None:
        unknown_base = self.root / "external_adapter"
        unknown_base.mkdir()
        output = self.root / "fact_sft"

        manifest = write_adapter_provenance(
            output,
            stage="fact_sft",
            created_at_utc="2026-07-16T12:00:00+00:00",
            base_adapter_dir=unknown_base,
        )

        self.assertFalse(manifest["provenance_complete"])
        self.assertEqual([item["stage"] for item in manifest["stages"]], ["fact_sft"])
        resolution = resolve_adapter_provenance(output)
        self.assertFalse(resolution.complete)
        self.assertIn("incomplete ancestry", " ".join(resolution.warnings).lower())

    def test_strict_manifest_validation_rejects_unknown_or_unordered_fields(self) -> None:
        valid = {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "provenance_complete": True,
            "stages": [
                {"stage": "cpt", "status": "completed"},
                {"stage": "dpo", "status": "completed"},
            ],
        }
        self.assertEqual(validate_adapter_provenance_manifest(valid), valid)

        with self.assertRaisesRegex(ValueError, "fields"):
            validate_adapter_provenance_manifest({**valid, "api_key": "must-not-leak"})
        with self.assertRaisesRegex(ValueError, "unique and ordered"):
            validate_adapter_provenance_manifest(
                {
                    **valid,
                    "stages": [
                        {"stage": "dpo", "status": "completed"},
                        {"stage": "fact_sft", "status": "completed"},
                    ],
                }
            )
        with self.assertRaisesRegex(ValueError, "completed"):
            validate_adapter_provenance_manifest(
                {**valid, "stages": [{"stage": "cpt", "status": "running"}]}
            )

    def test_invalid_manifest_falls_back_without_leaking_payload_values(self) -> None:
        adapter = self.root / "adapter"
        secret = "credential-sentinel-value"
        self._write_json(
            adapter / PROVENANCE_FILENAME,
            {
                "schema_version": PROVENANCE_SCHEMA_VERSION,
                "provenance_complete": True,
                "stages": [{"stage": "cpt", "status": "completed"}],
                "api_key": secret,
            },
        )
        self._write_legacy_stage(adapter, "cpt")

        resolution = resolve_adapter_provenance(adapter)

        self.assertEqual(resolution.source, "legacy_recursive")
        self.assertTrue(resolution.complete)
        self.assertEqual(resolution.stages, ("cpt",))
        self.assertNotIn(secret, json.dumps(resolution.report_data()))
        self.assertIn("fallback", " ".join(resolution.warnings).lower())

    def test_legacy_metadata_recovers_full_recursive_chain(self) -> None:
        cpt = self.root / "cpt"
        fact_sft = self.root / "fact_sft"
        dpo = self.root / "dpo"
        grpo = self.root / "grpo"
        self._write_legacy_stage(cpt, "cpt")
        self._write_legacy_stage(fact_sft, "fact_sft", cpt)
        self._write_legacy_stage(dpo, "dpo", fact_sft)
        self._write_legacy_stage(grpo, "grpo", dpo)

        resolution = resolve_adapter_provenance(grpo)

        self.assertEqual(resolution.source, "legacy_recursive")
        self.assertTrue(resolution.complete)
        self.assertEqual(resolution.stages, ("cpt", "fact_sft", "dpo", "grpo"))
        self.assertEqual(resolution.warnings, ())

    def test_legacy_copied_metadata_retains_partial_stages_and_sanitized_warning(self) -> None:
        adapter = self.root / "grpo"
        missing_dpo = self.root / "missing_dpo"
        missing_fact_sft = self.root / "missing_fact_sft"
        secret = "private-judge-key-sentinel"
        self._write_legacy_stage(adapter, "grpo", missing_dpo)
        self._write_json(
            adapter / "base_dpo_training_metadata.json",
            {
                "status": "completed",
                "dpo": {"beta": 0.1, "api_key": secret},
                "base_adapter_dir": str(missing_fact_sft),
                "created_at_utc": "2026-07-16T12:00:00+00:00",
            },
        )

        resolution = resolve_adapter_provenance(adapter)

        self.assertEqual(resolution.stages, ("dpo", "grpo"))
        self.assertFalse(resolution.complete)
        self.assertIn("missing base adapter", " ".join(resolution.warnings).lower())
        self.assertNotIn(secret, json.dumps(resolution.report_data()))

    def test_legacy_cycle_and_depth_limit_are_reported(self) -> None:
        grpo = self.root / "cycle_grpo"
        dpo = self.root / "cycle_dpo"
        self._write_legacy_stage(grpo, "grpo", dpo)
        self._write_legacy_stage(dpo, "dpo", grpo)

        cycle = resolve_adapter_provenance(grpo)
        self.assertFalse(cycle.complete)
        self.assertIn("cycle", " ".join(cycle.warnings).lower())

        deep_grpo = self.root / "deep_grpo"
        deep_dpo = self.root / "deep_dpo"
        deep_cpt = self.root / "deep_cpt"
        self._write_legacy_stage(deep_grpo, "grpo", deep_dpo)
        self._write_legacy_stage(deep_dpo, "dpo", deep_cpt)
        self._write_legacy_stage(deep_cpt, "cpt")

        depth_limited = resolve_adapter_provenance(deep_grpo, max_depth=2)
        self.assertFalse(depth_limited.complete)
        self.assertEqual(depth_limited.stages, ("dpo", "grpo"))
        self.assertIn("maximum", " ".join(depth_limited.warnings).lower())

    def test_stage_append_rejects_regression(self) -> None:
        cpt = self.root / "cpt"
        dpo = self.root / "dpo"
        timestamp = "2026-07-16T12:00:00+00:00"
        write_adapter_provenance(cpt, stage="cpt", created_at_utc=timestamp)
        write_adapter_provenance(dpo, stage="dpo", created_at_utc=timestamp, base_adapter_dir=cpt)

        with self.assertRaisesRegex(ValueError, "cannot precede"):
            write_adapter_provenance(
                self.root / "invalid",
                stage="fact_sft",
                created_at_utc=timestamp,
                base_adapter_dir=dpo,
            )

    def test_same_stage_continuation_replaces_the_previous_stage_record(self) -> None:
        first = self.root / "first_grpo"
        second = self.root / "second_grpo"
        write_adapter_provenance(
            first,
            stage="grpo",
            created_at_utc="2026-07-16T12:00:00+00:00",
        )

        manifest = write_adapter_provenance(
            second,
            stage="grpo",
            created_at_utc="2026-07-16T13:00:00+00:00",
            base_adapter_dir=first,
        )

        self.assertEqual(
            manifest["stages"],
            [
                {
                    "stage": "grpo",
                    "status": "completed",
                    "created_at_utc": "2026-07-16T13:00:00+00:00",
                }
            ],
        )

    def test_peft_adapter_structure_validation(self) -> None:
        missing = self.root / "missing"
        self.assertEqual(inspect_peft_adapter(missing), "directory does not exist")

        not_directory = self.root / "file"
        not_directory.write_text("x", encoding="utf-8")
        self.assertEqual(inspect_peft_adapter(not_directory), "path is not a directory")

        empty = self.root / "empty"
        empty.mkdir()
        self.assertIn("adapter_config.json", inspect_peft_adapter(empty) or "")
        self.assertIn("adapter_model.safetensors or adapter_model.bin", inspect_peft_adapter(empty) or "")

        config_only = self.root / "config_only"
        config_only.mkdir()
        (config_only / "adapter_config.json").write_text("{}", encoding="utf-8")
        self.assertIn("adapter_model.safetensors or adapter_model.bin", inspect_peft_adapter(config_only) or "")

        weights_only = self.root / "weights_only"
        weights_only.mkdir()
        (weights_only / "adapter_model.safetensors").write_bytes(b"weights")
        self.assertIn("adapter_config.json", inspect_peft_adapter(weights_only) or "")

        for weight_name in ("adapter_model.safetensors", "adapter_model.bin"):
            with self.subTest(weight_name=weight_name):
                valid = self.root / weight_name.replace(".", "_")
                valid.mkdir()
                (valid / "adapter_config.json").write_text("{}", encoding="utf-8")
                (valid / weight_name).write_bytes(b"weights")
                self.assertIsNone(inspect_peft_adapter(valid))


if __name__ == "__main__":
    unittest.main()
