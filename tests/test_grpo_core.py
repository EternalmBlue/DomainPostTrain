from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.grpo_core import (
    build_grpo_reward_functions,
    build_builtin_reward_functions,
    length_bounds_reward,
    normalise_grpo_record,
    parse_reward_judge_score,
    reference_overlap_reward,
    reward_judge_metadata,
    refusal_reward,
    term_constraint_reward,
)


ROOT = Path(__file__).resolve().parents[1]


class GrpoCoreTests(unittest.TestCase):
    def test_default_config_and_entrypoints_reference_grpo_chain(self) -> None:
        config_text = (ROOT / "configs" / "domain_post_training.yaml").read_text(encoding="utf-8")
        pipeline_text = (ROOT / "scripts" / "training" / "train_pipeline.py").read_text(encoding="utf-8-sig")
        entrypoint_text = (ROOT / "scripts" / "training" / "train_grpo.py").read_text(encoding="utf-8")
        merge_text = (ROOT / "pipeline" / "adapter_merge.py").read_text(encoding="utf-8")

        self.assertIn("grpo:", config_text)
        self.assertIn('input_path: "data/grpo/reward_examples.jsonl"', config_text)
        self.assertIn('prepared_dataset_dir: "outputs/grpo_dataset"', config_text)
        self.assertIn('output_dir: "outputs/grpo_adapter"', config_text)
        self.assertIn("require_base_adapter: true", config_text)
        self.assertNotIn("reward_functions:", config_text)
        self.assertNotIn("reward_models:", config_text)
        self.assertIn("reward_judge:", config_text)
        self.assertIn("enabled: false", config_text)
        self.assertIn('api_key_env: "GRPO_REWARD_JUDGE_API_KEY"', config_text)
        self.assertIn("prepare_grpo_dataset", pipeline_text)
        self.assertIn("train_grpo", pipeline_text)
        self.assertIn("--skip_grpo", pipeline_text)
        self.assertIn("from pipeline.grpo import main", entrypoint_text)
        self.assertLess(merge_text.index("grpo.output_dir"), merge_text.index("dpo.output_dir"))

    def test_packaged_grpo_rows_have_reward_signal(self) -> None:
        path = ROOT / "data" / "grpo" / "reward_examples.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertGreaterEqual(len(rows), 1)
        for index, row in enumerate(rows):
            normalised = normalise_grpo_record(
                row,
                source_path=path,
                source_index=index,
                system_prompt="System boundary.",
            )
            self.assertTrue(normalised["prompt"].endswith("Assistant:"))
            self.assertTrue(
                normalised["reference_answer"]
                or normalised["required_terms"]
                or normalised["forbidden_terms"]
                or normalised["must_refuse"]
            )

    def test_missing_reward_signal_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "reward signal"):
            normalise_grpo_record(
                {"prompt": "Answer this."},
                source_path=Path("sample.jsonl"),
                source_index=0,
                system_prompt="System boundary.",
            )

    def test_builtin_rewards_score_expected_boundaries(self) -> None:
        completions = ["The documentation does not specify credentials.", "Here is the password token value."]
        self.assertGreater(reference_overlap_reward(completions, ["documentation credentials", "password token"])[0], 0)
        self.assertGreater(term_constraint_reward(completions, ["documentation", "password"], ["token", "token"])[0], 0)
        self.assertLess(term_constraint_reward(completions, ["documentation", "password"], ["token", "token"])[1], 1)
        self.assertEqual(refusal_reward([completions[0]], [True]), [1.0])
        self.assertEqual(refusal_reward([completions[1]], [True]), [-0.5])
        self.assertEqual(length_bounds_reward([""], 1, 10), [-1.0])

    def test_reward_builder_rejects_unknown_name(self) -> None:
        names = [func.__name__ for func in build_builtin_reward_functions({"builtin_rewards": ["reference_overlap", "refusal"]})]
        self.assertIn("reference_overlap_reward", names)
        with self.assertRaisesRegex(ValueError, "Unsupported GRPO"):
            build_builtin_reward_functions({"builtin_rewards": ["missing_reward"]})

    def test_reward_judge_config_validation_fails_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "base_url"):
            build_grpo_reward_functions({"builtin_rewards": [], "reward_judge": {"enabled": True, "model": "judge"}})
        with self.assertRaisesRegex(ValueError, "model"):
            build_grpo_reward_functions({"builtin_rewards": [], "reward_judge": {"enabled": True, "base_url": "http://localhost:8000/v1"}})
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "GRPO_REWARD_JUDGE_API_KEY"):
                build_grpo_reward_functions(
                    {
                        "builtin_rewards": [],
                        "reward_judge": {
                            "enabled": True,
                            "base_url": "http://localhost:8000/v1",
                            "model": "judge",
                        },
                    }
                )

    def test_reward_judge_score_parsing_and_clamping(self) -> None:
        self.assertEqual(parse_reward_judge_score('{"score": 0.75, "reason": "ok"}', [0, 1]), 0.75)
        self.assertEqual(parse_reward_judge_score('```json\n{"score": 1.5, "reason": "ok"}\n```', [0, 1]), 1.0)
        self.assertEqual(parse_reward_judge_score('{"score": -0.5, "reason": "ok"}', [0, 1]), 0.0)
        for payload in ["not json", '{"reason": "missing"}', '{"score": "1"}']:
            with self.assertRaisesRegex(ValueError, "score|JSON"):
                parse_reward_judge_score(payload, [0, 1])

    def test_grpo_reward_builder_appends_enabled_judge(self) -> None:
        functions = build_grpo_reward_functions({"builtin_rewards": ["reference_overlap"], "reward_judge": {"enabled": False}})
        self.assertEqual([func.__name__ for func in functions], ["reference_overlap_reward"])
        with patch.dict(os.environ, {"GRPO_REWARD_JUDGE_API_KEY": "test-key"}, clear=True):
            functions = build_grpo_reward_functions(
                {
                    "builtin_rewards": ["reference_overlap"],
                    "reward_judge": {
                        "enabled": True,
                        "base_url": "http://localhost:8000/v1",
                        "model": "judge",
                    },
                }
            )
        self.assertEqual([func.__name__ for func in functions], ["reference_overlap_reward", "openai_compatible_reward_judge"])

    def test_reward_judge_metadata_is_secret_free(self) -> None:
        with patch.dict(os.environ, {"GRPO_REWARD_JUDGE_API_KEY": "secret-value"}, clear=True):
            metadata = reward_judge_metadata(
                {
                    "reward_judge": {
                        "enabled": True,
                        "base_url": "http://localhost:8000/v1/",
                        "model": "judge",
                    }
                }
            )
        self.assertEqual(metadata["base_url"], "http://localhost:8000/v1")
        self.assertEqual(metadata["api_key_env"], "GRPO_REWARD_JUDGE_API_KEY")
        self.assertNotIn("secret-value", json.dumps(metadata))


if __name__ == "__main__":
    unittest.main()
