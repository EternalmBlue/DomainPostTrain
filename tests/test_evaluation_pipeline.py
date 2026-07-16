from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.evaluation import _generation_config, evaluate, evaluate_target


class EvaluationPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _config(self, *, fail_pipeline: bool = True) -> dict:
        questions = self.root / "quality.jsonl"
        questions.write_text(
            "\n".join(
                json.dumps({"category": category, "question": f"{category} question"})
                for category in ("domain_knowledge", "safety_boundary", "base_regression")
            ),
            encoding="utf-8",
        )
        return {
            "base_model_name_or_path": "base",
            "training": {"merged_output_dir": "merged"},
            "fact_sft": {"system_prompt": "Domain system boundary."},
            "eval": {
                "question_file": str(questions),
                "temperature": 0.0,
                "quality_gate": {
                    "enabled": True,
                    "fail_pipeline": fail_pipeline,
                    "minimum_pass_rate": {
                        "domain_knowledge": 1.0,
                        "safety_boundary": 1.0,
                        "base_regression": 1.0,
                    },
                },
            },
        }

    def test_evaluate_target_uses_shared_clean_inference_contract(self) -> None:
        generation = {
            "response": "Clean final answer",
            "reasoning_present": True,
            "prompt_tokens": 12,
            "completion_tokens": 8,
            "max_tokens_reached": False,
        }
        with (
            patch("pipeline.evaluation._load_model_and_tokenizer", return_value=(object(), object())),
            patch("pipeline.evaluation._generate", return_value=generation) as generate,
        ):
            report = evaluate_target(
                "merged",
                "merged-model",
                self._config(),
                self.root,
                {
                    "domain_knowledge": ["Question"],
                    "safety_boundary": [],
                    "base_regression": [],
                },
            )

        generate.assert_called_once()
        self.assertEqual(generate.call_args.kwargs["system_prompt"], "Domain system boundary.")
        self.assertEqual(report["inference"]["prompt_mode"], "chat_template_or_fallback")
        self.assertFalse(report["inference"]["enable_thinking"])
        self.assertTrue(report["inference"]["clean_answer"])
        self.assertEqual(report["samples"][0]["response"], "Clean final answer")
        self.assertTrue(report["samples"][0]["reasoning_present"])

    def test_evaluation_uses_shared_deterministic_generation_defaults(self) -> None:
        self.assertEqual(_generation_config({})["max_new_tokens"], 256)
        self.assertEqual(_generation_config({})["temperature"], 0.0)
        self.assertFalse(_generation_config({})["do_sample"])

    def test_generate_uses_chat_template_and_cleans_reasoning(self) -> None:
        import torch

        from pipeline.evaluation import _generate

        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 2

            def __init__(self) -> None:
                self.template_kwargs = None

            def apply_chat_template(self, _messages, **kwargs):
                self.template_kwargs = kwargs
                return "rendered chat prompt"

            def __call__(self, prompt, return_tensors):
                self.prompt = prompt
                self.return_tensors = return_tensors
                return {"input_ids": torch.tensor([[10, 11, 12]])}

            def decode(self, _tokens, skip_special_tokens):
                self.skip_special_tokens = skip_special_tokens
                return "<think>private analysis</think>## Final answer"

        class Model:
            def parameters(self):
                yield torch.nn.Parameter(torch.zeros(1))

            def generate(self, **kwargs):
                self.kwargs = kwargs
                return torch.tensor([[10, 11, 12, 20, 21]])

        tokenizer = Tokenizer()
        model = Model()
        result = _generate(
            model,
            tokenizer,
            "Question",
            {
                "max_new_tokens": 8,
                "temperature": 0.0,
                "top_p": 0.9,
                "repetition_penalty": 1.05,
                "no_repeat_ngram_size": 6,
                "do_sample": False,
            },
            system_prompt="System boundary.",
        )

        self.assertFalse(tokenizer.template_kwargs["enable_thinking"])
        self.assertEqual(tokenizer.prompt, "rendered chat prompt")
        self.assertNotIn("temperature", model.kwargs)
        self.assertNotIn("top_p", model.kwargs)
        self.assertEqual(model.kwargs["no_repeat_ngram_size"], 6)
        self.assertEqual(result["response"], "Final answer")
        self.assertTrue(result["reasoning_present"])
        self.assertEqual(result["prompt_tokens"], 3)
        self.assertEqual(result["completion_tokens"], 2)

    def test_evaluate_records_failed_quality_gate_without_hiding_outputs(self) -> None:
        target_report = {
            "target": "merged",
            "model_path": "merged",
            "sections": [
                {"category": "domain_knowledge", "passed": 1, "total": 1, "pass_rate": 1.0, "results": []},
                {"category": "safety_boundary", "passed": 0, "total": 1, "pass_rate": 0.0, "results": []},
                {"category": "base_regression", "passed": 1, "total": 1, "pass_rate": 1.0, "results": []},
            ],
            "samples": [],
        }
        output_dir = self.root / "eval"
        with patch("pipeline.evaluation.evaluate_target", return_value=target_report):
            report = evaluate(self._config(), self.root / "config.yaml", ["merged"], output_dir)

        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["quality_gate"]["status"], "failed")
        self.assertTrue(report["quality_gate"]["should_fail_pipeline"])
        self.assertTrue((output_dir / "eval_report.json").is_file())
        saved = json.loads((output_dir / "eval_report.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["quality_gate"]["status"], "failed")

    def test_generation_error_is_a_technical_evaluation_failure_even_when_gate_is_disabled(self) -> None:
        config = self._config()
        config["eval"]["quality_gate"]["enabled"] = False
        output_dir = self.root / "eval-error"
        with (
            patch("pipeline.evaluation._load_model_and_tokenizer", return_value=(object(), object())),
            patch("pipeline.evaluation._generate", side_effect=RuntimeError("generation failed")),
        ):
            report = evaluate(config, self.root / "config.yaml", ["merged"], output_dir)

        self.assertEqual(report["status"], "completed_with_failures")
        self.assertEqual(report["generation_error_count"], 3)
        self.assertEqual(report["quality_gate"]["status"], "not_enforced")

    def test_invalid_gate_configuration_fails_before_loading_a_model(self) -> None:
        config = self._config()
        config["eval"]["quality_gate"]["typo"] = True
        with (
            patch("pipeline.evaluation._load_model_and_tokenizer") as load_model,
            self.assertRaisesRegex(ValueError, "unsupported keys"),
        ):
            evaluate(config, self.root / "config.yaml", ["merged"], self.root / "invalid-gate")
        load_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
