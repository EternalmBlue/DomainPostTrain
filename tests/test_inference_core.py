from __future__ import annotations

import unittest

from pipeline.inference_core import (
    NO_REASONING_INSTRUCTION,
    PLAIN_TEXT_INSTRUCTION,
    build_messages_prompt,
    build_prompt,
    clean_answer,
    plain_text_answer,
    resolve_generation_settings,
    split_reasoning,
    system_prompt_with_output_rule,
)


class _ModernTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "modern-template"


class _LegacyTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(self, _messages, **kwargs):
        self.calls.append(kwargs)
        if "enable_thinking" in kwargs:
            raise TypeError("unsupported keyword")
        return "legacy-template"


class _BrokenTokenizer:
    def apply_chat_template(self, _messages, **_kwargs):
        raise RuntimeError("broken template")


class InferenceCoreTests(unittest.TestCase):
    def test_generation_defaults_are_deterministic_and_shared(self) -> None:
        self.assertEqual(
            resolve_generation_settings({}),
            {
                "max_new_tokens": 256,
                "temperature": 0.0,
                "top_p": 0.9,
                "repetition_penalty": 1.05,
                "no_repeat_ngram_size": 6,
                "do_sample": False,
            },
        )

    def test_generation_settings_validate_public_ranges(self) -> None:
        invalid = (
            {"max_new_tokens": 0},
            {"temperature": -0.1},
            {"top_p": 0},
            {"top_p": 1.1},
            {"repetition_penalty": 0},
            {"no_repeat_ngram_size": -1},
        )
        for settings in invalid:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                resolve_generation_settings(settings)

    def test_output_rules_are_added_once(self) -> None:
        prompt = system_prompt_with_output_rule("Domain assistant.")
        self.assertEqual(prompt.count(NO_REASONING_INSTRUCTION), 1)
        self.assertEqual(prompt.count(PLAIN_TEXT_INSTRUCTION), 1)
        self.assertEqual(system_prompt_with_output_rule(prompt), prompt)

    def test_existing_think_rule_is_detected_case_insensitively(self) -> None:
        prompt = system_prompt_with_output_rule("Never emit <THINK> tags. Use plain text.")
        self.assertNotIn(NO_REASONING_INSTRUCTION, prompt)
        self.assertNotIn(PLAIN_TEXT_INSTRUCTION, prompt)

    def test_build_prompt_disables_thinking_for_modern_template(self) -> None:
        tokenizer = _ModernTokenizer()
        rendered = build_prompt(tokenizer, "  user question  ", False, "System boundary.")

        self.assertEqual(rendered, "modern-template")
        messages, kwargs = tokenizer.calls[0]
        self.assertEqual(messages[1], {"role": "user", "content": "user question"})
        self.assertTrue(kwargs["add_generation_prompt"])
        self.assertFalse(kwargs["tokenize"])
        self.assertFalse(kwargs["enable_thinking"])

    def test_build_prompt_retries_legacy_template_without_thinking_argument(self) -> None:
        tokenizer = _LegacyTokenizer()
        self.assertEqual(build_prompt(tokenizer, "question", False, "system"), "legacy-template")
        self.assertEqual(len(tokenizer.calls), 2)
        self.assertIn("enable_thinking", tokenizer.calls[0])
        self.assertNotIn("enable_thinking", tokenizer.calls[1])

    def test_raw_prompt_is_unchanged_and_does_not_call_template(self) -> None:
        tokenizer = _ModernTokenizer()
        self.assertEqual(build_prompt(tokenizer, "  raw prompt  ", True, "system"), "  raw prompt  ")
        self.assertEqual(tokenizer.calls, [])

    def test_template_failure_uses_role_label_fallback(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
        ]
        rendered = build_messages_prompt(_BrokenTokenizer(), messages)
        self.assertEqual(rendered, "System: system\nUser: question\nAssistant:\n")

    def test_split_reasoning_handles_complete_and_unbalanced_tags(self) -> None:
        answer, reasoning = split_reasoning("<think>private steps</think>Visible answer")
        self.assertEqual(answer, "Visible answer")
        self.assertEqual(reasoning, "private steps")

        answer, reasoning = split_reasoning("private steps</think>Visible answer")
        self.assertEqual(answer, "Visible answer")
        self.assertEqual(reasoning, "private steps")

        answer, reasoning = split_reasoning("Visible answer<think>private steps")
        self.assertEqual(answer, "Visible answer")
        self.assertEqual(reasoning, "private steps")

    def test_clean_answer_removes_reasoning_and_repeated_turns(self) -> None:
        raw = "<think>private</think>Final answer\nUser: another question"
        self.assertEqual(clean_answer(raw), "Final answer")

    def test_split_reasoning_treats_text_markers_case_insensitively(self) -> None:
        answer, reasoning = split_reasoning("Final answer\nanalysis: private steps")
        self.assertEqual(answer, "Final answer")
        self.assertEqual(reasoning, "private steps")

    def test_plain_text_answer_normalises_markdown_and_literal_newlines(self) -> None:
        raw = "# Heading\\n- **first**/n2. `second`\n```text\nthird\n```"
        self.assertEqual(plain_text_answer(raw), "Heading first second third")


if __name__ == "__main__":
    unittest.main()
