# Packaged Mock Training Data

This directory contains static mock data for the DomainPostTrain example project. The fictional domain is `AsterHelp`, an internal support knowledge-base assistant.

The data is intentionally small and readable:

- `data/cpt/source_documents/*.md` contains CPT-style source documents.
- `data/sft/*.jsonl` contains `instruction`/`output` examples.
- `data/dpo/preference_examples.jsonl` contains `prompt`/`chosen`/`rejected` preference pairs.
- `data/grpo/reward_examples.jsonl` contains GRPO prompts plus optional trusted context, reference answers, and built-in reward signals. Prompt-only rows are valid when the external judge is enabled, but private-domain factual grading requires `trusted_context` or `reference_answer`.
- `data/eval/quality_questions.jsonl` contains post-training quality evaluation questions.

GRPO built-in rewards are disabled by default. When the external judge is disabled, every row must contain at least one signal matching an enabled reward:

| Built-in reward | Canonical fields | Accepted aliases / notes |
| --- | --- | --- |
| `reference_overlap` | `reference_answer` | `answer`, `solution`, `ground_truth`, or `expected` |
| `term_constraints` | `required_terms` and/or `forbidden_terms` | Required: `must_include` / `keywords`; forbidden: `must_not_include` / `banned_terms`; the built-in formula uses case-insensitive literal substring checks |
| `refusal` | `must_refuse: true` | `requires_refusal: true`; only rows that explicitly require refusal provide this signal |
| `length_bounds` | `min_completion_chars` and/or `max_completion_chars` | Inclusive character bounds computed locally after trimming the completion |

The four built-in scores keep their existing deterministic formulas. They are separate from the default external `reward_judge` score and may be combined only when explicitly listed in `grpo.builtin_rewards`. `forbidden_terms_mode: semantic|literal` controls external Judge interpretation; it does not change the built-in `term_constraints` formula.

Do not put private documents, credentials, customer tickets, source code, internal prompts, or license-restricted data into this directory before publishing a derivative repository.
