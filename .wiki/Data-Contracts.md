# Data Contracts

Use this page when replacing the packaged mock data with your own domain data.

中文提示: 发布派生仓库前, 不要把私有文档, 客户数据, 凭据, 私有 system prompt, 源代码或许可证受限语料放进 `data/`.

## Packaged data layout

```text
data/cpt/source_documents/*.md      # CPT source documents
data/sft/*.jsonl                    # Fact-SFT examples
data/dpo/preference_examples.jsonl  # DPO preference pairs
data/grpo/reward_examples.jsonl     # GRPO reward examples
data/eval/quality_questions.jsonl   # post-training quality evaluation questions
```

The sample domain is `AsterHelp`, a fictional internal support knowledge-base assistant. It is deliberately small and static.

## CPT documents

Input:

```text
data/cpt/source_documents/*.md
```

Purpose:

- Teach domain concepts, policies, procedures, safety boundaries, and troubleshooting knowledge.
- Feed corpus discovery, safety preflight, and coverage-aware dataset construction.

Important config:

```yaml
corpus:
  input_paths:
    - "../data/cpt/source_documents"
```

## Fact-SFT rows

Input:

```text
data/sft/*.jsonl
```

Common fields:

| Field | Meaning |
|---|---|
| `instruction` | User task or question. |
| `output` | Expected assistant answer. |

Fact-SFT uses assistant-only loss, so prompt tokens are masked and only answer tokens train the model.

## DPO rows

Input:

```text
data/dpo/preference_examples.jsonl
```

Required fields:

| Field | Meaning |
|---|---|
| `prompt` | User prompt or task. |
| `chosen` | Preferred answer. |
| `rejected` | Less preferred answer. |

Rules:

- `prompt`, `chosen`, and `rejected` must be non-empty.
- `chosen` must differ from `rejected`.

## GRPO rows

Input:

```text
data/grpo/reward_examples.jsonl
```

Required baseline:

- `prompt` must be present.
- At least one reward signal must be present.

Reward signal fields:

| Field | Meaning |
|---|---|
| `reference_answer` | Reference text for overlap-style scoring or judge context. |
| `required_terms` | Terms the completion should include. |
| `forbidden_terms` | Terms the completion should avoid. |
| `must_refuse` | Whether the correct behavior is refusal. |
| `min_completion_chars` | Optional lower length bound. |
| `max_completion_chars` | Optional upper length bound. |
| `category` | Optional grouping label for reports and judge context. |

See [GRPO And Reward Judge](GRPO-and-Reward-Judge) for built-in rewards and external judge scoring.

## Quality evaluation rows

Input:

```text
data/eval/quality_questions.jsonl
```

Common fields:

| Field | Meaning |
|---|---|
| `category` | Evaluation category such as `domain_knowledge`, `safety_boundary`, or `base_regression`. |
| `question` | Prompt used for post-training quality evaluation. |

Quality evaluation is not a training validation set. It runs after training or merge to inspect output behavior.

## Publication checklist

Before publishing a derivative repository:

1. Confirm all data is public, licensed, and safe to distribute.
2. Remove private paths, customer names, internal URLs, secrets, and proprietary prompts.
3. Check `configs/domain_post_training.yaml` for private defaults.
4. Keep `outputs/`, `models/`, `.venv/`, `__pycache__/`, and training artifacts out of commits.

