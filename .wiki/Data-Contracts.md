# 数据契约 / Data Contracts

## 中文

当你要用自己的领域数据替换仓库自带 mock 数据时，使用本页。

中文提示：发布衍生仓库前，不要把私有文档、客户数据、凭据、私有 system prompt、源代码或许可证受限语料放进 `data/`。

## 打包数据结构

```text
data/cpt/source_documents/*.md      # CPT source documents
data/sft/*.jsonl                    # Fact-SFT examples
data/dpo/preference_examples.jsonl  # DPO preference pairs
data/grpo/reward_examples.jsonl     # GRPO reward examples
data/eval/quality_questions.jsonl   # post-training quality evaluation questions
```

样例领域是虚构的 `AsterHelp` 内部支持知识库助手，数据刻意保持很小且静态。

## CPT 文档

输入：

```text
data/cpt/source_documents/*.md
```

用途：

- 教给模型领域概念、政策、流程、安全边界和排障知识。
- 用于 corpus discovery、安全预检查和覆盖度感知的数据集构建。

关键配置：

```yaml
corpus:
  input_paths:
    - "../data/cpt/source_documents"
```

## Fact-SFT 行

输入：

```text
data/sft/*.jsonl
```

常见字段：

| 字段 | 含义 |
|---|---|
| `instruction` | 用户任务或问题。 |
| `output` | 期望助手回答。 |

Fact-SFT 使用 assistant-only loss，因此 prompt token 会被 mask，只训练回答 token。

## DPO 行

输入：

```text
data/dpo/preference_examples.jsonl
```

必填字段：

| 字段 | 含义 |
|---|---|
| `prompt` | 用户提示或任务。 |
| `chosen` | 偏好的回答。 |
| `rejected` | 不偏好的回答。 |

规则：

- `prompt`、`chosen`、`rejected` 都必须非空。
- `chosen` 必须不同于 `rejected`。

## GRPO 行

输入：

```text
data/grpo/reward_examples.jsonl
```

基础要求：

- 必须存在 `prompt`。
- 至少存在一个奖励信号字段。

奖励信号字段：

| 字段 | 含义 |
|---|---|
| `reference_answer` | 参考答案，用于 overlap 类内置奖励，也会作为 judge 上下文。 |
| `required_terms` | 生成结果应包含的术语。 |
| `forbidden_terms` | 生成结果应避免的术语。 |
| `must_refuse` | 正确行为是否应该拒答。 |
| `min_completion_chars` | 可选的最短回答字符数。 |
| `max_completion_chars` | 可选的最长回答字符数。 |
| `category` | 可选分组标签，用于报告和 judge 上下文。 |

这些字段既可以驱动内置规则奖励，也可以作为外部 `reward_judge` 的评判上下文。项目不会再把 `reward_models` 的本地路径或 Hub ID 直接交给 TRL 加载。

详见 [GRPO 与 Reward Judge](GRPO-and-Reward-Judge)。

## 质量评估行

输入：

```text
data/eval/quality_questions.jsonl
```

常见字段：

| 字段 | 含义 |
|---|---|
| `category` | 评估类别，例如 `domain_knowledge`、`safety_boundary` 或 `base_regression`。 |
| `question` | 训练后质量评估使用的提示。 |

质量评估不是训练验证集。它在训练或合并后运行，用来检查输出行为。

## 发布前检查

发布衍生仓库前：

1. 确认所有数据都可公开、授权明确且适合分发。
2. 移除私有路径、客户名称、内部 URL、密钥和专有 prompt。
3. 检查 `configs/domain_post_training.yaml` 中是否残留私有默认值。
4. 不要提交 `outputs/`、`models/`、`.venv/`、`__pycache__/` 和训练产物。

---

## English

Use this page when replacing the packaged mock data with your own domain data.

Note: before publishing a derivative repository, do not put private documents, customer data, credentials, private system prompts, source code, or license-restricted corpora in `data/`.

## Packaged Data Layout

```text
data/cpt/source_documents/*.md      # CPT source documents
data/sft/*.jsonl                    # Fact-SFT examples
data/dpo/preference_examples.jsonl  # DPO preference pairs
data/grpo/reward_examples.jsonl     # GRPO reward examples
data/eval/quality_questions.jsonl   # post-training quality evaluation questions
```

The sample domain is `AsterHelp`, a fictional internal support knowledge-base assistant. It is deliberately small and static.

## CPT Documents

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

## Fact-SFT Rows

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

## DPO Rows

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

## GRPO Rows

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
| `reference_answer` | Reference text for overlap-style scoring and judge context. |
| `required_terms` | Terms the completion should include. |
| `forbidden_terms` | Terms the completion should avoid. |
| `must_refuse` | Whether the correct behavior is refusal. |
| `min_completion_chars` | Optional lower length bound. |
| `max_completion_chars` | Optional upper length bound. |
| `category` | Optional grouping label for reports and judge context. |

These fields can drive built-in rule rewards and provide context for the external `reward_judge`. The project no longer passes local `reward_models` paths or Hub IDs directly to TRL.

See [GRPO And Reward Judge](GRPO-and-Reward-Judge).

## Quality Evaluation Rows

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

## Publication Checklist

Before publishing a derivative repository:

1. Confirm all data is public, licensed, and safe to distribute.
2. Remove private paths, customer names, internal URLs, secrets, and proprietary prompts.
3. Check `configs/domain_post_training.yaml` for private defaults.
4. Keep `outputs/`, `models/`, `.venv/`, `__pycache__/`, and training artifacts out of commits.
