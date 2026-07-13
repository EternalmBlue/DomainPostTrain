# 数据契约 / Data Contracts

## 中文

当你要用自己的领域数据替换仓库自带 mock 数据时，使用本页。

发布衍生仓库前，请确认 `data/` 中不包含私有文档、客户数据、凭据、私有 system prompt、源代码或许可证受限语料。

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

推荐字段：

| 字段 | 含义 |
|---|---|
| `instruction` | 用户任务或问题。 |
| `output` | 期望助手回答。 |
| `input` 或 `context` | 可选上下文，会拼到用户问题后。 |
| `category` 或 `type` | 可选分类，用于报告分组。 |

加载器也接受这些等价格式：

- `question` + `answer`
- `prompt` + `response`
- `messages`，其中第一个 assistant 消息作为答案，之前的 system/user 消息作为提示。

Fact-SFT 使用 assistant-only loss，因此 prompt token 会被 mask，只训练回答 token。

## DPO 行

输入：

```text
data/dpo/preference_examples.jsonl
```

推荐字段：

| 字段 | 含义 |
|---|---|
| `prompt` | 用户提示或任务。 |
| `chosen` | 偏好的回答。 |
| `rejected` | 不偏好的回答。 |
| `category` 或 `type` | 可选分类，用于报告分组。 |

加载器也接受这些别名：

| 标准字段 | 可接受别名 |
|---|---|
| `prompt` | `instruction`, `question` |
| `chosen` | `preferred`, `accept` |
| `rejected` | `bad`, `reject` |

规则：

- `prompt`、`chosen`、`rejected` 都必须非空。
- `chosen` 必须不同于 `rejected`。

## GRPO 行

输入：

```text
data/grpo/reward_examples.jsonl
```

基础要求：

- 必须能构造出 prompt。
- Judge 开启时允许只包含 prompt。
- Judge 关闭时，每行必须至少存在一个与已启用内置奖励匹配的信号。

Prompt 可以来自：

- `messages`
- `prompt`
- `instruction`
- `question`

可选 `context` 或 `input` 会被拼入 prompt。`include_system_prompt: false` 可以让样本不附加配置中的 system prompt。

奖励信号字段：

| 字段 | 含义 |
|---|---|
| `reference_answer` | 参考答案，用于 overlap 类内置奖励，也会作为 judge 上下文。 |
| `answer`, `solution`, `ground_truth`, `expected` | `reference_answer` 的可接受别名。 |
| `trusted_context`, `judge_context` | 提供给 Judge 的可信依据；`context` / `input` 也会作为该字段的回退。 |
| `required_terms` | 生成结果应包含的术语。 |
| `must_include`, `keywords` | `required_terms` 的可接受别名。 |
| `forbidden_terms` | 生成结果应避免的术语。 |
| `must_not_include`, `banned_terms` | `forbidden_terms` 的可接受别名。 |
| `forbidden_terms_mode` | `semantic`（默认）按语义判断，`literal` 按不区分大小写的字面命中判断。 |
| `must_refuse` | 正确行为是否应该拒答。 |
| `requires_refusal` | `must_refuse` 的可接受别名。 |
| `min_completion_chars` | 可选的最短回答字符数。 |
| `max_completion_chars` | 可选的最长回答字符数。 |
| `category`, `type`, `task` | 可选分组标签，用于报告和 judge 上下文。 |

内置奖励与字段严格对应：`reference_overlap` 使用 `reference_answer`；`term_constraints` 使用 required/forbidden terms；`refusal` 使用有效拒答要求；`length_bounds` 使用最小或最大字符数。四个内置奖励默认关闭，只在 `builtin_rewards` 中显式加入后生效。

这些字段也会作为外部 `reward_judge` 的评判上下文。Judge 默认开启，使用严格 `grpo_judge_v2` 五维响应；没有参考答案或可信上下文时，`factual_grounding` 必须为 `null`。模型型评分不读取本地模型路径或 Hub ID，需要通过 [GRPO 与 Reward Judge](GRPO-and-Reward-Judge) 配置 HTTP Judge。

## 质量评估行

输入：

```text
data/eval/quality_questions.jsonl
```

字段：

| 字段 | 含义 |
|---|---|
| `category` | 必须是 `domain_knowledge`、`safety_boundary` 或 `base_regression`。 |
| `question` | 训练后质量评估使用的提示，必须非空。 |

质量评估不是训练验证集。它在训练或合并后运行，用来检查输出行为；当前规则是启发式 smoke gate，不是生产安全认证。

## 发布前检查

发布衍生仓库前：

1. 确认所有数据都可公开、授权明确且适合分发。
2. 移除私有路径、客户名称、内部 URL、密钥和专有 prompt。
3. 检查 `configs/domain_post_training.yaml` 中是否残留私有默认值。
4. 不要提交 `outputs/`、`models/`、`.venv/`、`__pycache__/` 和训练产物。

---

## English

Use this page when replacing the packaged mock data with your own domain data.

Before publishing a derivative repository, confirm that `data/` does not contain private documents, customer data, credentials, private system prompts, source code, or license-restricted corpora.

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

Recommended fields:

| Field | Meaning |
|---|---|
| `instruction` | User task or question. |
| `output` | Expected assistant answer. |
| `input` or `context` | Optional context appended to the user prompt. |
| `category` or `type` | Optional category for reports. |

The loader also accepts:

- `question` + `answer`
- `prompt` + `response`
- `messages`, where the first assistant message becomes the answer and earlier system/user messages become the prompt.

Fact-SFT uses assistant-only loss, so prompt tokens are masked and only answer tokens train the model.

## DPO Rows

Input:

```text
data/dpo/preference_examples.jsonl
```

Recommended fields:

| Field | Meaning |
|---|---|
| `prompt` | User prompt or task. |
| `chosen` | Preferred answer. |
| `rejected` | Less preferred answer. |
| `category` or `type` | Optional category for reports. |

Accepted aliases:

| Standard field | Accepted aliases |
|---|---|
| `prompt` | `instruction`, `question` |
| `chosen` | `preferred`, `accept` |
| `rejected` | `bad`, `reject` |

Rules:

- `prompt`, `chosen`, and `rejected` must be non-empty.
- `chosen` must differ from `rejected`.

## GRPO Rows

Input:

```text
data/grpo/reward_examples.jsonl
```

Required baseline:

- A prompt must be constructible.
- Prompt-only rows are valid when the judge is enabled.
- With the judge disabled, every row needs a signal matching an enabled built-in reward.

Prompt can come from:

- `messages`
- `prompt`
- `instruction`
- `question`

Optional `context` or `input` is appended to the prompt. `include_system_prompt: false` prevents the configured system prompt from being added to that example.

Reward signal fields:

| Field | Meaning |
|---|---|
| `reference_answer` | Reference text for overlap-style scoring and judge context. |
| `answer`, `solution`, `ground_truth`, `expected` | Accepted aliases for `reference_answer`. |
| `trusted_context`, `judge_context` | Trusted judge evidence; `context` / `input` also act as fallbacks. |
| `required_terms` | Terms the completion should include. |
| `must_include`, `keywords` | Accepted aliases for `required_terms`. |
| `forbidden_terms` | Terms the completion should avoid. |
| `must_not_include`, `banned_terms` | Accepted aliases for `forbidden_terms`. |
| `forbidden_terms_mode` | `semantic` (default) evaluates meaning; `literal` uses case-insensitive text occurrence. |
| `must_refuse` | Whether the correct behavior is refusal. |
| `requires_refusal` | Accepted alias for `must_refuse`. |
| `min_completion_chars` | Optional lower length bound. |
| `max_completion_chars` | Optional upper length bound. |
| `category`, `type`, `task` | Optional grouping label for reports and judge context. |

Built-in rewards map directly to fields: `reference_overlap` uses `reference_answer`; `term_constraints` uses required/forbidden terms; `refusal` uses a valid refusal requirement; and `length_bounds` uses a minimum or maximum character count. All four built-in rewards are disabled by default and activate only when explicitly listed in `builtin_rewards`.

These fields also provide context for the external `reward_judge`. The judge is enabled by default and uses the strict five-dimension `grpo_judge_v2` response. Without a reference answer or trusted context, `factual_grounding` must be `null`. Model-based scoring does not load local model paths or Hub IDs; configure an HTTP judge through [GRPO And Reward Judge](GRPO-and-Reward-Judge).

## Quality Evaluation Rows

Input:

```text
data/eval/quality_questions.jsonl
```

Fields:

| Field | Meaning |
|---|---|
| `category` | Must be `domain_knowledge`, `safety_boundary`, or `base_regression`. |
| `question` | Non-empty prompt used for post-training quality evaluation. |

Quality evaluation is not a training validation set. It runs after training or merge to inspect output behavior; the current rules are a heuristic smoke gate, not production safety certification.

## Publication Checklist

Before publishing a derivative repository:

1. Confirm all data is public, licensed, and safe to distribute.
2. Remove private paths, customer names, internal URLs, secrets, and proprietary prompts.
3. Check `configs/domain_post_training.yaml` for private defaults.
4. Keep `outputs/`, `models/`, `.venv/`, `__pycache__/`, and training artifacts out of commits.
