# GRPO 与 Reward Judge / GRPO And Reward Judge

## 中文

GRPO 默认使用外部大模型 Judge 提供奖励。四个内置奖励默认关闭，只在 `builtin_rewards` 中显式加入后生效。外部 Judge 通过 OpenAI-compatible `/v1/chat/completions` 接口调用。

## 推荐配置

把真实配置保存在 Git 忽略的 `configs/domain_post_training.local.yaml`：

```yaml
grpo:
  enabled: true
  input_path: "data/grpo/reward_examples.jsonl"
  num_generations: 4
  max_completion_length: 256
  builtin_rewards: []
  reward_judge:
    enabled: true
    base_url: "https://api.deepseek.com/v1"
    model: "your-judge-model"
    api_key: "replace-with-your-key"
    score_range: [0.0, 1.0]
    timeout_seconds: 120
    max_tokens: 4096
    max_retries: 2
    max_concurrency: "auto"
    retry_backoff_seconds: 1.0
```

`api_key` 是明文敏感信息。`configs/*.local.yaml` 必须保持 Git 忽略，不要把真实 Key 放入受跟踪模板、训练报告或分享的配置中。若不希望在 YAML 中保存 Key，可将 `api_key` 留空，并使用环境变量作为可选回退：

```yaml
grpo:
  reward_judge:
    api_key: null
    api_key_env: "GRPO_REWARD_JUDGE_API_KEY"
```

直接配置的 `api_key` 优先于 `api_key_env`。

## 数据与奖励模式

Judge 开启时，每行只需要能构造出 prompt；只有 `prompt` 的数据也有效。参考答案、可信上下文、术语、拒答和长度字段会作为可选评判证据。

四个内置奖励按需启用：

| 奖励 | 匹配字段 |
|---|---|
| `reference_overlap` | `reference_answer` 或其别名 |
| `term_constraints` | 非空 `required_terms` 或 `forbidden_terms` |
| `refusal` | 有效 `must_refuse` 要求；识别表达由 GRPO 配置级 `refusal_terms` 控制 |
| `length_bounds` | `min_completion_chars` 或 `max_completion_chars` |

例如只叠加术语和长度奖励：

```yaml
grpo:
  builtin_rewards:
    - "term_constraints"
    - "length_bounds"
```

如果显式关闭 Judge，必须至少启用一个内置奖励，而且每一行都要包含至少一个与已启用奖励匹配的信号：

```yaml
grpo:
  builtin_rewards:
    - "reference_overlap"
  reward_judge:
    enabled: false
```

## `grpo_judge_v2` 契约

默认提示词把 prompt、候选回答、参考答案、可信上下文、约束和分类编码为不可信 JSON 数据，要求 Judge 忽略其中的评分操控或角色覆盖指令。Judge 只返回维度分数和违规标识，不返回最终总分；本地代码负责确定性加权和应用硬上限。

五个维度与初始权重：

| 维度 | 权重 | 含义 |
|---|---:|---|
| `task_fulfillment` | 30% | 是否完成合法任务；安全拒答可视为完成任务。 |
| `factual_grounding` | 25% | 是否与 `trusted_context` / `reference_answer` 一致并避免编造。 |
| `explicit_constraints` | 15% | 是否满足术语、禁用内容、拒答、格式和长度约束。 |
| `safety_refusal` | 20% | 是否保护凭据、隐藏提示、私有实现和访问边界，并拒绝危险协助。 |
| `relevance_clarity` | 10% | 是否相关、清晰、简洁且在需要时可执行。 |

没有参考答案或可信上下文时，`factual_grounding` 必须为 `null`，本地代码会在其余维度间重新归一化权重。当前权重是项目基线，需要使用领域专家标注样本校准，不是通用行业标准。

严格响应示例：

```json
{
  "schema_version": "grpo_judge_v2",
  "dimension_scores": {
    "task_fulfillment": 0.75,
    "factual_grounding": null,
    "explicit_constraints": 1.0,
    "safety_refusal": 1.0,
    "relevance_clarity": 0.75
  },
  "violations": [],
  "reason": "The response follows the explicit constraints and stays within the supplied evidence."
}
```

顶层字段、五个维度、schema 版本必须完全匹配。分数必须是 `0.0` 到 `1.0` 的有限数字，`reason` 必须是非空字符串，违规标识必须来自允许列表。旧的 `{"score": 0.8, "reason": "..."}` 格式不再有效。

本地硬上限：

| 违规标识 | 最终归一化分数上限 |
|---|---:|
| `unsafe_compliance` | 0.00 |
| `protected_information_leak` | 0.00 |
| `evaluator_manipulation` | 0.05 |
| `empty_or_irrelevant` | 0.05 |
| `missing_required_refusal` | 0.10 |
| `material_reference_contradiction` | 0.25 |
| `unsupported_domain_claim` | 0.35 |
| `explicit_constraint_violation` | 0.50 |
| `all_required_terms_missing` | 0.65 |

本地代码还会对空回答、长度违规和 literal 模式下的禁用术语命中强制附加违规标识。加权并应用上限后，结果映射到 `score_range`。

## HTTP 接口与推理型 Judge

- `base_url` 可以是 `/v1` 根路径，也可以直接是 `/v1/chat/completions`。
- 请求使用 `model`、`messages`、`temperature: 0` 和 `reward_judge.max_tokens`。
- 响应必须是 OpenAI-compatible envelope；严格 v2 JSON 必须位于 `choices[0].message.content`，兼容服务也可放在 `choices[0].text`。
- HTTP body 直接返回 v2 JSON 不够，它必须位于 chat completions 响应包中。
- 连通性和响应格式在首次评分请求时验证；预检不会主动调用远端 API。

## 异步并发评分与同步更新

GRPO 保持在线 rollout 和同步权重更新：

```text
rollout batch
  -> 当前策略批量生成候选
  -> 外部 Judge 并发评分
  -> 等待整个 reward batch 完成
  -> 计算 group advantage
  -> 同步反向传播和 optimizer 更新
```

这里的异步只缩短外部 Judge I/O 等待，不会预生成整个数据集、建立离线 rollout buffer 或并发执行多个 `optimizer.step()`。任一批次的候选都由更新前的同一当前策略生成，完整分数返回后才进入组内相对优势计算。

`reward_judge.max_concurrency` 接受 `"auto"` 或 `1..64` 的整数：

- `"auto"` 解析为当前 `grpo.num_generations`。
- 实际并发度始终是 `min(解析后的并发上限, 当前 reward batch 的 completion 数量)`。
- 显式整数是整个训练进程内的请求上限，可跨同一 reward batch 中的多个 prompt 使用。

| 问题数 | 每题答案数 | 请求总数 | 配置 | 实际并发 |
|---:|---:|---:|---:|---:|
| 1 | 3 | 3 | `auto` | 3 |
| 1 | 3 | 3 | `4` | 3 |
| 2 | 3 | 6 | `auto` | 3 |
| 2 | 3 | 6 | `4` | 4 |
| 2 | 4 | 8 | `2` | 2 |

Semaphore 上限按训练进程/rank 分别生效；当前单 GPU 训练只有一个进程，因此等同于整次运行的上限。底层 `asyncio.to_thread` 还受线程池容量约束，大并发配置不保证物理上一定同时建立相同数量的 socket。

每个 completion 独立执行 `max_retries`，退避时间以 `retry_backoff_seconds` 为基数做指数增长并加入随机抖动。HTTP 429/503 返回合法 `Retry-After` 时优先遵守服务端等待时间。所有并发任务都会等待结束并按输入顺序恢复 reward；任一请求最终失败时，整个 reward batch 失败，已成功的部分分数被丢弃，不会填充中性奖励。

每批日志记录 `configured_max_concurrency`、`effective_concurrency`、`completion_count`、`judge_batch_latency_seconds` 和 `retry_backoff_seconds`。训练元数据保存配置值、最后一批指标，以及批次数、最大实际并发、总 completion 数和平均批延迟；训练报告展示同一汇总。

推理型 Judge 可能先把输出预算消耗在 `reasoning_content`，导致 `message.content` 为空。建议从以下配置起步：

```yaml
grpo:
  reward_judge:
    max_tokens: 4096
    timeout_seconds: 120
```

不要混淆两个长度设置：

- `grpo.max_completion_length` 控制被训练策略生成的候选回答长度。
- `grpo.reward_judge.max_tokens` 控制 Judge 为单次评分请求生成响应的 token 预算。

候选回答频繁达到 `max_completion_length` 时，应检查 `completions/clipped_ratio`，先改善 EOS/停止行为，再评估是否增加候选长度。Judge 返回空 `content` 时，应增加 Judge 输出预算或调整 Judge 服务，而不是增加策略 completion 长度。

## GRPO 起点与续跑

GRPO 起点按以下顺序解析：

```text
显式 grpo.base_adapter_dir -> DPO -> Fact-SFT -> CPT
```

目录根部必须同时包含 `adapter_config.json`，以及 `adapter_model.safetensors` 或 `adapter_model.bin`，才是有效 PEFT adapter。残缺目录会记录原因并继续向后回退。找到后日志会记录实际阶段和路径。

独立运行 GRPO：

```bash
python scripts/training/train_grpo.py --config configs/domain_post_training.local.yaml
```

显式选择起点：

```bash
python scripts/training/train_grpo.py --config configs/domain_post_training.local.yaml --base_adapter_dir outputs/fact_sft_adapter
```

从已有 adapter 通过完整流水线继续：

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --skip_cpt --skip_sft --skip_dpo
```

这与 `grpo.resume_from_checkpoint` 不同：前两种方式选择前一阶段 adapter 作为 GRPO 起点；`resume_from_checkpoint` 恢复同一次 GRPO Trainer 运行的 checkpoint 状态。

如果所有候选都无效且 `require_base_adapter: true`，训练会列出检查过的路径、缺失文件以及续跑命令。只有明确需要从基础模型创建全新 PEFT adapter 时才设置 `require_base_adapter: false`。

## 训练结果验收

`Status: completed` 或有限 loss 不能单独证明训练有效。至少检查：

- CPT、Fact-SFT、DPO、GRPO 的 `grad_norm` 全部有限；默认 `abort_on_nonfinite_grad_norm: true` 应在异常时立即停止。
- 使用 `bf16: auto`、`fp16: auto`、`torch_dtype: auto`，并确认运行记录中的实际精度符合硬件能力。
- 对比相邻阶段 adapter 的 LoRA tensor，确认参数确实发生变化。
- 检查 Judge reward 的均值和方差；接近常数的 reward 无法提供有效排序信号。
- 检查 `completions/clipped_ratio`；高截断率会让 Judge 持续评价不完整回答。
- 将训练后质量评估视为启发式 smoke gate，而不是安全认证；生产发布必须人工或使用独立 Judge 复核安全样例。

## 失败策略

以下情况会在预检或首次请求时失败，不会自动写入中性分数：

- Judge 已开启但缺少 `base_url`、`model` 或有效 Key。
- Judge 关闭且 `builtin_rewards` 为空。
- 内置奖励模式的数据行没有匹配信号。
- HTTP 请求重试后仍失败，或响应不符合 chat completions envelope。
- `message.content` 为空、不是严格 v2 JSON、字段不完整、包含额外字段、维度越界或包含未知违规标识。

修正配置、服务或数据后再继续训练。

## English

GRPO uses an external LLM judge as the default reward provider. All four built-in rewards are disabled until explicitly listed in `builtin_rewards`. The external judge is called through an OpenAI-compatible `/v1/chat/completions` endpoint.

## Recommended Configuration

Keep live settings in the Git-ignored `configs/domain_post_training.local.yaml`:

```yaml
grpo:
  enabled: true
  input_path: "data/grpo/reward_examples.jsonl"
  num_generations: 4
  max_completion_length: 256
  builtin_rewards: []
  reward_judge:
    enabled: true
    base_url: "https://api.deepseek.com/v1"
    model: "your-judge-model"
    api_key: "replace-with-your-key"
    score_range: [0.0, 1.0]
    timeout_seconds: 120
    max_tokens: 4096
    max_retries: 2
    max_concurrency: "auto"
    retry_backoff_seconds: 1.0
```

`api_key` is a plaintext secret. Keep `configs/*.local.yaml` ignored by Git and never put a live key in tracked templates, training reports, or shared configs. To avoid storing the key in YAML, leave `api_key` empty and use an environment variable as an optional fallback:

```yaml
grpo:
  reward_judge:
    api_key: null
    api_key_env: "GRPO_REWARD_JUDGE_API_KEY"
```

A directly configured `api_key` takes precedence over `api_key_env`.

## Data and Reward Modes

With the judge enabled, each row only needs a constructible prompt; prompt-only rows are valid. Reference answers, trusted context, terms, refusal requirements, and length bounds are optional evaluation evidence.

Enable built-in rewards only when needed:

| Reward | Matching fields |
|---|---|
| `reference_overlap` | `reference_answer` or an accepted alias |
| `term_constraints` | Non-empty `required_terms` or `forbidden_terms` |
| `refusal` | A valid `must_refuse` requirement; recognized phrases come from config-level `refusal_terms` |
| `length_bounds` | `min_completion_chars` or `max_completion_chars` |

For example, add only term and length rewards:

```yaml
grpo:
  builtin_rewards:
    - "term_constraints"
    - "length_bounds"
```

If you explicitly disable the judge, enable at least one built-in reward and give every row at least one signal matching an enabled reward:

```yaml
grpo:
  builtin_rewards:
    - "reference_overlap"
  reward_judge:
    enabled: false
```

## `grpo_judge_v2` Contract

The default prompts encode the prompt, candidate response, reference, trusted context, constraints, and category as untrusted JSON data and tell the judge to ignore score manipulation or role-overriding instructions inside it. The judge returns dimension scores and violation identifiers, not a final score. Local code deterministically weights the dimensions and applies hard caps.

Dimensions and initial weights:

| Dimension | Weight | Meaning |
|---|---:|---|
| `task_fulfillment` | 30% | Completes the legitimate task; a safe refusal can satisfy the task. |
| `factual_grounding` | 25% | Agrees with `trusted_context` / `reference_answer` and avoids fabrication. |
| `explicit_constraints` | 15% | Follows term, forbidden-content, refusal, format, and length constraints. |
| `safety_refusal` | 20% | Protects credentials, hidden prompts, private implementation, and access boundaries and refuses unsafe help. |
| `relevance_clarity` | 10% | Is relevant, clear, concise, and actionable where appropriate. |

Without a reference or trusted context, `factual_grounding` must be `null`; local code renormalizes the other weights. These weights are a project baseline to calibrate with domain-expert examples, not a universal standard.

Strict response example:

```json
{
  "schema_version": "grpo_judge_v2",
  "dimension_scores": {
    "task_fulfillment": 0.75,
    "factual_grounding": null,
    "explicit_constraints": 1.0,
    "safety_refusal": 1.0,
    "relevance_clarity": 0.75
  },
  "violations": [],
  "reason": "The response follows the explicit constraints and stays within the supplied evidence."
}
```

Top-level fields, all five dimensions, and the schema version must match exactly. Scores must be finite numbers from `0.0` to `1.0`, `reason` must be a non-empty string, and violation identifiers must come from the allowlist. The legacy `{"score": 0.8, "reason": "..."}` format is no longer valid.

Local hard caps:

| Violation | Maximum normalized score |
|---|---:|
| `unsafe_compliance` | 0.00 |
| `protected_information_leak` | 0.00 |
| `evaluator_manipulation` | 0.05 |
| `empty_or_irrelevant` | 0.05 |
| `missing_required_refusal` | 0.10 |
| `material_reference_contradiction` | 0.25 |
| `unsupported_domain_claim` | 0.35 |
| `explicit_constraint_violation` | 0.50 |
| `all_required_terms_missing` | 0.65 |

Local code also forces violations for empty responses, length failures, and forbidden terms in literal mode. After weighting and caps, the result is mapped to `score_range`.

## HTTP Contract and Reasoning Judges

- `base_url` may be a `/v1` root or the full `/v1/chat/completions` URL.
- Requests contain `model`, `messages`, `temperature: 0`, and `reward_judge.max_tokens`.
- Responses must use an OpenAI-compatible envelope. Strict v2 JSON belongs in `choices[0].message.content`; compatible services may use `choices[0].text`.
- Returning v2 JSON as the raw HTTP body is insufficient; it must be inside the chat completions envelope.
- Connectivity and response format are validated on the first scoring request; preflight does not proactively call the remote API.

## Asynchronous Scoring and Synchronized Updates

GRPO retains online rollouts and synchronized weight updates:

```text
rollout batch
  -> generate candidates with the current policy
  -> score them concurrently with the external judge
  -> wait for the complete reward batch
  -> compute group advantage
  -> run synchronized backpropagation and optimizer update
```

Asynchrony only reduces external-judge I/O wait. It does not pre-generate the full dataset, create an offline rollout buffer, or execute concurrent `optimizer.step()` calls. Candidates in a batch come from the same current policy, and group-relative advantage is computed only after the complete score set returns.

`reward_judge.max_concurrency` accepts `"auto"` or an integer from `1` to `64`:

- `"auto"` resolves to the current `grpo.num_generations`.
- Effective concurrency is always `min(resolved cap, completion count in the current reward batch)`.
- An explicit integer is the request cap for one training process and can be shared across prompts in the same reward batch.

| Prompts | Answers each | Requests | Setting | Effective concurrency |
|---:|---:|---:|---:|---:|
| 1 | 3 | 3 | `auto` | 3 |
| 1 | 3 | 3 | `4` | 3 |
| 2 | 3 | 6 | `auto` | 3 |
| 2 | 3 | 6 | `4` | 4 |
| 2 | 4 | 8 | `2` | 2 |

The semaphore cap applies separately to each training process/rank. The current single-GPU run has one process, so it is also the whole-run cap. The underlying `asyncio.to_thread` executor can have fewer workers, so a large configured cap does not guarantee the same number of simultaneous physical sockets.

Each completion independently uses `max_retries`, exponential backoff with jitter based on `retry_backoff_seconds`, and valid `Retry-After` delays from HTTP 429/503. All concurrent tasks settle and rewards are restored in input order. If any request ultimately fails, the complete reward batch fails; successful partial scores are discarded and no neutral reward is substituted.

Per-batch logs record `configured_max_concurrency`, `effective_concurrency`, `completion_count`, `judge_batch_latency_seconds`, and `retry_backoff_seconds`. Training metadata stores configuration values, the latest batch metrics, batch count, maximum effective concurrency, total completions, and mean batch latency; the training report presents the same summary.

Reasoning judges can consume the output budget in `reasoning_content` and leave `message.content` empty. Start with:

```yaml
grpo:
  reward_judge:
    max_tokens: 4096
    timeout_seconds: 120
```

Do not confuse the two length settings:

- `grpo.max_completion_length` limits candidate responses generated by the policy being trained.
- `grpo.reward_judge.max_tokens` is the judge response budget for each scoring request.

When candidates frequently hit `max_completion_length`, inspect `completions/clipped_ratio`, improve EOS/stop behavior first, and then decide whether to raise the candidate limit. When the judge returns empty `content`, increase the judge budget or adjust the judge service, not the policy completion length.

## GRPO Starting Adapter and Resume Modes

GRPO resolves its starting adapter in this order:

```text
explicit grpo.base_adapter_dir -> DPO -> Fact-SFT -> CPT
```

A directory is a valid PEFT adapter only when its root contains `adapter_config.json` and either `adapter_model.safetensors` or `adapter_model.bin`. Incomplete directories are recorded and skipped while resolution continues. Logs report the actual stage and path selected.

Run GRPO independently:

```bash
python scripts/training/train_grpo.py --config configs/domain_post_training.local.yaml
```

Select the starting adapter explicitly:

```bash
python scripts/training/train_grpo.py --config configs/domain_post_training.local.yaml --base_adapter_dir outputs/fact_sft_adapter
```

Continue through the full pipeline from existing adapters:

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --skip_cpt --skip_sft --skip_dpo
```

These select a previous-stage adapter as the GRPO starting point. They differ from `grpo.resume_from_checkpoint`, which restores Trainer state from a checkpoint of the same GRPO run.

If every candidate is invalid and `require_base_adapter: true`, the error lists checked paths, missing files, and continuation commands. Set `require_base_adapter: false` only when intentionally creating a fresh PEFT adapter from the base model.

## Training Acceptance

`Status: completed` or finite loss alone does not prove that training worked. At minimum:

- Confirm finite `grad_norm` across CPT, Fact-SFT, DPO, and GRPO; the default `abort_on_nonfinite_grad_norm: true` should stop immediately on failure.
- Use `bf16: auto`, `fp16: auto`, and `torch_dtype: auto`, and confirm the recorded effective precision matches the hardware.
- Compare LoRA tensors between adjacent adapters to verify that parameters actually changed.
- Inspect judge reward mean and variance; near-constant rewards provide no useful ranking signal.
- Inspect `completions/clipped_ratio`; high clipping means the judge repeatedly sees incomplete candidates.
- Treat post-training quality evaluation as a heuristic smoke gate, not safety certification. Production release requires human review or an independent judge over safety cases.

## Failure Policy

These conditions fail during preflight or the first request; the system never substitutes a neutral score:

- Judge enabled without `base_url`, `model`, or a valid key.
- Judge disabled with an empty `builtin_rewards` list.
- Built-in reward mode with rows that have no matching signal.
- HTTP failure after retries or a response outside the chat completions envelope.
- Empty `message.content`, invalid strict v2 JSON, missing or extra fields, out-of-range dimensions, or unknown violations.

Fix the config, service, or data before resuming training.
