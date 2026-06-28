# GRPO 与 Reward Judge / GRPO And Reward Judge

## 中文

当你要启用 GRPO 或基于外部模型做奖励评分时，使用本页。

中文提示：本项目不再把本地 reward model 路径或 Hugging Face Hub ID 直接交给 TRL。模型评分统一通过 OpenAI-compatible HTTP 接口完成。

## GRPO 输入要求

每条 GRPO 数据必须包含：

- `prompt`
- 至少一个奖励信号：`reference_answer`、`required_terms`、`forbidden_terms` 或 `must_refuse`

可选字段 `min_completion_chars`、`max_completion_chars`、`category` 可以改善奖励行为和报告可读性。

## 启用 GRPO

```yaml
grpo:
  enabled: true
  input_path: "data/grpo/reward_examples.jsonl"
  num_generations: 4
  builtin_rewards:
    - "reference_overlap"
    - "term_constraints"
    - "refusal"
    - "length_bounds"
```

只运行 GRPO 阶段：

```bash
python scripts/training/train_grpo.py --config configs/domain_post_training.yaml --max_steps 10
```

## 内置奖励

| 奖励 | 使用字段 |
|---|---|
| `reference_overlap` | `reference_answer` |
| `term_constraints` | `required_terms`、`forbidden_terms` |
| `refusal` | `must_refuse`、`refusal_terms` |
| `length_bounds` | `min_completion_chars`、`max_completion_chars` |

只启用数据中确实有字段支撑的奖励。

## 外部 reward judge

可选外部 judge 配置在 `grpo.reward_judge`：

```yaml
grpo:
  reward_judge:
    enabled: true
    base_url: "http://localhost:8000/v1"
    api_key_env: "GRPO_REWARD_JUDGE_API_KEY"
    model: "local-reward-judge"
    score_range: [0.0, 1.0]
    timeout_seconds: 30
    max_retries: 2
```

行为规则：

- judge 是单一 OpenAI-compatible chat completions client。
- 本地或托管服务必须在配置的 `base_url` 下暴露 `/chat/completions`。
- judge 必须返回包含数值 `score` 的 JSON，例如 `{"score": 0.8, "reason": "..."}`。
- 程序只读取 `score`，并按 `score_range` clamp。
- API key 只从 `api_key_env` 指定的环境变量读取。
- API key 值不会写入训练元数据。

## 本地 judge 服务

本地 judge 模型必须先部署成 OpenAI-compatible HTTP API，再启动 GRPO。

可接受的本地 base URL 示例：

```yaml
base_url: "http://localhost:8000/v1"
base_url: "http://localhost:8000/v1/chat/completions"
```

训练前设置环境变量：

```bash
export GRPO_REWARD_JUDGE_API_KEY="local-dev-key"
```

Windows PowerShell:

```powershell
$env:GRPO_REWARD_JUDGE_API_KEY = "local-dev-key"
```

## DeepSeek judge 示例

```yaml
grpo:
  reward_judge:
    enabled: true
    base_url: "https://api.deepseek.com"
    api_key_env: "DEEPSEEK_API_KEY"
    model: "deepseek-v4-flash"
```

## GLM judge 示例

```yaml
grpo:
  reward_judge:
    enabled: true
    base_url: "https://open.bigmodel.cn/api/paas/v4"
    api_key_env: "ZAI_API_KEY"
    model: "glm-5.2"
```

## 失败策略

Reward judge 失败时默认 fail closed，也就是直接报错，不静默给中性分。以下情况都会失败：

- 缺少 `base_url`
- 缺少 `model`
- 缺少 API key 环境变量
- HTTP 请求重试后仍失败
- 响应不是可解析 JSON
- 缺少 `score` 或 `score` 不是数值

继续训练前应修正 judge 服务或配置。这个设计是为了避免评分失控或无声劣化。

## 相关页面

- [数据契约](Data-Contracts)
- [配置](Configuration)
- [训练流水线](Training-Pipeline)
- [故障排查](Troubleshooting)

---

## English

Use this page when enabling GRPO or model-based reward scoring.

Note: this project no longer passes local reward model paths or Hugging Face Hub IDs directly to TRL. Model-based reward scoring is standardized through an OpenAI-compatible HTTP API.

## GRPO Input Requirements

Each GRPO row must contain:

- `prompt`
- at least one reward signal: `reference_answer`, `required_terms`, `forbidden_terms`, or `must_refuse`

Optional fields such as `min_completion_chars`, `max_completion_chars`, and `category` can improve reward behavior and reporting.

## Enable GRPO

```yaml
grpo:
  enabled: true
  input_path: "data/grpo/reward_examples.jsonl"
  num_generations: 4
  builtin_rewards:
    - "reference_overlap"
    - "term_constraints"
    - "refusal"
    - "length_bounds"
```

Run only the GRPO stage:

```bash
python scripts/training/train_grpo.py --config configs/domain_post_training.yaml --max_steps 10
```

## Built-in Rewards

| Reward | Uses |
|---|---|
| `reference_overlap` | `reference_answer` |
| `term_constraints` | `required_terms`, `forbidden_terms` |
| `refusal` | `must_refuse`, `refusal_terms` |
| `length_bounds` | `min_completion_chars`, `max_completion_chars` |

Enable only rewards represented by your dataset fields.

## External Reward Judge

The optional external judge is configured under `grpo.reward_judge`:

```yaml
grpo:
  reward_judge:
    enabled: true
    base_url: "http://localhost:8000/v1"
    api_key_env: "GRPO_REWARD_JUDGE_API_KEY"
    model: "local-reward-judge"
    score_range: [0.0, 1.0]
    timeout_seconds: 30
    max_retries: 2
```

Behavior:

- The judge is a single OpenAI-compatible chat completions client.
- The local or hosted service must expose `/chat/completions` under the configured `base_url`.
- The judge must return JSON containing a numeric `score`, such as `{"score": 0.8, "reason": "..."}`.
- The program reads only `score` and clamps it to `score_range`.
- API keys are read from the environment variable named by `api_key_env`.
- API key values are not written to training metadata.

## Local Judge Service

A local judge model must be deployed behind an OpenAI-compatible HTTP API before GRPO starts.

Acceptable local base URL examples:

```yaml
base_url: "http://localhost:8000/v1"
base_url: "http://localhost:8000/v1/chat/completions"
```

Set an environment variable before training:

```bash
export GRPO_REWARD_JUDGE_API_KEY="local-dev-key"
```

Windows PowerShell:

```powershell
$env:GRPO_REWARD_JUDGE_API_KEY = "local-dev-key"
```

## DeepSeek Judge Example

```yaml
grpo:
  reward_judge:
    enabled: true
    base_url: "https://api.deepseek.com"
    api_key_env: "DEEPSEEK_API_KEY"
    model: "deepseek-v4-flash"
```

## GLM Judge Example

```yaml
grpo:
  reward_judge:
    enabled: true
    base_url: "https://open.bigmodel.cn/api/paas/v4"
    api_key_env: "ZAI_API_KEY"
    model: "glm-5.2"
```

## Failure Policy

Reward judge failures are fail-closed:

- missing `base_url`
- missing `model`
- missing API key environment variable
- HTTP failure after retries
- invalid JSON response
- missing or non-numeric `score`

Fix the judge service or config before continuing. Silent neutral scoring is intentionally avoided.

## Related Pages

- [Data Contracts](Data-Contracts)
- [Configuration](Configuration)
- [Training Pipeline](Training-Pipeline)
- [Troubleshooting](Troubleshooting)
