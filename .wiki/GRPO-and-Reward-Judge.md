# GRPO 与 Reward Judge / GRPO And Reward Judge

## 中文

当你要启用 GRPO 或基于外部模型做奖励评分时，使用本页。

GRPO 的模型型奖励只通过 OpenAI-compatible HTTP judge 调用。无论 judge 是本地部署模型、DeepSeek、GLM 还是其他托管服务，都需要提供 chat completions API。

## GRPO 输入要求

每条 GRPO 数据必须能构造出 prompt，并且至少包含一种奖励信号：

- `reference_answer`
- `required_terms`
- `forbidden_terms`
- `must_refuse`

可选字段 `min_completion_chars`、`max_completion_chars`、`category` 可以改善奖励行为和报告可读性。字段别名见 [数据契约](Data-Contracts)。

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

如果没有 DPO adapter，把 GRPO 起点指向 Fact-SFT adapter：

```bash
python scripts/training/train_grpo.py --config configs/domain_post_training.yaml --base_adapter_dir outputs/fact_sft_adapter
```

## 内置奖励

| 奖励 | 使用字段 |
|---|---|
| `reference_overlap` | `reference_answer` |
| `term_constraints` | `required_terms`、`forbidden_terms` |
| `refusal` | `must_refuse`、`refusal_terms` |
| `length_bounds` | `min_completion_chars`、`max_completion_chars` |

启用奖励前，确认数据包含对应字段。

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

- `base_url` 可以是 `/v1` 根路径，也可以直接是 `/v1/chat/completions`。
- 程序按 OpenAI chat completions wire format 发送 `model`、`messages`、`temperature: 0`、`max_tokens: 256`。
- HTTP 响应必须是 OpenAI-compatible envelope，程序读取 `choices[0].message.content`；如果没有 `message.content`，才读取 `choices[0].text`。
- `message.content` 本身必须是 JSON 对象，形如 `{"score": 0.8, "reason": "..."}`。程序只读取 `score`，并按 `score_range` clamp。
- API key 只从 `api_key_env` 指定的环境变量读取，训练元数据只记录环境变量名，不记录密钥值。

HTTP 响应示例：

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "{\"score\": 0.8, \"reason\": \"Matches the reference and avoids forbidden terms.\"}"
      }
    }
  ]
}
```

注意：HTTP body 直接返回 `{"score": 0.8}` 不符合接口要求；它必须位于 chat completions envelope 的 assistant content 中。

## 本地 judge 服务

本地 judge 模型需要先部署成 OpenAI-compatible HTTP API，再启动 GRPO。

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

训练前可以先用最小请求验证 judge：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-dev-key" \
  -d '{
    "model": "local-reward-judge",
    "messages": [
      {"role": "system", "content": "Return only JSON."},
      {"role": "user", "content": "Score this answer. Return {\"score\": 0.5, \"reason\": \"test\"}."}
    ],
    "temperature": 0,
    "max_tokens": 64
  }'
```

确认响应中的 assistant content 可以解析出数值 `score`。

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

Reward judge 异常时训练会失败；系统不会自动写入中性分。以下情况都会失败：

- 缺少 `base_url`
- 缺少 `model`
- 缺少 API key 环境变量
- HTTP 请求重试后仍失败
- HTTP 响应不是 OpenAI-compatible chat completions envelope
- assistant content 不是可解析 JSON
- 缺少 `score` 或 `score` 不是数值

继续训练前应修正 judge 服务或配置。

## 相关页面

- [数据契约](Data-Contracts)
- [配置](Configuration)
- [训练流水线](Training-Pipeline)
- [操作手册](Operations-Runbook)
- [故障排查](Troubleshooting)

---

## English

Use this page when enabling GRPO or model-based reward scoring.

Model-based GRPO rewards are called only through an OpenAI-compatible HTTP judge. Whether the judge is a local model, DeepSeek, GLM, or another hosted service, it must provide a chat completions API.

## GRPO Input Requirements

Each GRPO row must allow the loader to build a prompt and must contain at least one reward signal:

- `reference_answer`
- `required_terms`
- `forbidden_terms`
- `must_refuse`

Optional fields such as `min_completion_chars`, `max_completion_chars`, and `category` can improve reward behavior and reporting. See [Data Contracts](Data-Contracts) for accepted aliases.

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

If there is no DPO adapter, point GRPO at the Fact-SFT adapter:

```bash
python scripts/training/train_grpo.py --config configs/domain_post_training.yaml --base_adapter_dir outputs/fact_sft_adapter
```

## Built-in Rewards

| Reward | Uses |
|---|---|
| `reference_overlap` | `reference_answer` |
| `term_constraints` | `required_terms`, `forbidden_terms` |
| `refusal` | `must_refuse`, `refusal_terms` |
| `length_bounds` | `min_completion_chars`, `max_completion_chars` |

Before enabling a reward, confirm that the dataset contains the corresponding fields.

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

- `base_url` may be the `/v1` root or the full `/v1/chat/completions` URL.
- The program sends OpenAI chat completions wire format with `model`, `messages`, `temperature: 0`, and `max_tokens: 256`.
- The HTTP response must be an OpenAI-compatible envelope. The program reads `choices[0].message.content`; if that is absent, it reads `choices[0].text`.
- `message.content` itself must be a JSON object such as `{"score": 0.8, "reason": "..."}`. The program reads only `score` and clamps it to `score_range`.
- API keys are read from the environment variable named by `api_key_env`; training metadata records only the variable name, not the secret value.

HTTP response example:

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "{\"score\": 0.8, \"reason\": \"Matches the reference and avoids forbidden terms.\"}"
      }
    }
  ]
}
```

Returning `{"score": 0.8}` as the raw HTTP body is not enough; it must be inside the assistant content of a chat completions envelope.

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

Before training, test the judge with a minimal request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-dev-key" \
  -d '{
    "model": "local-reward-judge",
    "messages": [
      {"role": "system", "content": "Return only JSON."},
      {"role": "user", "content": "Score this answer. Return {\"score\": 0.5, \"reason\": \"test\"}."}
    ],
    "temperature": 0,
    "max_tokens": 64
  }'
```

Confirm that the assistant content can be parsed as a numeric `score`.

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

Reward judge errors fail the training run; the system does not write a neutral score automatically. These cases fail:

- missing `base_url`
- missing `model`
- missing API key environment variable
- HTTP failure after retries
- HTTP response is not an OpenAI-compatible chat completions envelope
- assistant content is not parseable JSON
- missing or non-numeric `score`

Fix the judge service or config before continuing.

## Related Pages

- [Data Contracts](Data-Contracts)
- [Configuration](Configuration)
- [Training Pipeline](Training-Pipeline)
- [Operations Runbook](Operations-Runbook)
- [Troubleshooting](Troubleshooting)
