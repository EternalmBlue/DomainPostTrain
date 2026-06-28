# GRPO And Reward Judge

Use this page when enabling GRPO or model-based reward scoring.

中文提示: 本项目不再把本地 reward model 路径或 Hub ID 直接交给 TRL. 模型评分统一走 OpenAI-compatible HTTP 接口。

## GRPO input requirements

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

## Built-in rewards

| Reward | Uses |
|---|---|
| `reference_overlap` | `reference_answer` |
| `term_constraints` | `required_terms`, `forbidden_terms` |
| `refusal` | `must_refuse`, `refusal_terms` |
| `length_bounds` | `min_completion_chars`, `max_completion_chars` |

Enable only rewards represented by your dataset fields.

## External reward judge

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

## Local judge service

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

## DeepSeek judge example

```yaml
grpo:
  reward_judge:
    enabled: true
    base_url: "https://api.deepseek.com"
    api_key_env: "DEEPSEEK_API_KEY"
    model: "deepseek-v4-flash"
```

## GLM judge example

```yaml
grpo:
  reward_judge:
    enabled: true
    base_url: "https://open.bigmodel.cn/api/paas/v4"
    api_key_env: "ZAI_API_KEY"
    model: "glm-5.2"
```

## Failure policy

Reward judge failures are fail-closed:

- missing `base_url`
- missing `model`
- missing API key environment variable
- HTTP failure after retries
- invalid JSON response
- missing or non-numeric `score`

Fix the judge service or config before continuing. Silent neutral scoring is intentionally avoided.

## Related pages

- [Data Contracts](Data-Contracts)
- [Configuration](Configuration)
- [Training Pipeline](Training-Pipeline)
- [Troubleshooting](Troubleshooting)

