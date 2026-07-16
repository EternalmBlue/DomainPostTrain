# 配置 / Configuration

## 中文

当你要把 DomainPostTrain 改成自己的领域、调整阶段输出或修改训练行为时，使用本页。

完整参数参考在 `configs/README.md`。本页只覆盖最常改的配置。

## 主配置文件

默认配置：

```text
configs/domain_post_training.yaml
```

推荐工作流是创建 Git 忽略的本地配置，真实 Judge Key 只放在这个文件中：

```powershell
Copy-Item configs/domain_post_training.yaml configs/domain_post_training.local.yaml
```

然后运行：

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml
```

`configs/*.local.yaml` 应保持在 Git 忽略列表中。`api_key` 是明文凭据；不要提交、上传、粘贴到日志或共享训练配置。环境变量仍可作为可选回退，但不是默认配置方式。

## 路径解析规则

- 训练产物通常相对仓库根目录解析，例如 `outputs/lora_adapter`。
- 数据输入路径通常优先相对配置文件解析。默认配置中的 `../data/...` 指向仓库级 `data/...`。
- `null` 表示让代码使用默认行为。

## 优先替换这些配置

| 配置项 | 为什么重要 |
|---|---|
| `base_model_repo_id` | `download_models.py` 使用的 Hugging Face Hub 仓库。 |
| `base_model_name_or_path` | 训练、合并和推理实际加载的本地路径或模型 ID。 |
| `corpus.input_paths` | CPT 领域源文档。 |
| `fact_sft.input_paths` | Fact-SFT JSONL 数据。 |
| `dpo.input_path` | DPO 偏好数据，启用 DPO 时使用。 |
| `grpo.input_path` | GRPO 奖励提示数据，启用 GRPO 时使用。 |
| `eval.question_file` | 训练后质量评估问题。 |
| `fact_sft.system_prompt` | 领域角色、知识边界和安全边界。 |

`base_model_repo_id` 和 `base_model_name_or_path` 不重复：前者只告诉 `download_models.py` 从哪里下载，后者告诉训练、合并和推理从哪里加载。默认组合会把 Hub 模型下载到 `models/base-model`，之后训练从这个本地目录读取。

## 阶段开关

```yaml
fact_sft:
  enabled: true

dpo:
  enabled: true

grpo:
  enabled: true
  builtin_rewards: []
  reward_judge:
    enabled: true
    base_url: "https://your-judge.example/v1"
    model: "your-judge-model"
    api_key: "replace-with-your-key"
    timeout_seconds: 120
    max_retries: 2
    max_concurrency: "auto"
    retry_backoff_seconds: 1.0
    max_tokens: 4096
```

完整流水线先运行 CPT，再运行已启用的后续阶段。默认模板开启 DPO 和 GRPO，因此真实训练前必须准备对应数据并配置 Judge。Judge 默认开启，四个内置奖励默认不启用；如果显式关闭 Judge，`builtin_rewards` 必须至少包含一个与数据字段匹配的奖励。也可以通过命令行跳过阶段，见 [训练流水线](Training-Pipeline)。

`max_concurrency: "auto"` 默认解析为 `num_generations`，每批实际并发为 `min(并发上限, completion 数量)`。显式整数可让同一 reward batch 中的多个 prompt 共享更高并发上限；评分整批完成后才计算 group advantage 并同步更新权重。该上限按训练进程/rank 生效，线程池容量还可能让实际连接数更低。每个请求独立退避重试，429/503 的合法 `Retry-After` 会被遵守，任一最终失败会使整批 reward 失败。

如果不希望在 YAML 中保存 Key，可以把 `api_key` 留空，并设置可选回退：

```yaml
grpo:
  reward_judge:
    api_key: null
    api_key_env: "GRPO_REWARD_JUDGE_API_KEY"
```

## 显存敏感配置

显存紧张时，优先降低序列长度、LoRA rank，启用 4-bit 加载和 gradient checkpointing：

```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  max_seq_length: 768
  load_in_4bit: true
  gradient_checkpointing: true

peft:
  r: 8
  lora_alpha: 8
```

四个训练阶段建议让精度自动匹配硬件，并对非有限梯度立即失败：

```yaml
training:
  bf16: auto
  fp16: auto
  torch_dtype: auto
  abort_on_nonfinite_grad_norm: true
```

Fact-SFT、DPO 和 GRPO 的同名字段遵循相同行为：支持 BF16 时优先 BF16，否则在 CUDA 上回退 FP16；CPU 不开启两者。自动选择只是硬件兼容默认值，训练后仍要检查 `grad_norm` 和 adapter 是否真实更新。

## 验证集与质量评估

训练验证集和训练后质量评估不是一回事：

- 验证集：训练过程中用于 loss/eval 信号。
- 质量评估：训练后检查事实回答、安全拒答和基础能力回归。

内置质量评估是启发式 smoke gate，不是生产级安全认证。发布模型前仍需使用独立 Judge 或人工评审复核安全样例。

评估和实际推理共用 chat template、`enable_thinking: false`、reasoning 清理与纯文本规则。默认 `eval.quality_gate` 要求三个类别全部通过；门禁失败保留产物并使完整流水线返回 `8`。`fail_pipeline: false` 只降级为警告，`--skip_eval` 则记录 `not_evaluated`，两者都不会把产物标记为 release-ready。

mock 数据很小，所以默认关闭验证：

```yaml
corpus:
  validation_mode: "none"
fact_sft:
  validation_ratio: 0
dpo:
  validation_ratio: 0
```

真实项目建议使用独立 held-out CPT 文档：

```yaml
corpus:
  validation_mode: "separate_sources"
  validation_sources:
    - "../data/cpt_validation/source_documents"
```

## 相关页面

- [数据契约](Data-Contracts)
- [训练流水线](Training-Pipeline)
- [GRPO 与 Reward Judge](GRPO-and-Reward-Judge)
- [故障排查](Troubleshooting)

---

## English

Use this page when adapting DomainPostTrain to another domain, changing stage outputs, or tuning training behavior.

The full parameter reference lives in `configs/README.md`. This page focuses on the settings users usually edit first.

## Main Config File

The default config is:

```text
configs/domain_post_training.yaml
```

Create a Git-ignored local config and keep the real judge key only in that file:

```powershell
Copy-Item configs/domain_post_training.yaml configs/domain_post_training.local.yaml
```

Then run commands with:

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml
```

Keep `configs/*.local.yaml` ignored by Git. `api_key` is a plaintext credential; never commit it, upload it, print it in logs, or share the live training config. Environment variables remain an optional fallback, not the primary configuration path.

## Path Rules

- Training artifacts usually resolve relative to the repository root, such as `outputs/lora_adapter`.
- Data input paths are usually resolved relative to the config file first. In the default config, `../data/...` points back to repository-level `data/...`.
- `null` means the code should use its default behavior.

## Replace These First

| Setting | Why it matters |
|---|---|
| `base_model_repo_id` | Hugging Face Hub repo used by `download_models.py`. |
| `base_model_name_or_path` | Actual local path or model id loaded for training, merge, and inference. |
| `corpus.input_paths` | CPT source documents for the domain. |
| `fact_sft.input_paths` | Fact-SFT JSONL data. |
| `dpo.input_path` | DPO preference data, if DPO is enabled. |
| `grpo.input_path` | GRPO reward-prompt data, if GRPO is enabled. |
| `eval.question_file` | Post-training quality evaluation questions. |
| `fact_sft.system_prompt` | Domain role, knowledge boundary, and safety boundary. |

`base_model_repo_id` and `base_model_name_or_path` are not duplicates. The first tells `download_models.py` what to download; the second tells training, merge, and inference what to load. The default pairing downloads the Hub snapshot into `models/base-model` and then trains from that local directory.

## Stage Switches

```yaml
fact_sft:
  enabled: true

dpo:
  enabled: true

grpo:
  enabled: true
  builtin_rewards: []
  reward_judge:
    enabled: true
    base_url: "https://your-judge.example/v1"
    model: "your-judge-model"
    api_key: "replace-with-your-key"
    timeout_seconds: 120
    max_retries: 2
    max_concurrency: "auto"
    retry_backoff_seconds: 1.0
    max_tokens: 4096
```

The full pipeline starts with CPT and then runs enabled later stages. The default template enables DPO and GRPO, so real training requires their datasets and a configured judge. The judge is enabled by default while all four built-in rewards are opt-in. If you explicitly disable the judge, `builtin_rewards` must contain at least one reward backed by the row data. You can also skip stages from the command line. See [Training Pipeline](Training-Pipeline).

`max_concurrency: "auto"` resolves to `num_generations`, and effective concurrency for each batch is `min(cap, completion count)`. An explicit integer lets prompts in the same reward batch share a larger cap; group advantage and synchronized weight updates run only after the complete score set returns. The cap applies per training process/rank, and executor capacity may make physical connection concurrency lower. Requests retry independently with backoff, valid `Retry-After` values on 429/503 are honored, and one final request failure fails the complete reward batch.

To avoid storing the key in YAML, leave `api_key` empty and configure the optional fallback:

```yaml
grpo:
  reward_judge:
    api_key: null
    api_key_env: "GRPO_REWARD_JUDGE_API_KEY"
```

## Memory-Sensitive Defaults

When GPU memory is tight, start with smaller sequence length, smaller LoRA rank, 4-bit loading, and gradient checkpointing:

```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  max_seq_length: 768
  load_in_4bit: true
  gradient_checkpointing: true

peft:
  r: 8
  lora_alpha: 8
```

Let all four training stages select precision for the hardware and fail on non-finite gradients:

```yaml
training:
  bf16: auto
  fp16: auto
  torch_dtype: auto
  abort_on_nonfinite_grad_norm: true
```

Fact-SFT, DPO, and GRPO use the same stage-level fields: BF16 is preferred when supported, CUDA otherwise falls back to FP16, and CPU enables neither. Automatic selection is only a compatibility default; after training, still inspect `grad_norm` and verify that adapter tensors actually changed.

## Validation and Quality Evaluation

Training validation and post-training quality evaluation are different:

- Validation set: used during training for loss/eval signal.
- Quality evaluation: run after training to check factual answers, safe refusals, and regressions.

The built-in quality evaluation is a heuristic smoke gate, not production safety certification. Independently judge or manually review safety cases before release.

Evaluation shares the deployed chat template, `enable_thinking: false`, reasoning cleanup, and plain-text rules. The default `eval.quality_gate` requires all three categories to pass; a failed gate retains artifacts and makes the full pipeline exit `8`. `fail_pipeline: false` only downgrades failure to a warning, while `--skip_eval` records `not_evaluated`; neither marks artifacts release-ready.

The mock data is small, so validation is disabled by default:

```yaml
corpus:
  validation_mode: "none"
fact_sft:
  validation_ratio: 0
dpo:
  validation_ratio: 0
```

For real projects, prefer independent held-out CPT documents:

```yaml
corpus:
  validation_mode: "separate_sources"
  validation_sources:
    - "../data/cpt_validation/source_documents"
```

## Related Pages

- [Data Contracts](Data-Contracts)
- [Training Pipeline](Training-Pipeline)
- [GRPO And Reward Judge](GRPO-and-Reward-Judge)
- [Troubleshooting](Troubleshooting)
