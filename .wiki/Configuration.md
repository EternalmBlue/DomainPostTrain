# 配置 / Configuration

## 中文

当你要把 DomainPostTrain 改成自己的领域、调整阶段输出或修改训练行为时，使用本页。

完整参数参考在 `configs/README.md`。本页只覆盖最常改的配置。

## 主配置文件

默认配置：

```text
configs/domain_post_training.yaml
```

推荐工作流：

```bash
cp configs/domain_post_training.yaml configs/my_domain.yaml
```

然后运行：

```bash
python scripts/training/train_pipeline.py --config configs/my_domain.yaml
```

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

## 阶段开关

```yaml
fact_sft:
  enabled: true

dpo:
  enabled: false

grpo:
  enabled: false
```

完整流水线先运行 CPT，再运行已启用的后续阶段。也可以通过命令行跳过阶段，见 [训练流水线](Training-Pipeline)。

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

## 验证集与质量评估

训练验证集和训练后质量评估不是一回事：

- 验证集：训练过程中用于 loss/eval 信号。
- 质量评估：训练后检查事实回答、安全拒答和基础能力回归。

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

Recommended workflow:

```bash
cp configs/domain_post_training.yaml configs/my_domain.yaml
```

Then run commands with:

```bash
python scripts/training/train_pipeline.py --config configs/my_domain.yaml
```

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

## Stage Switches

```yaml
fact_sft:
  enabled: true

dpo:
  enabled: false

grpo:
  enabled: false
```

The full pipeline starts with CPT, then runs enabled later stages. You can also skip stages from the command line. See [Training Pipeline](Training-Pipeline).

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

## Validation and Quality Evaluation

Training validation and post-training quality evaluation are different:

- Validation set: used during training for loss/eval signal.
- Quality evaluation: run after training to check factual answers, safe refusals, and regressions.

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
