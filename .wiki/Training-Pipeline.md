# 训练流水线 / Training Pipeline

## 中文

当你要运行、跳过或调试训练阶段时，使用本页。

## 阶段顺序

```text
CPT -> Fact-SFT -> optional DPO -> optional GRPO -> merge -> quality eval
```

流水线先产出 PEFT adapter，再把选定 adapter 合并成完整 Hugging Face 模型。

## 完整流水线

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml
```

主要产物：

| 产物 | 何时出现 |
|---|---|
| `outputs/lora_adapter/` | CPT adapter。 |
| `outputs/fact_sft_adapter/` | Fact-SFT adapter。 |
| `outputs/dpo_adapter/` | `dpo.enabled=true` 时的 DPO adapter。 |
| `outputs/grpo_adapter/` | `grpo.enabled=true` 时的 GRPO adapter。 |
| `outputs/merged_model/` | 合并模型。 |
| `outputs/cpt_dataset/coverage_report.md` | CPT 覆盖度报告。 |
| `outputs/eval/eval_report.md` | 训练后质量评估报告。 |
| `outputs/reports/pipeline_report.md` | 流水线摘要报告。 |

## Smoke test

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --smoke_test --device cpu
```

真实 GPU 训练前建议先运行这个命令。它验证链路，不下载默认基础模型。

## 跳过阶段

跳过阶段必须显式声明：

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --skip_cpt
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --skip_cpt --skip_sft
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --skip_cpt --skip_sft --skip_dpo
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --skip_cpt --skip_sft --skip_dpo --skip_grpo
```

只有当上游 adapter 已存在，或某阶段明确不需要时，才使用 skip flags。

## 通过配置启用 DPO

```yaml
dpo:
  enabled: true
  input_path: "data/dpo/preference_examples.jsonl"
```

DPO 输入行需要 `prompt`、`chosen`、`rejected`，并且 `chosen != rejected`。

## 单独运行 GRPO

```bash
python scripts/training/train_grpo.py --config configs/domain_post_training.yaml --max_steps 10
```

GRPO 位于 DPO 或 Fact-SFT 之后。它会为每个 prompt 采样多个 completion，并应用内置奖励和可选外部 reward judge。详见 [GRPO 与 Reward Judge](GRPO-and-Reward-Judge)。

## 合并规则

当 `merge.adapter_dir` 为 `null` 时，合并阶段会按以下优先级自动选择最新可用阶段：

```text
GRPO -> DPO -> Fact-SFT -> CPT
```

只有在需要合并特定 adapter 时才显式设置 `merge.adapter_dir`。

## 静态验证

如果训练依赖不可用：

```bash
python -m compileall pipeline scripts serve_inference.py
```

这个命令只检查语法，不验证 CUDA、数据加载、模型加载或 TRL 行为。

---

## English

Use this page when running, skipping, or debugging training stages.

## Stage Order

```text
CPT -> Fact-SFT -> optional DPO -> optional GRPO -> merge -> quality eval
```

The pipeline produces PEFT adapters first, then merges the selected adapter into a full Hugging Face model.

## Full Pipeline

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml
```

Important outputs:

| Output | When it appears |
|---|---|
| `outputs/lora_adapter/` | CPT adapter. |
| `outputs/fact_sft_adapter/` | Fact-SFT adapter. |
| `outputs/dpo_adapter/` | DPO adapter when `dpo.enabled=true`. |
| `outputs/grpo_adapter/` | GRPO adapter when `grpo.enabled=true`. |
| `outputs/merged_model/` | Merged model. |
| `outputs/cpt_dataset/coverage_report.md` | CPT coverage report. |
| `outputs/eval/eval_report.md` | Post-training quality evaluation report. |
| `outputs/reports/pipeline_report.md` | Pipeline summary report. |

## Smoke Test

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --smoke_test --device cpu
```

Use this before real GPU training. It validates wiring without downloading the default base model.

## Skip Stages

Stage skipping is explicit:

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --skip_cpt
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --skip_cpt --skip_sft
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --skip_cpt --skip_sft --skip_dpo
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --skip_cpt --skip_sft --skip_dpo --skip_grpo
```

Use skip flags only when the required upstream adapter already exists or the stage is intentionally disabled.

## Enable DPO by Configuration

```yaml
dpo:
  enabled: true
  input_path: "data/dpo/preference_examples.jsonl"
```

DPO input rows require `prompt`, `chosen`, and `rejected`, with `chosen != rejected`.

## Run GRPO Directly

```bash
python scripts/training/train_grpo.py --config configs/domain_post_training.yaml --max_steps 10
```

GRPO runs after DPO or Fact-SFT. It samples multiple completions per prompt and applies built-in rewards plus the optional external reward judge. See [GRPO And Reward Judge](GRPO-and-Reward-Judge).

## Merge Behavior

When `merge.adapter_dir` is `null`, adapter merge auto-selects the newest available stage in priority order:

```text
GRPO -> DPO -> Fact-SFT -> CPT
```

Set `merge.adapter_dir` only when you want to merge a specific adapter.

## Static Verification

If training dependencies are unavailable:

```bash
python -m compileall pipeline scripts serve_inference.py
```

This verifies syntax only. It does not validate CUDA, dataset loading, model loading, or TRL behavior.
