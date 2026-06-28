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

更多报告解释见 [操作手册](Operations-Runbook)。

## 冒烟测试

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --smoke_test --device cpu
```

真实 GPU 训练前建议先运行这个命令。它验证配置加载、语料发现、数据集准备、PEFT adapter 保存、merge 和报告链路，不下载默认基础模型。

## 跳过阶段

跳过阶段必须显式声明：

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --skip_cpt
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --skip_cpt --skip_sft
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --skip_cpt --skip_sft --skip_dpo
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --skip_cpt --skip_sft --skip_dpo --skip_grpo
```

只有当上游 adapter 已存在，或某阶段明确不需要时，才使用跳过参数。

常用运行控制参数：

| 参数 | 用途 |
|---|---|
| `--skip_preflight` | 跳过 CPT 前的语料安全预检查。 |
| `--allow_unsafe_corpus` | 允许在 preflight blocked 时继续训练，见 [操作手册](Operations-Runbook)。 |
| `--skip_merge` | 训练完成后不合并 adapter。 |
| `--skip_eval` | 合并后不运行质量评估。 |
| `--device cpu|cuda|cuda:0|auto` | 覆盖配置中的训练设备。 |

## 通过配置启用 DPO

```yaml
dpo:
  enabled: true
  input_path: "data/dpo/preference_examples.jsonl"
```

DPO 输入行需要 `prompt`、`chosen`、`rejected`，并且 `chosen != rejected`。DPO 默认从 `outputs/fact_sft_adapter` 继续训练。

## 单独运行 GRPO

```bash
python scripts/training/train_grpo.py --config configs/domain_post_training.yaml --max_steps 10
```

这个命令会把 `grpo.enabled` 置为 true，默认先准备 GRPO 数据集，再运行训练。GRPO 默认从 `outputs/dpo_adapter` 继续训练；如果 DPO 未启用，可以把 `grpo.base_adapter_dir` 或命令行 `--base_adapter_dir` 指向 `outputs/fact_sft_adapter`。

常用 GRPO 调试参数：

| 参数 | 用途 |
|---|---|
| `--prepare_only` | 只准备 `outputs/grpo_dataset`，不训练。 |
| `--train_only` | 使用已准备的数据集直接训练。 |
| `--num_generations` | 覆盖每个 prompt 的采样数量。 |
| `--max_completion_length` | 覆盖 completion 最大长度。 |
| `--base_adapter_dir` | 指定 GRPO 起点 adapter。 |

GRPO 会为每个 prompt 采样多个 completion，并应用内置奖励和可选外部 reward judge。详见 [GRPO 与 Reward Judge](GRPO-and-Reward-Judge)。

## 合并规则

当 `merge.adapter_dir` 为 `null` 且没有从命令行传入 `--adapter_dir` 时，合并阶段按启用阶段选择 adapter：

```text
enabled GRPO -> enabled DPO -> enabled Fact-SFT -> CPT
```

也就是说，如果 `grpo.enabled=false`，即使 `outputs/grpo_adapter` 存在，也不会自动被选中。需要合并特定 adapter 时，显式指定：

```bash
python scripts/model_artifacts/merge_adapter.py --config configs/domain_post_training.yaml --adapter_dir outputs/grpo_adapter
```

或配置：

```yaml
merge:
  adapter_dir: "outputs/grpo_adapter"
```

## 静态验证

如果训练依赖不可用：

```bash
python -m compileall pipeline scripts serve_inference.py
```

这个命令只检查语法，不验证 CUDA、数据加载、模型加载或 TRL 行为。训练前环境诊断见 [操作手册](Operations-Runbook)。

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

See [Operations Runbook](Operations-Runbook) for report interpretation.

## Smoke Test

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --smoke_test --device cpu
```

Use this before real GPU training. It validates config loading, corpus discovery, dataset preparation, PEFT adapter saving, merge, and report generation without downloading the default base model.

## Skip Stages

Stage skipping is explicit:

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --skip_cpt
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --skip_cpt --skip_sft
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --skip_cpt --skip_sft --skip_dpo
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --skip_cpt --skip_sft --skip_dpo --skip_grpo
```

Use skip flags only when the required upstream adapter already exists or the stage is intentionally disabled.

Common run-control flags:

| Flag | Purpose |
|---|---|
| `--skip_preflight` | Skip corpus safety preflight before CPT. |
| `--allow_unsafe_corpus` | Continue when preflight is blocked; see [Operations Runbook](Operations-Runbook). |
| `--skip_merge` | Do not merge after training. |
| `--skip_eval` | Do not run quality evaluation after merge. |
| `--device cpu|cuda|cuda:0|auto` | Override the configured training device. |

## Enable DPO by Configuration

```yaml
dpo:
  enabled: true
  input_path: "data/dpo/preference_examples.jsonl"
```

DPO input rows require `prompt`, `chosen`, and `rejected`, with `chosen != rejected`. DPO defaults to continuing from `outputs/fact_sft_adapter`.

## Run GRPO Directly

```bash
python scripts/training/train_grpo.py --config configs/domain_post_training.yaml --max_steps 10
```

This command sets `grpo.enabled` to true and, by default, prepares the GRPO dataset before training. GRPO defaults to continuing from `outputs/dpo_adapter`; if DPO is disabled, point `grpo.base_adapter_dir` or `--base_adapter_dir` at `outputs/fact_sft_adapter`.

Common GRPO debugging flags:

| Flag | Purpose |
|---|---|
| `--prepare_only` | Prepare `outputs/grpo_dataset` without training. |
| `--train_only` | Train from an already prepared dataset. |
| `--num_generations` | Override completions sampled per prompt. |
| `--max_completion_length` | Override maximum completion length. |
| `--base_adapter_dir` | Select the starting adapter for GRPO. |

GRPO samples multiple completions per prompt and applies built-in rewards plus the optional external reward judge. See [GRPO And Reward Judge](GRPO-and-Reward-Judge).

## Merge Behavior

When `merge.adapter_dir` is `null` and no `--adapter_dir` argument is provided, merge chooses among enabled stages:

```text
enabled GRPO -> enabled DPO -> enabled Fact-SFT -> CPT
```

If `grpo.enabled=false`, an existing `outputs/grpo_adapter` is not selected automatically. To merge a specific adapter, pass it explicitly:

```bash
python scripts/model_artifacts/merge_adapter.py --config configs/domain_post_training.yaml --adapter_dir outputs/grpo_adapter
```

Or configure:

```yaml
merge:
  adapter_dir: "outputs/grpo_adapter"
```

## Static Verification

If training dependencies are unavailable:

```bash
python -m compileall pipeline scripts serve_inference.py
```

This verifies syntax only. It does not validate CUDA, dataset loading, model loading, or TRL behavior. See [Operations Runbook](Operations-Runbook) for environment diagnostics.
