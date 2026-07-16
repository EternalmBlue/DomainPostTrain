# 训练流水线 / Training Pipeline

## 中文

当你要运行、跳过或调试训练阶段时，使用本页。

## 阶段顺序

```text
CPT -> Fact-SFT -> DPO -> GRPO -> merge -> heuristic quality eval
```

流水线先产出 PEFT adapter，再把选定 adapter 合并成完整 Hugging Face 模型。

每个 adapter 同时保存自包含的 `adapter_provenance.json`。merge 报告从该 manifest 准确列出 CPT、Fact-SFT、DPO、GRPO；旧 adapter 使用递归 legacy metadata 回溯。质量门禁失败时训练和合并产物仍保留，但完整流水线返回 `8`，报告显示 `quality_gate_failed` 和 `release_ready: false`。

## 完整流水线

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml
```

默认配置开启 DPO 和 GRPO。真实运行使用 Git 忽略的 local YAML 保存外部 Judge 配置；不需要某个阶段时显式关闭或跳过。

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
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --skip_cpt
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --skip_cpt --skip_sft
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --skip_cpt --skip_sft --skip_dpo
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --skip_cpt --skip_sft --skip_dpo --skip_grpo
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
python scripts/training/train_grpo.py --config configs/domain_post_training.local.yaml --max_steps 10
```

这个命令会把 `grpo.enabled` 置为 true，默认先准备 GRPO 数据集，再运行训练。外部 Judge 默认启用，四个内置奖励默认关闭；Judge 模式允许 prompt-only 数据。

GRPO 起点按“显式 `grpo.base_adapter_dir` -> DPO -> Fact-SFT -> CPT”解析。候选目录根部必须同时包含 `adapter_config.json`，以及 `adapter_model.safetensors` 或 `adapter_model.bin`；残缺目录会被跳过并继续回退。

常用 GRPO 调试参数：

| 参数 | 用途 |
|---|---|
| `--prepare_only` | 只准备 `outputs/grpo_dataset`，不训练。 |
| `--train_only` | 使用已准备的数据集直接训练。 |
| `--num_generations` | 覆盖每个 prompt 的采样数量。 |
| `--max_completion_length` | 覆盖策略候选回答的最大长度；它不控制 Judge 输出预算。 |
| `--base_adapter_dir` | 指定 GRPO 起点 adapter。 |

GRPO 会为每个 prompt 采样多个 completion，并应用默认外部 Judge 及显式启用的内置奖励。推理型 Judge 的输出预算由 `grpo.reward_judge.max_tokens` 控制，建议为 `4096`，超时建议为 `120` 秒。详见 [GRPO 与 Reward Judge](GRPO-and-Reward-Judge)。

外部评分按当前 rollout batch 异步并发：候选生成后，以 `min(max_concurrency 上限, completion 数量)` 发出请求，等待整批结束后才计算 group advantage 并同步执行反向传播和 optimizer 更新。`max_concurrency: "auto"` 跟随 `num_generations`；显式整数可跨同一 reward batch 的多个 prompt 使用。它是每个训练进程/rank 的 Semaphore 上限，线程池容量可能进一步降低物理并发。不会预生成整个数据集，也不会并发更新模型。每个请求独立退避重试，429/503 遵守合法 `Retry-After`；任一最终失败则整批 reward 失败。

从已存在的 DPO/Fact-SFT/CPT adapter 继续流水线：

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --skip_cpt --skip_sft --skip_dpo
```

该命令选择前一阶段 adapter 作为新的 GRPO 起点。`grpo.resume_from_checkpoint` 则恢复同一次 GRPO Trainer 运行的 checkpoint，二者用途不同。

## 实跑验收

- 四阶段使用 `bf16: auto`、`fp16: auto`、`torch_dtype: auto`，支持 BF16 时优先使用 BF16。
- 保持 `abort_on_nonfinite_grad_norm: true`，任何非有限梯度都应立即失败。
- 不要只看有限 loss 或 completed 状态；对比相邻 adapter 的 LoRA tensor，确认参数实际更新。
- 检查 Judge reward 均值/方差和 `completions/clipped_ratio`。高截断率优先通过 EOS/停止行为处理，再考虑增加 `max_completion_length`。
- 质量评估只是启发式 smoke gate，不是生产安全认证。

## 合并规则

当 `merge.adapter_dir` 为 `null` 且没有从命令行传入 `--adapter_dir` 时，合并阶段按启用阶段选择 adapter：

```text
enabled GRPO -> enabled DPO -> enabled Fact-SFT -> CPT
```

也就是说，如果 `grpo.enabled=false`，即使 `outputs/grpo_adapter` 存在，也不会自动被选中。需要合并特定 adapter 时，显式指定：

```bash
python scripts/model_artifacts/merge_adapter.py --config configs/domain_post_training.local.yaml --adapter_dir outputs/grpo_adapter
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
CPT -> Fact-SFT -> DPO -> GRPO -> merge -> heuristic quality eval
```

The pipeline produces PEFT adapters first, then merges the selected adapter into a full Hugging Face model.

Each adapter also stores a self-contained `adapter_provenance.json`. Merge reports derive the CPT, Fact-SFT, DPO, and GRPO stage chain from it, with recursive legacy-metadata fallback for older adapters. A failed quality gate retains training and merge artifacts but makes the full pipeline exit `8`, with `quality_gate_failed` and `release_ready: false` in the report.

## Full Pipeline

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml
```

DPO and GRPO are enabled by default. Use a Git-ignored local YAML for the live external-judge settings; explicitly disable or skip a stage you do not need.

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
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --skip_cpt
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --skip_cpt --skip_sft
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --skip_cpt --skip_sft --skip_dpo
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --skip_cpt --skip_sft --skip_dpo --skip_grpo
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
python scripts/training/train_grpo.py --config configs/domain_post_training.local.yaml --max_steps 10
```

This command sets `grpo.enabled` to true and prepares the GRPO dataset before training by default. The external judge is enabled by default while all four built-in rewards are opt-in; judge mode accepts prompt-only rows.

GRPO resolves its starting point as explicit `grpo.base_adapter_dir` -> DPO -> Fact-SFT -> CPT. A candidate root must contain `adapter_config.json` plus either `adapter_model.safetensors` or `adapter_model.bin`; incomplete directories are skipped while fallback continues.

Common GRPO debugging flags:

| Flag | Purpose |
|---|---|
| `--prepare_only` | Prepare `outputs/grpo_dataset` without training. |
| `--train_only` | Train from an already prepared dataset. |
| `--num_generations` | Override completions sampled per prompt. |
| `--max_completion_length` | Override policy candidate length; it does not control the judge response budget. |
| `--base_adapter_dir` | Select the starting adapter for GRPO. |

GRPO samples multiple completions per prompt and applies the default external judge plus explicitly enabled built-in rewards. Reasoning-judge output is controlled by `grpo.reward_judge.max_tokens`; use `4096` tokens and a `120` second timeout as a starting point. See [GRPO And Reward Judge](GRPO-and-Reward-Judge).

External scoring is asynchronous within the current rollout batch: after candidate generation, requests run with `min(max_concurrency cap, completion count)`, and the trainer waits for the full batch before computing group advantage and performing synchronized backpropagation and optimizer update. `max_concurrency: "auto"` follows `num_generations`; an explicit integer can be shared across prompts in the same reward batch. It is a per-process/rank semaphore cap, and executor capacity may reduce physical concurrency further. The pipeline does not pre-generate the full dataset or update the model concurrently. Requests retry independently with backoff, valid `Retry-After` on 429/503 is honored, and any final failure fails the complete reward batch.

Continue the pipeline from existing DPO/Fact-SFT/CPT adapters:

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --skip_cpt --skip_sft --skip_dpo
```

This selects a previous-stage adapter as a new GRPO starting point. `grpo.resume_from_checkpoint` instead restores a checkpoint from the same GRPO Trainer run.

## Real-Run Acceptance

- Use `bf16: auto`, `fp16: auto`, and `torch_dtype: auto` across all four stages; BF16 is preferred when supported.
- Keep `abort_on_nonfinite_grad_norm: true` so non-finite gradients fail immediately.
- Do not rely on finite loss or completed status; compare LoRA tensors across adjacent adapters to confirm real updates.
- Inspect judge reward mean/variance and `completions/clipped_ratio`. For high clipping, improve EOS/stop behavior before raising `max_completion_length`.
- Quality evaluation is a heuristic smoke gate, not production safety certification.

## Merge Behavior

When `merge.adapter_dir` is `null` and no `--adapter_dir` argument is provided, merge chooses among enabled stages:

```text
enabled GRPO -> enabled DPO -> enabled Fact-SFT -> CPT
```

If `grpo.enabled=false`, an existing `outputs/grpo_adapter` is not selected automatically. To merge a specific adapter, pass it explicitly:

```bash
python scripts/model_artifacts/merge_adapter.py --config configs/domain_post_training.local.yaml --adapter_dir outputs/grpo_adapter
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
