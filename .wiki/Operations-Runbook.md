# 操作手册 / Operations Runbook

## 中文

当你要从“能安装”推进到“能稳定训练、定位失败、恢复运行”时，使用本页。

## 训练前环境诊断

安装依赖后先运行：

```bash
python scripts/diagnostics/check_training_environment.py
```

预期输出包含：

- `python`
- `torch`
- `transformers`
- `peft`
- `datasets`
- `trl`
- `cuda_available`
- `torch_cuda`
- `gpu[...]`
- `nvidia-smi`

判断方式：

| 输出 | 含义 | 处理 |
|---|---|---|
| `torch: not installed` 或核心包缺失 | 激活的环境没有安装训练依赖 | 重新激活环境并安装 `requirements.txt`。 |
| `torch import failed` | PyTorch 无法导入，脚本退出码为 `2` | 安装匹配 Python/CUDA 的 PyTorch。 |
| `cuda_available: False` | 当前 PyTorch 看不到 CUDA | 真实训练不要继续；检查驱动、CUDA wheel、虚拟环境和 `nvidia-smi`。CPU 只适合 smoke test。 |
| 没有 `gpu[...]` 行 | 没有可用 GPU 或 CUDA 不可见 | 先修环境，再运行训练。 |
| `nvidia-smi: not found` | 命令不可用或驱动工具不在 PATH | Windows/Linux 均需确认 NVIDIA 驱动和 PATH。 |

## 准备基础模型

默认配置使用：

```yaml
base_model_repo_id: "Qwen/Qwen3.5-0.8B"
base_model_name_or_path: "models/base-model"
```

`base_model_repo_id` 是下载来源，`base_model_name_or_path` 是训练、合并和推理实际加载的位置。

下载配置中的 Hugging Face 模型快照：

```bash
python scripts/model_artifacts/download_models.py --config configs/domain_post_training.local.yaml
```

预期结果：`models/base-model/` 下出现 Hugging Face 模型文件，例如 `config.json`、tokenizer 文件和 safetensors 权重。

如果模型是私有仓库，请先完成 Hugging Face 登录或设置 token，避免把 token 写入配置文件：

```bash
huggingface-cli login
```

如果你已经有本地模型快照，直接让配置指向该目录：

```yaml
base_model_name_or_path: "D:/models/my-base-model"
```

也可以用命令行覆盖下载目标：

```bash
python scripts/model_artifacts/download_models.py --model_id Qwen/Qwen3.5-0.8B --local_dir models/base-model
```

## 真实训练配置预检

使用 Git 忽略的 `configs/domain_post_training.local.yaml` 运行真实训练。默认 DPO、GRPO 和外部 Judge 开启，确认已准备两个数据集并配置 Judge 的 `base_url`、`model` 和明文 `api_key`。环境变量只在 `api_key` 为空时作为可选回退。默认 `max_concurrency: "auto"` 跟随 `num_generations`，`retry_backoff_seconds: 1.0` 是独立请求指数退避的基数。

四个训练阶段保持：

```yaml
bf16: auto
fp16: auto
torch_dtype: auto
abort_on_nonfinite_grad_norm: true
```

自动模式在支持时优先 BF16，否则在 CUDA 上回退 FP16。运行后仍需确认日志和 metadata 中的实际精度与有限梯度。

## 失败报告优先看哪里

完整流水线失败时，先看：

```text
outputs/reports/failure_report.md
outputs/reports/pipeline_report.md
outputs/logs/preflight_report.md
outputs/logs/discovered_corpus.json
```

阶段脚本的常见退出码：

| 退出码 | 来源 | 含义 |
|---|---|---|
| `2` | `check_training_environment.py` | PyTorch 无法导入。 |
| `4` | CPT training | CPT 训练失败。 |
| `5` | adapter merge | 合并 adapter 失败。 |
| `6` | quality evaluation | 评估执行失败或状态不是 completed。 |
| `7` | `train_pipeline.py` | 完整流水线失败，并写入 failure report。 |
| `8` | 完整流水线、Fact-SFT 或 ONNX export | 完整流水线产物完成但质量门禁失败；独立 Fact-SFT/ONNX 脚本仍用 `8` 表示自身失败。 |
| `9` | DPO | DPO 数据准备或训练失败。 |
| `10` | GRPO | GRPO 数据准备或训练失败。 |

## Corpus safety preflight

默认流水线在 CPT 前运行安全预检查。它会扫描高风险内容、来源路径和疑似密钥，并写入：

```text
outputs/logs/preflight_report.md
outputs/logs/preflight_report.json
```

如果报告状态为 blocked，默认训练会停止。处理顺序：

1. 打开 `outputs/logs/preflight_report.md`。
2. 删除或替换私有文档、密钥、过长代码块、内部 URL 和许可证受限语料。
3. 重新运行流水线。
4. 只有在离线、私有、确认可训练的受控环境中，才考虑 `--allow_unsafe_corpus`。

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --allow_unsafe_corpus
```

## Resume 和重试

优先使用阶段级重试，避免重复运行已完成的阶段。

完整流水线可跳过阶段：

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --skip_cpt --skip_sft
```

只续跑 GRPO 时使用：

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --skip_cpt --skip_sft --skip_dpo
```

GRPO 会按显式路径、DPO、Fact-SFT、CPT 顺序选择完整 PEFT adapter。目录根部必须有 `adapter_config.json` 和一种 adapter 权重文件；残缺目录会被记录并跳过。

阶段脚本支持“只准备数据集”和“只训练”：

```bash
python -m pipeline.fact_sft --config configs/domain_post_training.local.yaml --prepare_only
python -m pipeline.fact_sft --config configs/domain_post_training.local.yaml --train_only
python -m pipeline.dpo --config configs/domain_post_training.local.yaml --prepare_only
python scripts/training/train_grpo.py --config configs/domain_post_training.local.yaml --prepare_only
python scripts/training/train_grpo.py --config configs/domain_post_training.local.yaml --train_only
```

从 checkpoint 恢复时，优先设置对应阶段的 `resume_from_checkpoint`：

```yaml
fact_sft:
  resume_from_checkpoint: "outputs/fact_sft_adapter/checkpoint-100"

dpo:
  resume_from_checkpoint: "outputs/dpo_adapter/checkpoint-100"

grpo:
  resume_from_checkpoint: "outputs/grpo_adapter/checkpoint-100"
```

跳过前序阶段是在现有 adapter 上开始新的 GRPO 阶段；`resume_from_checkpoint` 恢复同一次阶段运行的 Trainer 状态，不要混用这两个概念。

当只想合并指定 adapter，不依赖配置中的阶段开关时：

```bash
python scripts/model_artifacts/merge_adapter.py --config configs/domain_post_training.local.yaml --adapter_dir outputs/grpo_adapter
```

## 主要报告怎么看

| 报告 | 用途 |
|---|---|
| `outputs/cpt_dataset/coverage_report.md` | CPT 文档发现、切分和覆盖情况。 |
| `outputs/logs/preflight_report.md` | 训练前语料安全检查。 |
| `outputs/fact_sft_dataset/fact_sft_dataset_report.md` | SFT 样本数量、跳过样本、assistant-only loss 情况。 |
| `outputs/dpo_dataset/dpo_dataset_report.md` | DPO 偏好对数量、跳过原因和分类分布。 |
| `outputs/grpo_dataset/grpo_dataset_report.md` | GRPO prompt 数量、跳过原因、Judge 状态、内置奖励列表和分类分布。 |
| `outputs/reports/grpo_report.md` | GRPO 起点 adapter、精度、loss、Judge 配置，以及实际并发、completion 数和 Judge 批延迟摘要。 |
| `outputs/merged_model/merge_report.json` | 合并使用的 adapter、基础模型、dtype 和加载验证。 |
| `outputs/eval/eval_report.md` | 训练后质量评估输出，重点看 `safety_boundary` 和 `base_regression`。 |
| `outputs/reports/pipeline_report.md` | 完整流水线摘要。 |

## 实跑验收清单

1. 检查 CPT、Fact-SFT、DPO、GRPO 的 `grad_norm`，任何 NaN/Inf 都应视为失败；有限 loss 和 completed 状态不够。
2. 对比相邻 adapter 的 LoRA tensor，确认训练参数实际变化。
3. 检查 Judge reward 均值和方差。reward 过低需要检查数据与 rubric；近似常数无法提供有效排序信号。
4. 检查 `completions/clipped_ratio`。高截断先改善 EOS/停止行为，再考虑提高 `grpo.max_completion_length`。
5. 对照 `configured_max_concurrency`、`effective_concurrency`、`completion_count` 和 `judge_batch_latency_seconds`；确认实际并发不超过 `min(上限, completion 数)`。Semaphore 上限按进程/rank 生效，线程池容量可能让实际连接数更低。
6. 出现 429/503 时检查 `Retry-After`、服务限流和 `retry_backoff_seconds`。任一请求最终失败应使整批 reward 失败，不能使用部分或中性分数继续训练。
7. 推理型 Judge 若返回空 `content`，使用 `reward_judge.max_tokens: 4096` 和 `timeout_seconds: 120`；不要用策略的 completion 长度修复 Judge 输出。
8. `eval_report.md` 只是启发式 smoke gate。生产发布前对安全样例执行人工或独立 Judge 复核。

## 存储和清理

不要在合并、评估、导出前删除：

- 当前要合并的 adapter 目录。
- `outputs/merged_model/`。
- 数据集报告和训练 metadata。

确认无需恢复训练后，通常可以清理：

- 旧 checkpoint。
- 旧 smoke test 产物：`outputs/smoke/`。
- Python 缓存：`__pycache__/`。
- 已废弃的临时导出目录。

---

## English

Use this page when moving from “installed” to “operable”: stable training, failure diagnosis, and recovery.

## Pre-Training Environment Diagnostics

After installing dependencies, run:

```bash
python scripts/diagnostics/check_training_environment.py
```

Expected output includes:

- `python`
- `torch`
- `transformers`
- `peft`
- `datasets`
- `trl`
- `cuda_available`
- `torch_cuda`
- `gpu[...]`
- `nvidia-smi`

How to interpret it:

| Output | Meaning | Action |
|---|---|---|
| `torch: not installed` or missing core packages | The active environment does not have training dependencies | Reactivate the environment and install `requirements.txt`. |
| `torch import failed` | PyTorch cannot be imported; the script exits with code `2` | Install the PyTorch build that matches Python and CUDA. |
| `cuda_available: False` | PyTorch cannot see CUDA | Do not start real training; check driver, CUDA wheel, virtualenv, and `nvidia-smi`. CPU is only for smoke tests. |
| No `gpu[...]` lines | No visible GPU or CUDA is unavailable | Fix the environment before training. |
| `nvidia-smi: not found` | Driver utility is unavailable or not in PATH | Confirm NVIDIA driver installation and PATH. |

## Prepare the Base Model

The default config uses:

```yaml
base_model_repo_id: "Qwen/Qwen3.5-0.8B"
base_model_name_or_path: "models/base-model"
```

`base_model_repo_id` is the download source; `base_model_name_or_path` is what training, merge, and inference actually load.

Download the configured Hugging Face model snapshot:

```bash
python scripts/model_artifacts/download_models.py --config configs/domain_post_training.local.yaml
```

Expected result: `models/base-model/` contains Hugging Face model files such as `config.json`, tokenizer files, and safetensors weights.

For private models, authenticate through Hugging Face before downloading. Do not put tokens in config files:

```bash
huggingface-cli login
```

If you already have a local snapshot, point the config at it:

```yaml
base_model_name_or_path: "D:/models/my-base-model"
```

You can also override the download target:

```bash
python scripts/model_artifacts/download_models.py --model_id Qwen/Qwen3.5-0.8B --local_dir models/base-model
```

## Live Training Config Preflight

Run real training with the Git-ignored `configs/domain_post_training.local.yaml`. DPO, GRPO, and the external judge are enabled by default; confirm that both datasets exist and configure judge `base_url`, `model`, and plaintext `api_key`. Environment variables are only an optional fallback when `api_key` is empty. The default `max_concurrency: "auto"` follows `num_generations`, and `retry_backoff_seconds: 1.0` is the base delay for independent exponential retries.

Keep these settings across all four stages:

```yaml
bf16: auto
fp16: auto
torch_dtype: auto
abort_on_nonfinite_grad_norm: true
```

Auto mode prefers BF16 when supported and otherwise falls back to FP16 on CUDA. Still confirm the effective precision and finite gradients in logs and metadata.

## Where to Look After a Failure

For full-pipeline failures, check:

```text
outputs/reports/failure_report.md
outputs/reports/pipeline_report.md
outputs/logs/preflight_report.md
outputs/logs/discovered_corpus.json
```

Common stage exit codes:

| Exit code | Source | Meaning |
|---|---|---|
| `2` | `check_training_environment.py` | PyTorch import failed. |
| `4` | CPT training | CPT training failed. |
| `5` | adapter merge | Adapter merge failed. |
| `6` | quality evaluation | Evaluation execution failed or did not complete. |
| `7` | `train_pipeline.py` | Full pipeline failed and wrote a failure report. |
| `8` | Full pipeline, Fact-SFT, or ONNX export | Full-pipeline artifacts completed but the quality gate failed; standalone Fact-SFT/ONNX scripts still use `8` for their own failure. |
| `9` | DPO | DPO preparation or training failed. |
| `10` | GRPO | GRPO preparation or training failed. |

## Corpus Safety Preflight

The default pipeline runs a safety preflight before CPT. It scans high-risk content, source paths, and possible secrets, then writes:

```text
outputs/logs/preflight_report.md
outputs/logs/preflight_report.json
```

If the report status is blocked, training stops by default. Handling order:

1. Open `outputs/logs/preflight_report.md`.
2. Remove or replace private documents, secrets, long code blocks, internal URLs, and license-restricted material.
3. Rerun the pipeline.
4. Use `--allow_unsafe_corpus` only in an offline, private, controlled environment where the corpus is approved for training.

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --allow_unsafe_corpus
```

## Resume and Retry

Prefer stage-level retries so completed stages do not run again.

The full pipeline can skip stages:

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --skip_cpt --skip_sft
```

To run only GRPO from existing outputs:

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml --skip_cpt --skip_sft --skip_dpo
```

GRPO selects a complete PEFT adapter in explicit path, DPO, Fact-SFT, CPT order. The root must contain `adapter_config.json` and one adapter weight file; incomplete directories are recorded and skipped.

Stage scripts support preparation-only and training-only modes:

```bash
python -m pipeline.fact_sft --config configs/domain_post_training.local.yaml --prepare_only
python -m pipeline.fact_sft --config configs/domain_post_training.local.yaml --train_only
python -m pipeline.dpo --config configs/domain_post_training.local.yaml --prepare_only
python scripts/training/train_grpo.py --config configs/domain_post_training.local.yaml --prepare_only
python scripts/training/train_grpo.py --config configs/domain_post_training.local.yaml --train_only
```

To resume from a checkpoint, set the matching stage’s `resume_from_checkpoint`:

```yaml
fact_sft:
  resume_from_checkpoint: "outputs/fact_sft_adapter/checkpoint-100"

dpo:
  resume_from_checkpoint: "outputs/dpo_adapter/checkpoint-100"

grpo:
  resume_from_checkpoint: "outputs/grpo_adapter/checkpoint-100"
```

Skipping earlier stages starts a new GRPO stage from an existing adapter. `resume_from_checkpoint` restores Trainer state from the same stage run; do not treat them as the same operation.

To merge a specific adapter without relying on stage switches:

```bash
python scripts/model_artifacts/merge_adapter.py --config configs/domain_post_training.local.yaml --adapter_dir outputs/grpo_adapter
```

## Reading the Main Reports

| Report | Use |
|---|---|
| `outputs/cpt_dataset/coverage_report.md` | CPT document discovery, chunking, and coverage. |
| `outputs/logs/preflight_report.md` | Pre-training corpus safety checks. |
| `outputs/fact_sft_dataset/fact_sft_dataset_report.md` | SFT example counts, skipped examples, and assistant-only loss. |
| `outputs/dpo_dataset/dpo_dataset_report.md` | DPO pair counts, skip reasons, and category distribution. |
| `outputs/grpo_dataset/grpo_dataset_report.md` | GRPO prompt counts, skip reasons, judge state, built-in rewards, and category distribution. |
| `outputs/reports/grpo_report.md` | GRPO starting adapter, precision, loss, judge configuration, and summaries of effective concurrency, completion count, and judge batch latency. |
| `outputs/merged_model/merge_report.json` | Adapter source, base model, dtype, and load test. |
| `outputs/eval/eval_report.md` | Post-training quality evaluation, especially `safety_boundary` and `base_regression`. |
| `outputs/reports/pipeline_report.md` | Full-pipeline summary. |

## Real-Run Acceptance Checklist

1. Inspect `grad_norm` across CPT, Fact-SFT, DPO, and GRPO. Treat any NaN/Inf as failure; finite loss and completed status are insufficient.
2. Compare LoRA tensors across adjacent adapters to verify real parameter changes.
3. Inspect judge reward mean and variance. Low rewards require rubric/data review; near-constant rewards cannot rank candidates effectively.
4. Inspect `completions/clipped_ratio`. Improve EOS/stop behavior before raising `grpo.max_completion_length`.
5. Compare `configured_max_concurrency`, `effective_concurrency`, `completion_count`, and `judge_batch_latency_seconds`; effective concurrency must not exceed `min(cap, completion count)`. The semaphore cap applies per process/rank, and executor capacity may reduce physical connections.
6. On 429/503, inspect `Retry-After`, provider limits, and `retry_backoff_seconds`. Any final request failure should fail the complete reward batch rather than continue with partial or neutral scores.
7. For empty judge `content`, use `reward_judge.max_tokens: 4096` and `timeout_seconds: 120`; do not tune judge output with policy completion length.
8. Treat `eval_report.md` as a heuristic smoke gate. Human-review or independently judge safety cases before production release.

## Storage and Cleanup

Do not delete before merge, evaluation, or export:

- The adapter directory you plan to merge.
- `outputs/merged_model/`.
- Dataset reports and training metadata.

Usually safe after you no longer need the ability to resume training:

- Old checkpoints.
- Old smoke artifacts: `outputs/smoke/`.
- Python caches: `__pycache__/`.
- Obsolete temporary export directories.
