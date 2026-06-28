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
base_model_repo_id: "Qwen/Qwen3.5-4B"
base_model_name_or_path: "models/base-model"
```

下载配置中的 Hugging Face 模型快照：

```bash
python scripts/model_artifacts/download_models.py --config configs/domain_post_training.yaml
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
python scripts/model_artifacts/download_models.py --model_id Qwen/Qwen3.5-4B --local_dir models/base-model
```

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
| `6` | quality evaluation | 质量评估失败或状态不是 completed。 |
| `7` | `train_pipeline.py` | 完整流水线失败，并写入 failure report。 |
| `8` | Fact-SFT 或 ONNX export | Fact-SFT 失败；ONNX export 也使用 `8` 表示导出失败。 |
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
python scripts/training/train_pipeline.py --config configs/my_domain.yaml --allow_unsafe_corpus
```

## Resume 和重试

优先使用阶段级重试，避免重复运行已完成的阶段。

完整流水线可跳过阶段：

```bash
python scripts/training/train_pipeline.py --config configs/my_domain.yaml --skip_cpt --skip_sft
```

阶段脚本支持“只准备数据集”和“只训练”：

```bash
python -m pipeline.fact_sft --config configs/my_domain.yaml --prepare_only
python -m pipeline.fact_sft --config configs/my_domain.yaml --train_only
python -m pipeline.dpo --config configs/my_domain.yaml --prepare_only
python scripts/training/train_grpo.py --config configs/my_domain.yaml --prepare_only
python scripts/training/train_grpo.py --config configs/my_domain.yaml --train_only
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

当只想合并指定 adapter，不依赖配置中的阶段开关时：

```bash
python scripts/model_artifacts/merge_adapter.py --config configs/my_domain.yaml --adapter_dir outputs/grpo_adapter
```

## 主要报告怎么看

| 报告 | 用途 |
|---|---|
| `outputs/cpt_dataset/coverage_report.md` | CPT 文档发现、切分和覆盖情况。 |
| `outputs/logs/preflight_report.md` | 训练前语料安全检查。 |
| `outputs/fact_sft_dataset/fact_sft_dataset_report.md` | SFT 样本数量、跳过样本、assistant-only loss 情况。 |
| `outputs/dpo_dataset/dpo_dataset_report.md` | DPO 偏好对数量、跳过原因和分类分布。 |
| `outputs/grpo_dataset/grpo_dataset_report.md` | GRPO prompt 数量、跳过原因、内置奖励列表和分类分布。 |
| `outputs/merged_model/merge_report.json` | 合并使用的 adapter、基础模型、dtype 和加载验证。 |
| `outputs/eval/eval_report.md` | 训练后质量评估输出，重点看 `safety_boundary` 和 `base_regression`。 |
| `outputs/reports/pipeline_report.md` | 完整流水线摘要。 |

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
base_model_repo_id: "Qwen/Qwen3.5-4B"
base_model_name_or_path: "models/base-model"
```

Download the configured Hugging Face model snapshot:

```bash
python scripts/model_artifacts/download_models.py --config configs/domain_post_training.yaml
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
python scripts/model_artifacts/download_models.py --model_id Qwen/Qwen3.5-4B --local_dir models/base-model
```

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
| `6` | quality evaluation | Quality evaluation failed or did not complete. |
| `7` | `train_pipeline.py` | Full pipeline failed and wrote a failure report. |
| `8` | Fact-SFT or ONNX export | Fact-SFT failed; ONNX export also uses `8` for export failure. |
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
python scripts/training/train_pipeline.py --config configs/my_domain.yaml --allow_unsafe_corpus
```

## Resume and Retry

Prefer stage-level retries so completed stages do not run again.

The full pipeline can skip stages:

```bash
python scripts/training/train_pipeline.py --config configs/my_domain.yaml --skip_cpt --skip_sft
```

Stage scripts support preparation-only and training-only modes:

```bash
python -m pipeline.fact_sft --config configs/my_domain.yaml --prepare_only
python -m pipeline.fact_sft --config configs/my_domain.yaml --train_only
python -m pipeline.dpo --config configs/my_domain.yaml --prepare_only
python scripts/training/train_grpo.py --config configs/my_domain.yaml --prepare_only
python scripts/training/train_grpo.py --config configs/my_domain.yaml --train_only
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

To merge a specific adapter without relying on stage switches:

```bash
python scripts/model_artifacts/merge_adapter.py --config configs/my_domain.yaml --adapter_dir outputs/grpo_adapter
```

## Reading the Main Reports

| Report | Use |
|---|---|
| `outputs/cpt_dataset/coverage_report.md` | CPT document discovery, chunking, and coverage. |
| `outputs/logs/preflight_report.md` | Pre-training corpus safety checks. |
| `outputs/fact_sft_dataset/fact_sft_dataset_report.md` | SFT example counts, skipped examples, and assistant-only loss. |
| `outputs/dpo_dataset/dpo_dataset_report.md` | DPO pair counts, skip reasons, and category distribution. |
| `outputs/grpo_dataset/grpo_dataset_report.md` | GRPO prompt counts, skip reasons, built-in rewards, and category distribution. |
| `outputs/merged_model/merge_report.json` | Adapter source, base model, dtype, and load test. |
| `outputs/eval/eval_report.md` | Post-training quality evaluation, especially `safety_boundary` and `base_regression`. |
| `outputs/reports/pipeline_report.md` | Full-pipeline summary. |

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
