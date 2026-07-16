# 快速开始 / Quick Start

## 中文

当你想从一个新 checkout 快速验证 DomainPostTrain 是否能跑通时，使用本页。

快速自检只验证链路是否可用，不代表已经完成真实模型训练。真实训练仍需要替换数据、配置基础模型、准备 GPU 和检查输出质量。

## 前置条件

- 与所选 PyTorch 构建兼容的 Python 环境。
- 主仓库已经 clone 到本地。
- 如果运行 smoke test 或训练，需要为 `outputs/` 预留磁盘空间。
- 真实训练需要 CUDA GPU。CPU 只适合做链路 smoke test。

## 1. 创建环境

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python --version
py -3.10 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果 `python --version` 没有输出，可能命中了 WindowsApps 空壳；后续使用 `py` 或虚拟环境中的 `\.venv\Scripts\python.exe`。预期结果：环境中包含 PyTorch、Transformers、Datasets、PEFT、TRL、Flask 和默认流水线依赖。

## 2. 复制默认配置

```bash
cp configs/domain_post_training.yaml configs/domain_post_training.local.yaml
```

Windows PowerShell:

```powershell
Copy-Item configs/domain_post_training.yaml configs/domain_post_training.local.yaml
```

编辑 `configs/domain_post_training.local.yaml`。该命名模式应被 Git 忽略，可用于保存明文 `grpo.reward_judge.api_key`。保留受跟踪的 `configs/domain_post_training.yaml` 作为无密钥模板。

默认 DPO 和 GRPO 开启。真实训练前需要准备对应数据，并在 local YAML 中填写 Judge 的 `base_url`、`model`、`api_key`。推理型 Judge 建议使用 `max_tokens: 4096` 和 `timeout_seconds: 120`。默认 `max_concurrency: "auto"` 跟随 `num_generations`；评分在当前 rollout batch 内并发，整批完成后才同步更新权重。

## 3. 运行 CPU smoke test

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --smoke_test --device cpu
```

预期结果：

- 在 `outputs/smoke/base_model` 下创建极小 smoke model。
- 覆盖数据准备、PEFT adapter 保存、adapter merge 和报告生成链路。
- 不下载默认基础模型。

## 4. ML 依赖不可用时的静态检查

```bash
python -m compileall pipeline scripts serve_inference.py
```

预期结果：Python 文件没有语法错误。这个检查不证明 GPU 训练依赖已经安装完成。

## 5. 下一步

下载基础模型并运行完整链路：

```bash
python scripts/model_artifacts/download_models.py --config configs/domain_post_training.local.yaml
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml
```

训练完成不等于达到生产质量。检查所有阶段的有限 `grad_norm`、LoRA tensor 实际变化、Judge reward 方差和 GRPO 截断率，并独立复核安全样例。

- 替换 mock 数据： [数据契约](Data-Contracts)
- 检查训练环境和基础模型： [操作手册](Operations-Runbook)
- 修改路径或训练设置： [配置](Configuration)
- 跑完整训练流程： [训练流水线](Training-Pipeline)
- 排查失败： [故障排查](Troubleshooting)

---

## English

Use this page when you want the shortest path from a fresh checkout to a verified DomainPostTrain setup.

The quick smoke path verifies wiring only. It does not mean a real model has been trained. Real training still requires replacing data, configuring the base model, preparing GPU resources, and checking output quality.

## Prerequisites

- A Python environment compatible with the chosen PyTorch build.
- A local checkout of the main repository.
- Enough disk space for `outputs/` if you run smoke tests or training.
- A CUDA GPU for real training. CPU is only practical for smoke wiring.

## 1. Create an Environment

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python --version
py -3.10 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `python --version` prints nothing, Windows may be resolving the WindowsApps placeholder. Use `py` or `\.venv\Scripts\python.exe` for subsequent commands. Expected result: the environment contains PyTorch, Transformers, Datasets, PEFT, TRL, Flask, and the other default pipeline dependencies.

## 2. Copy the Default Config

```bash
cp configs/domain_post_training.yaml configs/domain_post_training.local.yaml
```

Windows PowerShell:

```powershell
Copy-Item configs/domain_post_training.yaml configs/domain_post_training.local.yaml
```

Edit `configs/domain_post_training.local.yaml`. This name pattern should be ignored by Git and can hold the plaintext `grpo.reward_judge.api_key`. Keep tracked `configs/domain_post_training.yaml` as the key-free template.

DPO and GRPO are enabled by default. Before real training, prepare both datasets and set the judge `base_url`, `model`, and `api_key` in the local YAML. For reasoning judges, start with `max_tokens: 4096` and `timeout_seconds: 120`. The default `max_concurrency: "auto"` follows `num_generations`; scoring is concurrent within the current rollout batch, and weight updates remain synchronized after the full batch completes.

## 3. Run the CPU Smoke Test

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --smoke_test --device cpu
```

Expected result:

- A tiny smoke model is created under `outputs/smoke/base_model`.
- Dataset preparation, PEFT adapter saving, adapter merge, and report generation are exercised.
- The command does not download the default base model.

## 4. Static Check When ML Dependencies Are Unavailable

```bash
python -m compileall pipeline scripts serve_inference.py
```

Expected result: Python files compile without syntax errors. This does not prove that GPU training dependencies are installed.

## 5. Next Pages

Download the base model and run the complete chain:

```bash
python scripts/model_artifacts/download_models.py --config configs/domain_post_training.local.yaml
python scripts/training/train_pipeline.py --config configs/domain_post_training.local.yaml
```

A completed run is not proof of production quality. Check finite `grad_norm` across all stages, real LoRA tensor changes, judge reward variance, and GRPO clipping, and independently review safety cases.

- Replace the mock data: [Data Contracts](Data-Contracts)
- Check the training environment and base model: [Operations Runbook](Operations-Runbook)
- Change paths or training settings: [Configuration](Configuration)
- Run the full training workflow: [Training Pipeline](Training-Pipeline)
- Diagnose failures: [Troubleshooting](Troubleshooting)
