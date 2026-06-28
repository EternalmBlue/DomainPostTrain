# Quick Start

Use this page when you want the shortest path from a fresh checkout to a verified DomainPostTrain setup.

中文提示: 快速自检优先验证链路是否通, 不代表已经完成真实模型训练。

## Prerequisites

- Python environment compatible with the chosen PyTorch build.
- Git checkout of the main repository.
- Enough disk space for `outputs/` if you run smoke tests or training.
- CUDA GPU for real training. CPU is only practical for smoke wiring.

## 1. Create an environment

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Expected result: the environment contains PyTorch, Transformers, Datasets, PEFT, TRL, Flask, and the other default pipeline dependencies.

## 2. Copy the default config

```bash
cp configs/domain_post_training.yaml configs/my_domain.yaml
```

On Windows PowerShell:

```powershell
Copy-Item configs/domain_post_training.yaml configs/my_domain.yaml
```

Edit the copy first. Keep `configs/domain_post_training.yaml` as the baseline example.

## 3. Run the CPU smoke test

```bash
python scripts/training/train_pipeline.py --config configs/domain_post_training.yaml --smoke_test --device cpu
```

Expected result:

- A tiny smoke model is created under `outputs/smoke/base_model`.
- Dataset preparation, PEFT adapter saving, adapter merge, and report generation are exercised.
- The command does not download the default base model.

## 4. If ML dependencies are unavailable

Run a static check:

```bash
python -m compileall pipeline scripts serve_inference.py
```

Expected result: Python files compile without syntax errors. This does not prove that GPU training dependencies are installed.

## 5. Next pages

- Replace the mock data: [Data Contracts](Data-Contracts)
- Change paths or training settings: [Configuration](Configuration)
- Run the full training workflow: [Training Pipeline](Training-Pipeline)
- Diagnose failures: [Troubleshooting](Troubleshooting)

