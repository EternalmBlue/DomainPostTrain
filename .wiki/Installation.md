# 安装 / Installation

## 中文

当你要在本地机器或训练机器上准备 DomainPostTrain 环境时，使用本页。

## 默认依赖目标

`requirements.txt` 默认面向 CUDA 12.6 GPU 训练，并包含 PyTorch CUDA wheel index：

```text
--extra-index-url https://download.pytorch.org/whl/cu126
```

如果你的 CUDA runtime、Python ABI、驱动栈或操作系统不同，请先安装匹配的 PyTorch，再安装其余非 torch 依赖。

## Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

验证：

```bash
python -m compileall pipeline scripts serve_inference.py
```

## Windows PowerShell

```powershell
py -3.10 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

验证：

```powershell
& .\.venv\Scripts\python.exe -m compileall pipeline scripts serve_inference.py
```

## 可选 ONNX 依赖

ONNX 导出不属于默认训练路径。只有运行 `scripts/model_artifacts/export_onnx.py` 时才需要安装：

```bash
python -m pip install -r requirements-onnx.txt
```

ONNX GPU 导出可能需要与目标机器匹配的 ONNX Runtime CUDA wheel。

## 离线或私有 wheelhouse

推荐顺序：

1. 先安装与机器匹配的 `torch`、`torchvision`、`torchaudio` wheels。
2. 再从 `requirements.txt` 安装其余包，必要时排除 torch 包。
3. 运行 `python -m compileall pipeline scripts serve_inference.py`。
4. 确认 ML 依赖可用后再运行 smoke test。

## 常用安装检查

```bash
python -c "import torch; print(torch.__version__)"
python -c "import transformers, datasets, peft, trl; print('training deps ok')"
python -c "import flask; print('service deps ok')"
```

如果任一 import 失败，请在当前激活环境中安装缺失包。常见问题见 [故障排查](Troubleshooting)。

---

## English

Use this page when setting up a local or training-machine environment for DomainPostTrain.

## Default Dependency Target

`requirements.txt` targets CUDA 12.6 GPU training and includes a PyTorch CUDA wheel index:

```text
--extra-index-url https://download.pytorch.org/whl/cu126
```

If your CUDA runtime, Python ABI, driver stack, or operating system differs, install a matching PyTorch build first, then install the remaining non-torch dependencies.

## Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify:

```bash
python -m compileall pipeline scripts serve_inference.py
```

## Windows PowerShell

```powershell
py -3.10 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Verify:

```powershell
& .\.venv\Scripts\python.exe -m compileall pipeline scripts serve_inference.py
```

## Optional ONNX Dependencies

ONNX export is not part of the default path. Install it only when running `scripts/model_artifacts/export_onnx.py`:

```bash
python -m pip install -r requirements-onnx.txt
```

ONNX GPU export can require ONNX Runtime CUDA runtime wheels that match the target machine.

## Offline or Private Wheelhouse Setup

Recommended order:

1. Install the matching `torch`, `torchvision`, and `torchaudio` wheels for the machine.
2. Install the remaining packages from `requirements.txt`, excluding torch packages if necessary.
3. Run `python -m compileall pipeline scripts serve_inference.py`.
4. Run the smoke test only after the ML dependencies are available.

## Common Install Checks

```bash
python -c "import torch; print(torch.__version__)"
python -c "import transformers, datasets, peft, trl; print('training deps ok')"
python -c "import flask; print('service deps ok')"
```

If any import fails, install the missing package in the active environment. See [Troubleshooting](Troubleshooting) for common dependency failures.
