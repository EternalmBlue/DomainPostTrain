# Installation

Use this page when setting up a local or training-machine environment.

## Default dependency target

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

## Optional ONNX dependencies

ONNX export is not part of the default path. Install it only when running `scripts/model_artifacts/export_onnx.py`:

```bash
python -m pip install -r requirements-onnx.txt
```

ONNX GPU export can require ONNX Runtime CUDA runtime wheels that match the target machine.

## Offline or private wheelhouse setup

Recommended order:

1. Install the matching `torch`, `torchvision`, and `torchaudio` wheels for the machine.
2. Install the remaining packages from `requirements.txt`, excluding torch packages if necessary.
3. Run `python -m compileall pipeline scripts serve_inference.py`.
4. Run the smoke test only after the ML dependencies are available.

## Common install checks

```bash
python -c "import torch; print(torch.__version__)"
python -c "import transformers, datasets, peft, trl; print('training deps ok')"
python -c "import flask; print('service deps ok')"
```

If any import fails, install the missing package in the active environment. See [Troubleshooting](Troubleshooting) for common dependency failures.

