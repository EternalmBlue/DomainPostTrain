"""中文：检查本机或远程训练环境的 Python、核心依赖、CUDA 和 GPU 状态。
使用时机：训练前排查依赖安装、CUDA 可用性、GPU 型号和显存是否符合预期。

English: Inspect the local or remote training environment, including Python, core packages, CUDA, and GPU status.
Use it before training to verify dependency installation, CUDA availability, GPU model, and memory.
"""

from __future__ import annotations

import importlib.metadata as metadata
import subprocess
import sys
from pathlib import Path


SCRIPTS_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "_shared" / "bootstrap.py").is_file())
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _shared.bootstrap import bootstrap_project_root


TRAINING_ROOT = bootstrap_project_root(__file__)

from pipeline.cuda_bootstrap import ensure_pip_cuda_libraries_preferred

ensure_pip_cuda_libraries_preferred()


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def main() -> int:
    print(f"python: {sys.version.split()[0]}")
    for package in ("torch", "transformers", "peft", "accelerate", "datasets", "trl", "huggingface_hub"):
        print(f"{package}: {_package_version(package)}")

    try:
        import torch
    except ImportError:
        print("torch import failed")
        return 2

    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"torch_cuda: {getattr(torch.version, 'cuda', None)}")
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            total_gb = props.total_memory / (1024**3)
            print(f"gpu[{index}]: {props.name}, {total_gb:.2f} GiB, bf16={torch.cuda.is_bf16_supported()}")

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=False,
            text=True,
            capture_output=True,
        )
        if result.stdout.strip():
            print("nvidia-smi:")
            print(result.stdout.strip())
    except FileNotFoundError:
        print("nvidia-smi: not found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
