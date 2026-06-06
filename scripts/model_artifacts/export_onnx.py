"""中文：运行 ONNX 导出入口，把已 merge 的 Hugging Face 模型导出为 ONNX。
使用时机：需要在 ONNX Runtime 或兼容推理栈中测试部署产物时使用。

English: ONNX export entrypoint for converting a merged Hugging Face model to ONNX.
Use it when testing deployment artifacts with ONNX Runtime or compatible inference stacks.
"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "_shared" / "bootstrap.py").is_file())
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _shared.bootstrap import bootstrap_project_root


TRAINING_ROOT = bootstrap_project_root(__file__)

from pipeline.cuda_bootstrap import ensure_pip_cuda_libraries_preferred

ensure_pip_cuda_libraries_preferred()

from pipeline.onnx_export import main


if __name__ == "__main__":
    raise SystemExit(main())
