"""中文：单独运行 DPO 阶段的训练入口。
使用时机：Fact-SFT adapter 已准备好，只想用 `data/dpo` 偏好数据继续做 DPO 对齐时使用。

English: Standalone training entrypoint for the DPO stage.
Use it when the Fact-SFT adapter is ready and you only want to continue alignment with `data/dpo` preference data.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


SCRIPTS_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "_shared" / "bootstrap.py").is_file())
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _shared.bootstrap import bootstrap_project_root


TRAINING_ROOT = bootstrap_project_root(__file__)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pipeline.cuda_bootstrap import ensure_pip_cuda_libraries_preferred

ensure_pip_cuda_libraries_preferred()

from pipeline.dpo import main


if __name__ == "__main__":
    raise SystemExit(main())
