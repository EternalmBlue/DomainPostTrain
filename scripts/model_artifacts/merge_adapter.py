"""中文：运行 adapter merge 入口，把 LoRA/QLoRA adapter 合并回基座模型。
使用时机：CPT、Fact-SFT 或 DPO adapter 训练完成后，需要导出完整 Hugging Face 模型时使用。

English: Adapter merge entrypoint for merging a LoRA/QLoRA adapter back into the base model.
Use it after CPT, Fact-SFT, or DPO adapter training when exporting a full Hugging Face model.
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

from pipeline.adapter_merge import main


if __name__ == "__main__":
    raise SystemExit(main())
