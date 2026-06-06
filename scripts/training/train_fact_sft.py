"""中文：单独运行 Fact-SFT 阶段的训练入口。
使用时机：CPT adapter 已准备好，只想用 `data/sft` 的事实问答样例做 assistant-only loss 微调时使用。

English: Standalone training entrypoint for the Fact-SFT stage.
Use it when the CPT adapter is ready and you only want assistant-only-loss fine-tuning on `data/sft` examples.
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

from pipeline.fact_sft import main


if __name__ == "__main__":
    raise SystemExit(main())
