"""English: Standalone training entrypoint for the GRPO stage.
Use it when a base alignment adapter is ready and you want reward-optimized GRPO fine-tuning from `data/grpo` prompts.
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

from pipeline.grpo import main


if __name__ == "__main__":
    raise SystemExit(main())
