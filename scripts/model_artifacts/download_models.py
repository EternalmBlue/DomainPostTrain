"""中文：下载配置中的 Hugging Face/safetensors 基座模型到本地目录。
使用时机：训练前需要准备 `base_model_name_or_path` 指向的本地模型快照时使用。

English: Download the configured Hugging Face/safetensors base model to a local directory.
Use it before training when you need a local model snapshot for `base_model_name_or_path`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "_shared" / "bootstrap.py").is_file())
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _shared.bootstrap import bootstrap_project_root


TRAINING_ROOT = bootstrap_project_root(__file__)

from huggingface_hub import snapshot_download

from pipeline.utils import is_local_model_path, load_config, resolve_training_path


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-4B"
DEFAULT_LOCAL_DIR = "models/base-model"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download the configured Hugging Face base model.")
    parser.add_argument("--config", default="configs/domain_post_training.yaml")
    parser.add_argument("--model_id", default=None, help="HF model id for trainable safetensors weights.")
    parser.add_argument(
        "--local_dir",
        default=None,
        help="Local directory for the HF model snapshot. Defaults to config base_model_name_or_path when it is local.",
    )
    parser.add_argument("--skip_hf", action="store_true", help="Do not download the trainable HF model snapshot.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config, _ = load_config(args.config)
    configured_model_path = config.get("base_model_name_or_path", DEFAULT_MODEL_ID)
    model_id = args.model_id or config.get("base_model_repo_id")
    if not model_id:
        model_id = DEFAULT_MODEL_ID if is_local_model_path(configured_model_path) else configured_model_path

    local_dir_raw = args.local_dir
    if local_dir_raw is None:
        local_dir_raw = configured_model_path if is_local_model_path(configured_model_path) else DEFAULT_LOCAL_DIR
    local_dir = resolve_training_path(local_dir_raw, DEFAULT_LOCAL_DIR)

    if not args.skip_hf:
        print(f"Downloading trainable HF model {model_id} to {local_dir}")
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            ignore_patterns=["*.gguf"],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
