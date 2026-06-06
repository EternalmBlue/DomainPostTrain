"""中文：把 merge 后的 Hugging Face 模型导出并量化为 GGUF。
使用时机：训练和 adapter merge 完成后，需要面向 llama.cpp 或兼容运行时发布本地推理产物时使用。

English: Export and quantize the merged Hugging Face model to GGUF.
Use it after training and adapter merge when publishing a local inference artifact for llama.cpp or compatible runtimes.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPTS_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "_shared" / "bootstrap.py").is_file())
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _shared.bootstrap import bootstrap_project_root


TRAINING_ROOT = bootstrap_project_root(__file__)

from pipeline.utils import load_config, resolve_training_path


def _run(command: list[str], cwd: Path | None = None) -> None:
    print(" ".join(command))
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def _find_llama_cpp_converter(llama_cpp_dir: Path) -> Path:
    for name in ("convert_hf_to_gguf.py", "convert.py"):
        candidate = llama_cpp_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find llama.cpp HF-to-GGUF converter in {llama_cpp_dir}")


def _find_quantize_binary(llama_cpp_dir: Path) -> Path:
    names = ["llama-quantize", "quantize", "llama-quantize.exe", "quantize.exe"]
    for subdir in (llama_cpp_dir, llama_cpp_dir / "build" / "bin", llama_cpp_dir / "build" / "bin" / "Release"):
        for name in names:
            candidate = subdir / name
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Could not find llama.cpp quantize binary in {llama_cpp_dir}")


def _export_with_llama_cpp(args: argparse.Namespace, merged_model_dir: Path, final_path: Path, quant_method: str) -> None:
    if not args.llama_cpp_dir:
        raise ValueError("--llama_cpp_dir is required for --backend llama_cpp")
    llama_cpp_dir = Path(args.llama_cpp_dir).expanduser().resolve()
    converter = _find_llama_cpp_converter(llama_cpp_dir)
    quantize = _find_quantize_binary(llama_cpp_dir)
    temp_f16 = final_path.with_name(f"{final_path.stem}.F16.gguf")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    _run([sys.executable, str(converter), str(merged_model_dir), "--outfile", str(temp_f16), "--outtype", "f16"], cwd=llama_cpp_dir)
    _run([str(quantize), str(temp_f16), str(final_path), quant_method.upper()], cwd=llama_cpp_dir)


def _export_with_unsloth(args: argparse.Namespace, merged_model_dir: Path, final_path: Path, quant_method: str) -> None:
    from unsloth import FastLanguageModel

    final_path.parent.mkdir(parents=True, exist_ok=True)
    export_dir = final_path.parent / "_unsloth_export"
    export_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(merged_model_dir),
        max_seq_length=int(args.max_seq_length),
        load_in_4bit=False,
    )
    model.save_pretrained_gguf(str(export_dir), tokenizer, quantization_method=quant_method)
    candidates = sorted(export_dir.glob("*.gguf"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"Unsloth did not create a GGUF file in {export_dir}")
    shutil.copy2(candidates[0], final_path)
    print(f"Copied {candidates[0]} to {final_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the merged Hugging Face model to GGUF.")
    parser.add_argument("--config", default="configs/domain_post_training.yaml")
    parser.add_argument("--merged_model_dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--quantization_method", default=None, help="Default: config gguf.quantization_method, usually q4_k_m.")
    parser.add_argument("--backend", choices=["llama_cpp", "unsloth"], default="llama_cpp")
    parser.add_argument("--llama_cpp_dir", default=None, help="Path to a built llama.cpp checkout for llama_cpp backend.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Used only by the Unsloth backend.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config, _ = load_config(args.config)
    training_cfg = config.get("training", {})
    gguf_cfg = config.get("gguf", {})
    merged_model_dir = resolve_training_path(args.merged_model_dir or training_cfg.get("merged_output_dir"), "outputs/merged_model")
    output_dir = resolve_training_path(gguf_cfg.get("output_dir"), "models/gguf")
    output_name = gguf_cfg.get("output_name", "DomainPostTrain-Q4_K_M.gguf")
    final_path = Path(args.output).expanduser().resolve() if args.output else output_dir / output_name
    quant_method = str(args.quantization_method or gguf_cfg.get("quantization_method", "iq4_xs"))

    if not merged_model_dir.exists():
        raise FileNotFoundError(f"Merged HF model directory not found: {merged_model_dir}")
    if args.backend == "unsloth":
        _export_with_unsloth(args, merged_model_dir, final_path, quant_method)
    else:
        _export_with_llama_cpp(args, merged_model_dir, final_path, quant_method)
    print(f"GGUF written to {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
