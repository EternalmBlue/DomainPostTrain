from __future__ import annotations

import argparse
import importlib.util
import inspect
import traceback
from pathlib import Path
from typing import Any

import torch

from pipeline.modeling import configure_generation_tokens, disable_cache, load_tokenizer, load_transformers_model
from pipeline.utils import (
    load_config,
    resolve_training_path,
    setup_logging,
    short_error,
    torch_dtype_from_config,
    utc_now,
    write_json,
    write_text,
)


class CausalLMOnnxWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = input_ids.shape
        dtype = self.model.get_input_embeddings().weight.dtype
        positions = torch.arange(seq_length, device=input_ids.device)
        causal = positions.view(1, 1, seq_length, 1) >= positions.view(1, 1, 1, seq_length)
        padding = attention_mask.to(torch.bool).view(batch_size, 1, 1, seq_length)
        allowed = causal & padding
        additive_mask = torch.zeros((batch_size, 1, seq_length, seq_length), dtype=dtype, device=input_ids.device)
        additive_mask = additive_mask.masked_fill(~allowed, torch.finfo(dtype).min)
        outputs = self.model(input_ids=input_ids, attention_mask=additive_mask, use_cache=False)
        return outputs.logits


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for ONNX export, but torch.cuda.is_available() is false.")
    return device


def _validate_onnx_file(onnx_path: Path, logger) -> bool:
    try:
        import onnx
    except ImportError:
        logger.warning("onnx is not installed; skipping ONNX checker validation.")
        return False

    model = onnx.load(str(onnx_path), load_external_data=True)
    onnx.checker.check_model(model)
    return True


def _require_onnx_package() -> None:
    if importlib.util.find_spec("onnx") is None:
        raise RuntimeError(
            "Missing optional Python dependency 'onnx'. Install ONNX export extras with: "
            "python -m pip install -r requirements-onnx.txt"
        )


def _run_ort_shape_check(
    onnx_path: Path,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logger,
) -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning("onnxruntime is not installed; skipping ONNX Runtime shape check.")
        return {"ran": False, "reason": "onnxruntime not installed"}

    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception as exc:
            logger.warning("onnxruntime.preload_dlls() failed; continuing with provider selection: %s", exc)

    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available = set(ort.get_available_providers())
    selected = [provider for provider in preferred if provider in available]
    if not selected:
        return {"ran": False, "reason": f"No supported ONNX Runtime provider found. Available: {sorted(available)}"}

    session = ort.InferenceSession(str(onnx_path), providers=selected)
    outputs = session.run(
        None,
        {
            "input_ids": input_ids.detach().cpu().numpy(),
            "attention_mask": attention_mask.detach().cpu().numpy(),
        },
    )
    return {
        "ran": True,
        "available_providers": sorted(available),
        "providers": selected,
        "session_providers": session.get_providers(),
        "output_shapes": [list(output.shape) for output in outputs],
    }


def _export_with_compatible_kwargs(wrapper: torch.nn.Module, inputs: tuple[torch.Tensor, torch.Tensor], onnx_path: Path, kwargs: dict[str, Any]) -> None:
    signature = inspect.signature(torch.onnx.export)
    supported = {key: value for key, value in kwargs.items() if key in signature.parameters}
    if "external_data" not in supported and "use_external_data_format" in signature.parameters:
        supported["use_external_data_format"] = kwargs["external_data"]
    torch.onnx.export(wrapper, inputs, str(onnx_path), **supported)


def export_onnx(
    config: dict[str, Any],
    *,
    model_path: Path | None = None,
    output_dir: Path | None = None,
    output_name: str | None = None,
    prompt: str | None = None,
    dummy_seq_length: int | None = None,
    opset: int | None = None,
    dtype: str | None = None,
    device: str | None = None,
    attn_implementation: str | None = None,
    external_data: bool | None = None,
    validate: bool | None = None,
    ort_check: bool | None = None,
) -> dict[str, Any]:
    logger = setup_logging("export_onnx")
    training_cfg = config.get("training", {})
    onnx_cfg = config.get("onnx", {})
    trust_remote_code = bool(config.get("trust_remote_code", True))

    output_name = output_name or onnx_cfg.get("output_name", "model.onnx")
    prompt = prompt or onnx_cfg.get("prompt", "Explain the main documented domain concepts.")
    dummy_seq_length = int(dummy_seq_length or onnx_cfg.get("dummy_seq_length", 64))
    opset = int(opset or onnx_cfg.get("opset", 17))
    dtype = dtype or onnx_cfg.get("dtype", "float32")
    device = device or onnx_cfg.get("device", "auto")
    attn_implementation = attn_implementation or onnx_cfg.get("attn_implementation", "eager")
    external_data = bool(onnx_cfg.get("external_data", True) if external_data is None else external_data)
    validate = bool(onnx_cfg.get("validate", True) if validate is None else validate)
    ort_check = bool(onnx_cfg.get("ort_check", False) if ort_check is None else ort_check)

    model_path = model_path or resolve_training_path(training_cfg.get("merged_output_dir"), "outputs/merged_model")
    output_dir = output_dir or resolve_training_path(onnx_cfg.get("output_dir"), "outputs/onnx_model")
    onnx_path = output_dir / output_name
    if not model_path.exists():
        raise FileNotFoundError(f"Merged model path not found: {model_path}")
    _require_onnx_package()

    selected_device = _select_device(device)
    torch_dtype = torch_dtype_from_config(dtype, torch)
    model_kwargs: dict[str, Any] = {}
    if torch_dtype != "auto":
        model_kwargs["torch_dtype"] = torch_dtype
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    logger.info("Loading model for ONNX export from %s", model_path)
    tokenizer = load_tokenizer(model_path, trust_remote_code)
    model = load_transformers_model(str(model_path), trust_remote_code=trust_remote_code, logger=logger, **model_kwargs)
    configure_generation_tokens(model, tokenizer)
    if attn_implementation:
        model.config._attn_implementation = attn_implementation
    disable_cache(model)
    model.eval()
    model.to(selected_device)
    wrapper = CausalLMOnnxWrapper(model).eval().to(selected_device)

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max(1, dummy_seq_length),
        padding="max_length",
    )
    input_ids = encoded["input_ids"].to(selected_device)
    attention_mask = encoded["attention_mask"].to(selected_device)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Exporting ONNX model to %s", onnx_path)
    with torch.no_grad():
        _export_with_compatible_kwargs(
            wrapper,
            (input_ids, attention_mask),
            onnx_path,
            {
                "input_names": ["input_ids", "attention_mask"],
                "output_names": ["logits"],
                "dynamic_axes": {
                    "input_ids": {0: "batch", 1: "sequence"},
                    "attention_mask": {0: "batch", 1: "sequence"},
                    "logits": {0: "batch", 1: "sequence"},
                },
                "opset_version": opset,
                "do_constant_folding": True,
                "external_data": external_data,
                "dynamo": False,
            },
        )
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))

    onnx_checker_passed = _validate_onnx_file(onnx_path, logger) if validate else False
    ort_result = _run_ort_shape_check(onnx_path, input_ids, attention_mask, logger) if ort_check else {"ran": False}
    report = {
        "status": "completed",
        "model_path": str(model_path),
        "onnx_path": str(onnx_path),
        "output_dir": str(output_dir),
        "opset": opset,
        "dtype": dtype,
        "device": str(selected_device),
        "attn_implementation": attn_implementation,
        "external_data": external_data,
        "dummy_seq_length": dummy_seq_length,
        "onnx_checker_passed": onnx_checker_passed,
        "onnxruntime_check": ort_result,
        "created_at_utc": utc_now(),
    }
    write_json(output_dir / "onnx_export_report.json", report)
    write_text(
        output_dir / "README.md",
        "\n".join(
            [
                "# ONNX Export",
                "",
                f"- Source model: `{model_path}`",
                f"- ONNX file: `{onnx_path.name}`",
                f"- Opset: `{opset}`",
                f"- External data: `{external_data}`",
                "",
                "This export is a no-cache CausalLM logits graph with dynamic batch and sequence axes.",
                "Use ONNX Runtime providers such as CUDAExecutionProvider or CPUExecutionProvider for inference.",
                "",
            ]
        ),
    )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a merged Hugging Face CausalLM model to ONNX.")
    parser.add_argument("--config", default="configs/domain_post_training.yaml")
    parser.add_argument("--model_path", default=None, help="Merged model directory. Defaults to training.merged_output_dir.")
    parser.add_argument("--output_dir", default=None, help="ONNX output directory. Defaults to onnx.output_dir.")
    parser.add_argument("--output_name", default=None, help="ONNX file name. Defaults to onnx.output_name.")
    parser.add_argument("--prompt", default=None, help="Prompt used to build dummy export inputs.")
    parser.add_argument("--dummy_seq_length", type=int, default=None)
    parser.add_argument("--opset", type=int, default=None)
    parser.add_argument("--dtype", default=None, choices=["auto", "float32", "float16", "bfloat16", "fp32", "fp16", "bf16"])
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--attn_implementation", default=None, help="Attention implementation for export. Defaults to onnx.attn_implementation.")
    external_group = parser.add_mutually_exclusive_group()
    external_group.add_argument("--external_data", dest="external_data", action="store_true", help="Enable ONNX external data files.")
    external_group.add_argument("--no_external_data", dest="external_data", action="store_false", help="Disable ONNX external data files.")
    parser.set_defaults(external_data=None)
    validate_group = parser.add_mutually_exclusive_group()
    validate_group.add_argument("--validate", dest="validate", action="store_true", help="Run onnx.checker validation.")
    validate_group.add_argument("--no_validate", dest="validate", action="store_false", help="Skip onnx.checker validation.")
    parser.set_defaults(validate=None)
    parser.add_argument("--ort_check", action="store_true", default=None, help="Run an ONNX Runtime output shape check.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    logger = setup_logging("export_onnx", args.verbose)
    try:
        config, _ = load_config(args.config)
        model_path = Path(args.model_path).expanduser().resolve() if args.model_path else None
        output_dir = resolve_training_path(args.output_dir, "outputs/onnx_model") if args.output_dir else None
        report = export_onnx(
            config,
            model_path=model_path,
            output_dir=output_dir,
            output_name=args.output_name,
            prompt=args.prompt,
            dummy_seq_length=args.dummy_seq_length,
            opset=args.opset,
            dtype=args.dtype,
            device=args.device,
            attn_implementation=args.attn_implementation,
            external_data=args.external_data,
            validate=args.validate,
            ort_check=args.ort_check,
        )
        logger.info("ONNX export completed: %s", report["onnx_path"])
        return 0
    except Exception as exc:
        logger.error(short_error(exc))
        logger.debug(traceback.format_exc())
        return 8


if __name__ == "__main__":
    raise SystemExit(main())
