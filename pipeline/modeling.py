from __future__ import annotations

from pathlib import Path
from typing import Any

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from pipeline.utils import resolve_model_name_or_path

try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # pragma: no cover - depends on the installed Transformers build.
    AutoModelForImageTextToText = None


def load_tokenizer(model_name_or_path: str | Path, trust_remote_code: bool):
    resolved = resolve_model_name_or_path(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _select_model_loader(model_name_or_path: str | Path, trust_remote_code: bool, logger=None):
    resolved = resolve_model_name_or_path(model_name_or_path)
    config = AutoConfig.from_pretrained(resolved, trust_remote_code=trust_remote_code)
    architectures = list(getattr(config, "architectures", []) or [])
    model_type = str(getattr(config, "model_type", "") or "")
    uses_image_text_loader = model_type == "qwen3_5" or any("ForConditionalGeneration" in item for item in architectures)
    if uses_image_text_loader and AutoModelForImageTextToText is not None:
        if logger:
            logger.info("Using AutoModelForImageTextToText for model_type=%s architectures=%s", model_type, architectures)
        return AutoModelForImageTextToText
    if logger:
        logger.info("Using AutoModelForCausalLM for model_type=%s architectures=%s", model_type, architectures)
    return AutoModelForCausalLM


def load_transformers_model(
    model_name_or_path: str | Path,
    *,
    trust_remote_code: bool,
    logger=None,
    **model_kwargs: Any,
):
    resolved = resolve_model_name_or_path(model_name_or_path)
    loader = _select_model_loader(model_name_or_path, trust_remote_code, logger)
    return loader.from_pretrained(resolved, trust_remote_code=trust_remote_code, **model_kwargs)


def configure_generation_tokens(model, tokenizer) -> None:
    targets = [
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
        getattr(model, "generation_config", None),
    ]
    for target in targets:
        if target is None:
            continue
        target.eos_token_id = tokenizer.eos_token_id
        target.pad_token_id = tokenizer.pad_token_id


def disable_cache(model) -> None:
    targets = [getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)]
    for target in targets:
        if target is not None and hasattr(target, "use_cache"):
            target.use_cache = False
