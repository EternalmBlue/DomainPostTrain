from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from functools import partial
from pathlib import Path
from typing import Any, Callable


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
DEFAULT_REFUSAL_TERMS = (
    "cannot",
    "can't",
    "do not",
    "not able",
    "not provide",
    "does not specify",
    "not specified",
    "no documentation",
)
DEFAULT_REWARD_JUDGE_API_KEY_ENV = "GRPO_REWARD_JUDGE_API_KEY"
DEFAULT_REWARD_JUDGE_PROMPT_TEMPLATE = """Evaluate the assistant completion against the prompt and reward signals.
Return only a JSON object with this shape: {{"score": <number>, "reason": "<brief reason>"}}.
The score must be between {score_min} and {score_max}.

Prompt:
{prompt}

Assistant completion:
{completion}

Reference answer:
{reference_answer}

Required terms:
{required_terms}

Forbidden terms:
{forbidden_terms}

Must refuse:
{must_refuse}

Category:
{category}
"""


def as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if "\n" in stripped:
            return [part.strip() for part in stripped.splitlines() if part.strip()]
        if "," in stripped:
            return [part.strip() for part in stripped.split(",") if part.strip()]
        return [stripped]
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            result.extend(as_text_list(item))
        return result
    return [str(value).strip()] if str(value).strip() else []


def as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "must_refuse", "refusal"}
    return bool(value)


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content") or completion.get("text") or "")
    if isinstance(completion, list):
        parts = []
        for item in completion:
            text = completion_to_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    return str(completion or "")


def _normalise_prompt_from_messages(messages: list[dict[str, Any]]) -> str:
    prompt_parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if role == "assistant":
            break
        label = "System" if role == "system" else "User" if role == "user" else role.title() or "Message"
        prompt_parts.append(f"{label}: {content}")
    if not prompt_parts:
        raise ValueError("messages prompt has no non-assistant content")
    return "\n".join(prompt_parts).strip() + "\nAssistant:"


def _build_prompt(item: dict[str, Any], system_prompt: str) -> str:
    if isinstance(item.get("messages"), list):
        return _normalise_prompt_from_messages(item["messages"])
    prompt = str(item.get("prompt") or item.get("instruction") or item.get("question") or "").strip()
    if not prompt:
        raise ValueError("missing prompt")
    context = str(item.get("context") or item.get("input") or "").strip()
    if str(item.get("include_system_prompt", "true")).strip().lower() in {"0", "false", "no"}:
        lines = [f"User: {prompt}"]
    else:
        lines = [f"System: {system_prompt}", f"User: {prompt}"]
    if context:
        lines.append(f"Context: {context}")
    return "\n".join(lines).strip() + "\nAssistant:"


def normalise_grpo_record(
    item: dict[str, Any],
    *,
    source_path: Path,
    source_index: int,
    system_prompt: str,
) -> dict[str, Any]:
    prompt = _build_prompt(item, system_prompt)
    reference = str(
        item.get("reference_answer")
        or item.get("answer")
        or item.get("solution")
        or item.get("ground_truth")
        or item.get("expected")
        or ""
    ).strip()
    required_terms = as_text_list(item.get("required_terms") or item.get("must_include") or item.get("keywords"))
    forbidden_terms = as_text_list(item.get("forbidden_terms") or item.get("must_not_include") or item.get("banned_terms"))
    must_refuse = as_bool(item.get("must_refuse") or item.get("requires_refusal"))
    category = str(item.get("category") or item.get("type") or item.get("task") or "general")
    if not any([reference, required_terms, forbidden_terms, must_refuse]):
        raise ValueError("GRPO row needs at least one reward signal: reference_answer, required_terms, forbidden_terms, or must_refuse")
    return {
        "id": str(item.get("id") or f"{source_path.stem}-{source_index:06d}"),
        "prompt": prompt,
        "reference_answer": reference,
        "required_terms": required_terms,
        "forbidden_terms": forbidden_terms,
        "must_refuse": must_refuse,
        "min_completion_chars": item.get("min_completion_chars"),
        "max_completion_chars": item.get("max_completion_chars"),
        "category": category,
        "origin": str(item.get("origin") or ""),
        "source": str(item.get("source") or ""),
        "source_path": str(source_path),
        "source_index": int(source_index),
    }


def _token_set(text: str) -> set[str]:
    return {match.group(0).lower() for match in TOKEN_RE.finditer(text)}


def _values_for_batch(value: Any, batch_size: int) -> list[Any]:
    if isinstance(value, list) and len(value) == batch_size:
        return value
    return [value for _ in range(batch_size)]


def reference_overlap_reward(completions: list[Any], reference_answer: Any = None, **_: Any) -> list[float]:
    references = _values_for_batch(reference_answer, len(completions))
    rewards: list[float] = []
    for completion, reference in zip(completions, references):
        reference_text = completion_to_text(reference).strip()
        if not reference_text:
            rewards.append(0.0)
            continue
        completion_tokens = _token_set(completion_to_text(completion))
        reference_tokens = _token_set(reference_text)
        if not reference_tokens:
            rewards.append(0.0)
            continue
        rewards.append(len(completion_tokens & reference_tokens) / len(reference_tokens))
    return rewards


def term_constraint_reward(
    completions: list[Any],
    required_terms: Any = None,
    forbidden_terms: Any = None,
    **_: Any,
) -> list[float]:
    required_batch = _values_for_batch(required_terms, len(completions))
    forbidden_batch = _values_for_batch(forbidden_terms, len(completions))
    rewards: list[float] = []
    for completion, required, forbidden in zip(completions, required_batch, forbidden_batch):
        text = completion_to_text(completion).lower()
        required_list = [term.lower() for term in as_text_list(required)]
        forbidden_list = [term.lower() for term in as_text_list(forbidden)]
        score = 0.0
        if required_list:
            score += sum(1.0 for term in required_list if term in text) / len(required_list)
        if forbidden_list:
            score -= sum(1.0 for term in forbidden_list if term in text) / len(forbidden_list)
        rewards.append(score)
    return rewards


def refusal_reward(
    completions: list[Any],
    must_refuse: Any = None,
    *,
    refusal_terms: list[str] | tuple[str, ...] = DEFAULT_REFUSAL_TERMS,
    **_: Any,
) -> list[float]:
    refuse_flags = _values_for_batch(must_refuse, len(completions))
    terms = [term.lower() for term in refusal_terms]
    rewards: list[float] = []
    for completion, flag in zip(completions, refuse_flags):
        if not as_bool(flag):
            rewards.append(0.0)
            continue
        text = completion_to_text(completion).lower()
        rewards.append(1.0 if any(term in text for term in terms) else -0.5)
    return rewards


def length_bounds_reward(
    completions: list[Any],
    min_completion_chars: Any = None,
    max_completion_chars: Any = None,
    **_: Any,
) -> list[float]:
    min_batch = _values_for_batch(min_completion_chars, len(completions))
    max_batch = _values_for_batch(max_completion_chars, len(completions))
    rewards: list[float] = []
    for completion, min_chars, max_chars in zip(completions, min_batch, max_batch):
        text_len = len(completion_to_text(completion).strip())
        if text_len == 0:
            rewards.append(-1.0)
            continue
        score = 0.0
        if min_chars not in (None, "") and text_len < int(min_chars):
            score -= 0.5
        if max_chars not in (None, "") and text_len > int(max_chars):
            score -= 0.5
        rewards.append(score)
    return rewards


def reward_judge_enabled(grpo_cfg: dict[str, Any]) -> bool:
    reward_judge = grpo_cfg.get("reward_judge") or {}
    if not isinstance(reward_judge, dict):
        raise ValueError("grpo.reward_judge must be an object")
    return as_bool(reward_judge.get("enabled", False))


def _score_range(value: Any) -> tuple[float, float]:
    raw = value if value is not None else [0.0, 1.0]
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("grpo.reward_judge.score_range must contain exactly two numbers")
    score_min = float(raw[0])
    score_max = float(raw[1])
    if score_min > score_max:
        raise ValueError("grpo.reward_judge.score_range minimum cannot exceed maximum")
    return score_min, score_max


def _positive_float(value: Any, *, field: str, default: float) -> float:
    result = float(value if value is not None else default)
    if result <= 0:
        raise ValueError(f"{field} must be greater than 0")
    return result


def _non_negative_int(value: Any, *, field: str, default: int) -> int:
    result = int(value if value is not None else default)
    if result < 0:
        raise ValueError(f"{field} must be greater than or equal to 0")
    return result


def _reward_judge_config(grpo_cfg: dict[str, Any]) -> dict[str, Any]:
    reward_judge = grpo_cfg.get("reward_judge") or {}
    if not isinstance(reward_judge, dict):
        raise ValueError("grpo.reward_judge must be an object")
    enabled = as_bool(reward_judge.get("enabled", False))
    score_min, score_max = _score_range(reward_judge.get("score_range"))
    config = {
        "enabled": enabled,
        "base_url": str(reward_judge.get("base_url") or "").strip().rstrip("/"),
        "api_key_env": str(reward_judge.get("api_key_env") or DEFAULT_REWARD_JUDGE_API_KEY_ENV).strip(),
        "model": str(reward_judge.get("model") or "").strip(),
        "score_range": (score_min, score_max),
        "timeout_seconds": _positive_float(
            reward_judge.get("timeout_seconds"),
            field="grpo.reward_judge.timeout_seconds",
            default=30.0,
        ),
        "max_retries": _non_negative_int(
            reward_judge.get("max_retries"),
            field="grpo.reward_judge.max_retries",
            default=2,
        ),
        "prompt_template": str(reward_judge.get("prompt_template") or DEFAULT_REWARD_JUDGE_PROMPT_TEMPLATE),
    }
    if not enabled:
        return config
    if not config["base_url"]:
        raise ValueError("grpo.reward_judge.base_url is required when reward_judge.enabled is true")
    if not config["model"]:
        raise ValueError("grpo.reward_judge.model is required when reward_judge.enabled is true")
    if not config["api_key_env"]:
        raise ValueError("grpo.reward_judge.api_key_env is required when reward_judge.enabled is true")
    api_key = os.environ.get(config["api_key_env"])
    if not api_key:
        raise ValueError(f"Environment variable {config['api_key_env']} is required for grpo.reward_judge")
    config["api_key"] = api_key
    return config


def reward_judge_metadata(grpo_cfg: dict[str, Any]) -> dict[str, Any]:
    config = _reward_judge_config(grpo_cfg)
    score_min, score_max = config["score_range"]
    return {
        "enabled": config["enabled"],
        "base_url": config["base_url"],
        "api_key_env": config["api_key_env"],
        "model": config["model"],
        "score_range": [score_min, score_max],
        "timeout_seconds": config["timeout_seconds"],
        "max_retries": config["max_retries"],
    }


def _json_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _render_reward_judge_prompt(
    template: str,
    *,
    prompt: Any,
    completion: Any,
    reference_answer: Any,
    required_terms: Any,
    forbidden_terms: Any,
    must_refuse: Any,
    category: Any,
    score_min: float,
    score_max: float,
) -> str:
    values = {
        "prompt": completion_to_text(prompt),
        "completion": completion_to_text(completion),
        "reference_answer": completion_to_text(reference_answer),
        "required_terms": _json_text(required_terms),
        "forbidden_terms": _json_text(forbidden_terms),
        "must_refuse": _json_text(must_refuse),
        "category": _json_text(category),
        "score_min": score_min,
        "score_max": score_max,
    }
    try:
        return template.format(**values)
    except KeyError as exc:
        raise ValueError(f"Unknown grpo.reward_judge.prompt_template placeholder: {exc.args[0]}") from exc


def parse_reward_judge_score(content: Any, score_range: tuple[float, float] | list[float]) -> float:
    text = completion_to_text(content).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Reward judge response must be a JSON object with a numeric score")
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("Reward judge response must be a JSON object with a numeric score") from exc
    if not isinstance(payload, dict):
        raise ValueError("Reward judge response must be a JSON object with a numeric score")
    score = payload.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("Reward judge response must contain numeric score")
    score_min, score_max = _score_range(score_range)
    return min(score_max, max(score_min, float(score)))


def _reward_judge_endpoint(base_url: str) -> str:
    if base_url.endswith("/chat/completions"):
        return base_url
    return f"{base_url}/chat/completions"


def _extract_chat_completion_content(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise ValueError("Reward judge API response must be a JSON object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Reward judge API response must contain choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("Reward judge API choice must be an object")
    message = first.get("message")
    if isinstance(message, dict) and message.get("content") is not None:
        return completion_to_text(message["content"])
    if first.get("text") is not None:
        return completion_to_text(first["text"])
    raise ValueError("Reward judge API response did not include message.content")


def _call_openai_compatible_judge(config: dict[str, Any], prompt: str) -> str:
    endpoint = _reward_judge_endpoint(config["base_url"])
    payload = {
        "model": config["model"],
        "messages": [
            {
                "role": "system",
                "content": "You are a strict reward judge. Return only JSON and do not include markdown.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 256,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['api_key']}",
    }
    attempts = int(config["max_retries"]) + 1
    last_error: Exception | None = None
    for _ in range(attempts):
        request = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=float(config["timeout_seconds"])) as response:
                body = response.read().decode("utf-8")
            return _extract_chat_completion_content(json.loads(body))
        except (OSError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
    raise RuntimeError(f"Reward judge request failed after {attempts} attempt(s): {last_error}") from last_error


def build_openai_reward_judge(grpo_cfg: dict[str, Any]) -> Callable[..., list[float]]:
    config = _reward_judge_config(grpo_cfg)
    score_min, score_max = config["score_range"]

    def openai_compatible_reward_judge(
        completions: list[Any],
        prompt: Any = None,
        prompts: Any = None,
        reference_answer: Any = None,
        required_terms: Any = None,
        forbidden_terms: Any = None,
        must_refuse: Any = None,
        category: Any = None,
        **_: Any,
    ) -> list[float]:
        prompt_values = _values_for_batch(prompt if prompt is not None else prompts, len(completions))
        reference_values = _values_for_batch(reference_answer, len(completions))
        required_values = _values_for_batch(required_terms, len(completions))
        forbidden_values = _values_for_batch(forbidden_terms, len(completions))
        refusal_values = _values_for_batch(must_refuse, len(completions))
        category_values = _values_for_batch(category, len(completions))
        rewards: list[float] = []
        for index, completion in enumerate(completions):
            judge_prompt = _render_reward_judge_prompt(
                config["prompt_template"],
                prompt=prompt_values[index],
                completion=completion,
                reference_answer=reference_values[index],
                required_terms=required_values[index],
                forbidden_terms=forbidden_values[index],
                must_refuse=refusal_values[index],
                category=category_values[index],
                score_min=score_min,
                score_max=score_max,
            )
            response_content = _call_openai_compatible_judge(config, judge_prompt)
            rewards.append(parse_reward_judge_score(response_content, config["score_range"]))
        return rewards

    openai_compatible_reward_judge.__name__ = "openai_compatible_reward_judge"
    return openai_compatible_reward_judge


BUILTIN_REWARDS: dict[str, Callable[..., list[float]]] = {
    "reference_overlap": reference_overlap_reward,
    "term_constraints": term_constraint_reward,
    "refusal": refusal_reward,
    "length_bounds": length_bounds_reward,
}


def build_builtin_reward_functions(grpo_cfg: dict[str, Any]) -> list[Callable[..., list[float]]]:
    names = grpo_cfg.get("builtin_rewards", ["reference_overlap", "term_constraints", "refusal"])
    reward_names = as_text_list(names)
    functions: list[Callable[..., list[float]]] = []
    for name in reward_names:
        if name not in BUILTIN_REWARDS:
            raise ValueError(f"Unsupported GRPO builtin reward: {name}")
        func = BUILTIN_REWARDS[name]
        if name == "refusal" and grpo_cfg.get("refusal_terms"):
            configured = tuple(as_text_list(grpo_cfg.get("refusal_terms")))
            wrapped = partial(refusal_reward, refusal_terms=configured)
            wrapped.__name__ = "refusal_reward"
            functions.append(wrapped)
        else:
            functions.append(func)
    return functions


def build_grpo_reward_functions(grpo_cfg: dict[str, Any]) -> list[Any]:
    functions: list[Any] = build_builtin_reward_functions(grpo_cfg)
    if reward_judge_enabled(grpo_cfg):
        functions.append(build_openai_reward_judge(grpo_cfg))
    if not functions:
        raise RuntimeError("GRPO requires at least one built-in reward or enabled reward_judge.")
    return functions
