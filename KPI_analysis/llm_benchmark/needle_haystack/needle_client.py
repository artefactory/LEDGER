"""Deterministic vLLM / OpenAI-compatible client for the needle benchmark.

Differences from the multi-KPI ``client.py``, all in service of a *reproducible*
benchmark:

- Greedy decoding by default (``temperature=0``, ``top_p=1``) with a fixed
  per-request ``seed``. Two runs of the same model + prompt give identical
  outputs.
- No temperature bumping between retries. xgrammar already constrains output to
  the schema grammar, so genuine parse failures are near-impossible; a retry
  only helps if the first completion was truncated (hit ``max_tokens``). Retries
  therefore reuse the same decoding params but bump ``max_tokens`` and the seed,
  and the fact that a retry happened is recorded (``attempts``) for audit.
- Captures the full ``usage`` payload, including
  ``prompt_tokens_details.cached_tokens`` when vLLM reports it — direct evidence
  that prefix caching is serving the document prefill.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from openai import APIError, OpenAI
from pydantic import ValidationError

from needle_schema import NeedleAnswer


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_fences(raw: str) -> str:
    return _FENCE_RE.sub("", raw).strip()


def _parse_json_object(raw: str) -> dict[str, Any]:
    s = _strip_fences(raw)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object found in response: {raw!r}")
    return json.loads(s[start : end + 1])


def _usage_to_dict(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    try:
        return usage.model_dump()
    except Exception:
        out: dict[str, Any] = {}
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            v = getattr(usage, k, None)
            if v is not None:
                out[k] = v
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            out["prompt_tokens_details"] = {
                "cached_tokens": getattr(details, "cached_tokens", None)
            }
        return out or None


def _cached_tokens(usage_dict: dict[str, Any] | None) -> int | None:
    if not usage_dict:
        return None
    details = usage_dict.get("prompt_tokens_details") or {}
    if isinstance(details, dict):
        return details.get("cached_tokens")
    return None


@dataclass
class NeedleResult:
    answer: NeedleAnswer | None
    raw_response: str
    attempts: int
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    usage: dict[str, Any] | None = None
    error: str | None = None
    raw_history: list[str] = field(default_factory=list)


def make_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


_SCHEMA_DICT = NeedleAnswer.model_json_schema()
_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {"name": "NeedleAnswer", "schema": _SCHEMA_DICT, "strict": True},
}


def call_needle(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int = 1234,
    enable_thinking: bool | None = False,
    reasoning_effort: str | None = None,
    extra_body_overrides: dict[str, Any] | None = None,
    retries: int = 2,
) -> NeedleResult:
    """Call the model for one needle query and validate against ``NeedleAnswer``.

    Deterministic: same (model, messages, seed, params) -> same output. Retries
    keep the params fixed but bump ``max_tokens`` (to recover from a truncated
    JSON) and the seed; ``attempts`` records how many were needed.
    """
    extra_body: dict[str, Any] = {}
    if enable_thinking:
        extra_body["chat_template_kwargs"] = {"enable_thinking": True}
    elif enable_thinking is False:
        # Some Qwen3-style templates default thinking ON; turn it off explicitly
        # so a needle answer is a bare JSON object (and stays deterministic).
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    if reasoning_effort is not None:
        extra_body["reasoning_effort"] = reasoning_effort
    if extra_body_overrides:
        extra_body.update(extra_body_overrides)

    history: list[str] = []
    last_error: str | None = None
    last_raw = ""
    last_usage: dict[str, Any] | None = None
    started = time.monotonic()

    for attempt in range(1, retries + 1):
        attempt_max_tokens = max_tokens * attempt  # grow only if retrying
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                top_p=top_p,
                max_tokens=attempt_max_tokens,
                seed=seed + attempt - 1,
                response_format=_RESPONSE_FORMAT,
                extra_body=extra_body,
            )
        except APIError as e:
            last_error = f"api_error: {e}"
            history.append(last_error)
            time.sleep(min(2**attempt, 8))
            continue
        except Exception as e:  # noqa: BLE001
            last_error = f"client_error: {type(e).__name__}: {e}"
            history.append(last_error)
            time.sleep(min(2**attempt, 8))
            continue

        last_raw = resp.choices[0].message.content or ""
        history.append(last_raw)
        last_usage = _usage_to_dict(getattr(resp, "usage", None))

        try:
            obj = _parse_json_object(last_raw)
        except (ValueError, json.JSONDecodeError) as e:
            last_error = f"json_parse_error: {e}"
            continue
        try:
            answer = NeedleAnswer.model_validate(obj)
        except ValidationError as e:
            last_error = f"schema_validation_error: {e}"
            continue

        return NeedleResult(
            answer=answer,
            raw_response=last_raw,
            attempts=attempt,
            latency_s=time.monotonic() - started,
            prompt_tokens=(last_usage or {}).get("prompt_tokens"),
            completion_tokens=(last_usage or {}).get("completion_tokens"),
            cached_tokens=_cached_tokens(last_usage),
            usage=last_usage,
            error=None,
            raw_history=history,
        )

    return NeedleResult(
        answer=None,
        raw_response=last_raw,
        attempts=retries,
        latency_s=time.monotonic() - started,
        prompt_tokens=(last_usage or {}).get("prompt_tokens"),
        completion_tokens=(last_usage or {}).get("completion_tokens"),
        cached_tokens=_cached_tokens(last_usage),
        usage=last_usage,
        error=last_error or "unknown_error",
        raw_history=history,
    )
