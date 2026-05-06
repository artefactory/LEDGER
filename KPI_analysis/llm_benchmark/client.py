"""vLLM / OpenAI-compatible chat client for the KPI extraction benchmark.

Mirrors the proven pattern in
``doc_text_processing/CEO_word_extraction/cleaning_extractions/detect_letter_end.py``:
- ``openai.OpenAI`` against a configurable ``base_url`` (default
  ``http://localhost:8000/v1`` for a local vLLM server).
- ``response_format`` carries a JSON schema so vLLM's xgrammar guided
  decoding constrains the output. (xgrammar is selected at vLLM startup via
  ``--guided-decoding-backend xgrammar`` — that is the server operator's
  responsibility.)
- ``extra_body`` carries vLLM-specific knobs: thinking-mode toggle,
  ``top_k``, ``min_p``, ``repetition_penalty``.
- A small retry loop bumps temperature on parse / schema-validation errors
  before giving up.

Defensive: even with xgrammar enforcing the schema, we still strip stray
markdown fences and validate via Pydantic before trusting the output.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

try:
    from openai import APIError, OpenAI
except ImportError:  # pragma: no cover
    sys.stderr.write("openai SDK not installed. Run: uv add openai\n")
    raise

from pydantic import ValidationError

from schema import ReportExtraction


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _strip_fences(raw: str) -> str:
    return _FENCE_RE.sub("", raw).strip()


def _parse_json_object(raw: str) -> dict[str, Any]:
    """Parse the LLM's reply as a JSON object, with a permissive fallback."""
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


@dataclass
class CallResult:
    """One LLM call's outcome."""

    extraction: ReportExtraction | None
    raw_response: str
    attempts: int
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None
    raw_history: list[str] = field(default_factory=list)


def make_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


def call_extraction(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 4096,
    temperature: float = 0.5,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    enable_thinking: bool | None = None,
    reasoning_effort: str | None = None,
    retries: int = 3,
    schema_name: str = "ReportExtraction",
) -> CallResult:
    """Call the LLM, validate the response against ``ReportExtraction``, retry on failure.

    Temperature is bumped on each retry to break out of stuck decodes:
    ``temperature``, ``temperature + 0.2``, ``temperature + 0.4``, ...
    """
    schema_dict = ReportExtraction.model_json_schema()
    response_format: dict[str, Any] = {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": schema_dict,
            "strict": True,
        },
    }
    extra_body: dict[str, Any] = {
        "top_k": top_k,
        "min_p": min_p,
        "repetition_penalty": repetition_penalty,
    }
    # ``chat_template_kwargs`` is template-specific. Only emit keys that the
    # caller explicitly set, so we don't push an unknown kwarg into a
    # template that doesn't expect it.
    # - Qwen3 / Nemotron Nano 3 understand ``enable_thinking`` (True/False).
    # - gpt-oss (Harmony) understands ``reasoning_effort`` ("low"/"medium"/"high").
    # - Mistral templates have no thinking toggle — leave both unset.
    template_kwargs: dict[str, Any] = {}
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking
    if reasoning_effort is not None:
        template_kwargs["reasoning_effort"] = reasoning_effort
    if template_kwargs:
        extra_body["chat_template_kwargs"] = template_kwargs

    history: list[str] = []
    last_error: str | None = None
    last_raw = ""
    last_prompt_tokens: int | None = None
    last_completion_tokens: int | None = None
    started = time.monotonic()

    for attempt in range(1, retries + 1):
        attempt_temp = temperature + 0.2 * (attempt - 1)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=attempt_temp,
                max_tokens=max_tokens,
                top_p=top_p,
                response_format=response_format,
                extra_body=extra_body,
            )
        except APIError as e:
            last_error = f"api_error: {e}"
            history.append(last_error)
            time.sleep(min(2**attempt, 10))
            continue

        last_raw = resp.choices[0].message.content or ""
        history.append(last_raw)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            last_prompt_tokens = getattr(usage, "prompt_tokens", None)
            last_completion_tokens = getattr(usage, "completion_tokens", None)

        try:
            obj = _parse_json_object(last_raw)
        except (ValueError, json.JSONDecodeError) as e:
            last_error = f"json_parse_error: {e}"
            continue

        try:
            extraction = ReportExtraction.model_validate(obj)
        except ValidationError as e:
            last_error = f"schema_validation_error: {e}"
            continue

        return CallResult(
            extraction=extraction,
            raw_response=last_raw,
            attempts=attempt,
            latency_s=time.monotonic() - started,
            prompt_tokens=last_prompt_tokens,
            completion_tokens=last_completion_tokens,
            error=None,
            raw_history=history,
        )

    return CallResult(
        extraction=None,
        raw_response=last_raw,
        attempts=retries,
        latency_s=time.monotonic() - started,
        prompt_tokens=last_prompt_tokens,
        completion_tokens=last_completion_tokens,
        error=last_error or "unknown_error",
        raw_history=history,
    )
