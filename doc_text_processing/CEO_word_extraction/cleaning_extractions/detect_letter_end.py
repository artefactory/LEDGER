"""Detect the end of each CEO / Shareholder letter using a local LLM.

Reads the markdown files produced by ``extract_letters.py`` (default:
``doc_text_processing/CEO_word_extraction/extractions/``) and asks the LLM —
served via vLLM with the OpenAI-compatible chat completion API — to locate
the boundary between the actual letter and the irrelevant content that
follows (TOC, 10-K boilerplate, financial tables, marketing inserts, ...).

To save output tokens the LLM is asked to return ONLY two short anchor
strings copied verbatim from the input:

  - ``end_quote``  : the literal last ~6-15 words of the letter (typically
                     the signature block: "Chief Executive Officer", etc.).
  - ``next_quote`` : the literal first ~6-15 words of the irrelevant
                     content that immediately follows, or ``null`` if the
                     letter runs to the end of the file.

The companion script ``apply_cleaning.py`` then deterministically truncates
each ``.md`` file at the corresponding offset.

Output format is strict JSON; if the model returns malformed JSON or anchor
strings that cannot be located in the input, a retry is triggered (up to
``--retries`` times) before the record is marked as ``failed``.

Results are written to ``end_markers.json`` (per-file decisions plus run
metadata) under this directory.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError as e:  # pragma: no cover
    sys.stderr.write("openai SDK not installed. Run: uv add openai\n")
    raise

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = HERE.parent / "extractions"
DEFAULT_OUTPUT = HERE / "end_markers.json"


SYSTEM_PROMPT = """You are a precise document-segmentation assistant.

You receive the raw text of a markdown file extracted from an annual report.
The file BEGINS with a CEO / Chairman / Shareholder letter, but the
extraction window was deliberately oversized and usually includes UNRELATED
content after the letter ends (table of contents, 10-K boilerplate, financial
statements, marketing inserts, business overview, ...).

Your task: identify EXACTLY where the letter ends.

A letter typically ends with a closing salutation ("Sincerely,", "Yours
truly,", "Respectfully,", ...) followed by the signer's name and title
(e.g. "Chief Executive Officer", "Chairman of the Board"). When two or more
co-signers are present (e.g. CEO + Chairman), the letter ends after the LAST
signature block. Image placeholders (e.g. ![](images/...)) for handwritten
signatures may appear inside the signature block — keep them.

Return ONLY a JSON object with two fields, copied VERBATIM from the input:

{
  "end_quote": "<the last 6-15 words of the letter, exactly as they appear>",
  "next_quote": "<the first 6-15 words of the irrelevant content that follows, exactly as it appears, or null if the letter runs to the end of the file>"
}

Rules:
- Both strings MUST be substrings of the input — do not paraphrase, fix
  typos, or reorder words.
- Keep punctuation, capitalisation and OCR artefacts as-is.
- Do NOT include the page-split marker "<--- Page Split --->" inside either
  quote.
- Do NOT include code fences, commentary, or any text outside the JSON
  object.
- If the entire file is a single letter with no trailing irrelevant
  content, set "next_quote" to null."""


FEW_SHOTS: list[dict[str, str]] = [
    {
        "input": (
            "# TO OUR SHAREHOLDERS\n\n"
            "Dear Shareholders,\n\n"
            "Fiscal 2020 was a year of transformation for Acme Corp. We launched "
            "two new product lines and grew revenue by 14%.\n\n"
            "We thank you for your continued support.\n\n"
            "Sincerely,\n\n"
            "![](images/sig_0.jpg)\n\n"
            "Jane R. Doe\n"
            "President and Chief Executive Officer\n\n"
            "<--- Page Split --->\n\n"
            "## TABLE OF CONTENTS\n\n"
            "Item 1. Business ... 5\n"
            "Item 1A. Risk Factors ... 17\n"
        ),
        "output": json.dumps(
            {
                "end_quote": "Jane R. Doe\nPresident and Chief Executive Officer",
                "next_quote": "## TABLE OF CONTENTS",
            }
        ),
    },
    {
        "input": (
            "# CHAIRMAN'S STATEMENT\n\n"
            "I am pleased to present the annual results for the year ended 31 "
            "December 2018.\n\n"
            "Trading was robust across all three divisions and the Board has "
            "recommended a final dividend of 4.5p per share.\n\n"
            "Yours faithfully,\n\n"
            "Robert Smith\n"
            "Non-Executive Chairman\n"
            "12 March 2019\n"
        ),
        "output": json.dumps(
            {
                "end_quote": "Robert Smith\nNon-Executive Chairman\n12 March 2019",
                "next_quote": None,
            }
        ),
    },
    {
        "input": (
            "FELLOW SHAREHOLDERS,\n\n"
            "Our 2021 results reflect the resilience of our diversified model.\n\n"
            "Sincerely,\n\n"
            "![](images/4_0.jpg)\n\n"
            "Thomas A. Burke\n"
            "President and Chief Executive Officer\n\n"
            "![](images/4_1.jpg)\n\n"
            "Marsha C. Williams\n"
            "Lead Independent Director\n\n"
            "<--- Page Split --->\n\n"
            "## VEHICULAR THERMAL SOLUTIONS\n\n"
            "The vehicular industry is pushing for higher-efficiency products...\n"
        ),
        "output": json.dumps(
            {
                "end_quote": "Marsha C. Williams\nLead Independent Director",
                "next_quote": "## VEHICULAR THERMAL SOLUTIONS",
            }
        ),
    },
]


@dataclass
class Decision:
    name: str
    status: str  # ok | failed
    end_quote: str | None = None
    next_quote: str | None = None
    end_offset: int | None = None  # char offset of end_quote's last char (+1)
    next_offset: int | None = None  # char offset of next_quote's first char
    attempts: int = 0
    error: str | None = None
    raw_responses: list[str] = field(default_factory=list)


def build_messages(text: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for shot in FEW_SHOTS:
        messages.append({"role": "user", "content": shot["input"]})
        messages.append({"role": "assistant", "content": shot["output"]})
    messages.append({"role": "user", "content": text})
    return messages


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_json_response(raw: str) -> dict[str, Any]:
    """Robustly parse a JSON object from the model's reply.

    Strips markdown code fences if present and falls back to the first
    ``{...}`` block in the string.
    """
    s = _FENCE_RE.sub("", raw).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Greedy fallback: outermost {...}
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no JSON object found in response: {raw!r}")
    return json.loads(s[start : end + 1])


def normalise_quote(q: str | None) -> str | None:
    """Strip surrounding whitespace; collapse internal whitespace runs.

    The model occasionally drops or adds whitespace inside multi-line quotes
    (e.g. converting "\\n\\n" to "\\n"). The locator below handles that by
    matching against a whitespace-tolerant regex, but we still trim the
    edges here to keep the JSON output tidy.
    """
    if q is None:
        return None
    return q.strip()


def locate(haystack: str, needle: str) -> tuple[int, int] | None:
    """Find ``needle`` inside ``haystack``, tolerating whitespace differences.

    Returns ``(start, end)`` byte offsets of the first match, or ``None``.
    Tries an exact match first for speed, then a regex with ``\\s+`` between
    each whitespace-separated chunk.
    """
    idx = haystack.find(needle)
    if idx != -1:
        return (idx, idx + len(needle))
    chunks = [re.escape(c) for c in needle.split()]
    if not chunks:
        return None
    pattern = re.compile(r"\s+".join(chunks))
    m = pattern.search(haystack)
    if m is None:
        return None
    return (m.start(), m.end())


def call_model(
    client: OpenAI,
    model: str,
    text: str,
    *,
    temperature: float,
    max_tokens: int,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
    top_k: int = -1,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    enable_thinking: bool = False,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=build_messages(text),
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        presence_penalty=presence_penalty,
        response_format={"type": "json_object"},
        # vLLM-specific extensions — ignored by non-vLLM endpoints.
        extra_body={
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            # Disable Qwen3 thinking mode so output tokens go to the JSON
            # response, not the reasoning trace. Set enable_thinking=True (and
            # raise max_tokens accordingly) if you want chain-of-thought.
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        },
    )
    return resp.choices[0].message.content or ""


def process_file(
    path: Path,
    client: OpenAI,
    model: str,
    *,
    retries: int,
    temperature: float,
    max_tokens: int,
    top_p: float = 1.0,
    presence_penalty: float = 0.0,
    top_k: int = -1,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    enable_thinking: bool = False,
) -> Decision:
    text = path.read_text(encoding="utf-8")
    decision = Decision(name=path.stem, status="failed")

    for attempt in range(1, retries + 1):
        decision.attempts = attempt
        try:
            raw = call_model(
                client,
                model,
                text,
                # Bump temperature slightly on retry to escape a sticky bad output.
                temperature=temperature + 0.2 * (attempt - 1),
                max_tokens=max_tokens,
                top_p=top_p,
                presence_penalty=presence_penalty,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                enable_thinking=enable_thinking,
            )
        except Exception as e:
            decision.error = f"api_error: {e!r}"
            time.sleep(0.5)
            continue

        decision.raw_responses.append(raw)

        try:
            obj = parse_json_response(raw)
        except (ValueError, json.JSONDecodeError) as e:
            decision.error = f"bad_json: {e}"
            continue

        end_q = normalise_quote(obj.get("end_quote"))
        next_q = normalise_quote(obj.get("next_quote"))
        if not end_q:
            decision.error = "missing end_quote"
            continue

        end_loc = locate(text, end_q)
        if end_loc is None:
            decision.error = f"end_quote not found in source: {end_q!r}"
            continue

        next_loc = None
        if next_q is not None:
            next_loc = locate(text, next_q)
            if next_loc is None:
                decision.error = f"next_quote not found in source: {next_q!r}"
                continue
            if next_loc[0] < end_loc[1]:
                decision.error = (
                    f"next_quote ({next_loc[0]}) precedes end_quote ({end_loc[1]})"
                )
                continue

        decision.status = "ok"
        decision.end_quote = end_q
        decision.next_quote = next_q
        decision.end_offset = end_loc[1]
        decision.next_offset = next_loc[0] if next_loc else None
        decision.error = None
        return decision

    return decision


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible endpoint (vLLM default: http://localhost:8000/v1)",
    )
    ap.add_argument("--api-key", default="EMPTY", help="usually unused for local vLLM")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "total token budget per call. "
            "Default: 512 without thinking, 4096 with --thinking "
            "(reasoning trace + JSON both count against this limit)"
        ),
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=12,
        help="parallel in-flight requests against the vLLM server",
    )
    ap.add_argument("--limit", type=int, default=None, help="process at most N files")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="skip files already present (status=ok) in --output",
    )
    ap.add_argument(
        "--thinking",
        action="store_true",
        default=False,
        help=(
            "enable Qwen3 chain-of-thought thinking mode "
            "(disabled by default — thinking tokens eat into max_tokens budget)"
        ),
    )
    args = ap.parse_args()

    if args.max_tokens is None:
        args.max_tokens = 8192 if args.thinking else 512

    if not args.input_dir.is_dir():
        sys.exit(f"input dir not found: {args.input_dir}")

    files = sorted(p for p in args.input_dir.glob("*.md") if p.is_file())
    if args.limit:
        files = files[: args.limit]

    existing: dict[str, dict[str, Any]] = {}
    if args.resume and args.output.exists():
        prev = json.loads(args.output.read_text())
        for rec in prev.get("decisions", []):
            if rec.get("status") == "ok":
                existing[rec["name"]] = rec
        files = [p for p in files if p.stem not in existing]

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    print(
        f"Processing {len(files)} files "
        f"(model={args.model}, concurrency={args.concurrency}, retries={args.retries})",
        file=sys.stderr,
    )

    results: list[Decision] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                process_file,
                p,
                client,
                args.model,
                retries=args.retries,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
                presence_penalty=1.5,
                repetition_penalty=1.0,
                enable_thinking=args.thinking,
            ): p
            for p in files
        }
        for i, fut in enumerate(as_completed(futures), 1):
            path = futures[fut]
            d = fut.result()
            results.append(d)
            tag = "OK " if d.status == "ok" else "FAIL"
            print(
                f"[{i}/{len(files)}] {tag} {path.name} ({d.attempts} try)",
                file=sys.stderr,
            )
            if d.status != "ok":
                print(f"    reason: {d.error}", file=sys.stderr)

    decisions = [asdict(d) for d in results]
    decisions.extend(existing.values())
    decisions.sort(key=lambda r: r["name"])

    payload = {
        "meta": {
            "model": args.model,
            "base_url": args.base_url,
            "input_dir": str(args.input_dir),
            "n_files": len(decisions),
            "n_ok": sum(1 for r in decisions if r["status"] == "ok"),
            "n_failed": sum(1 for r in decisions if r["status"] != "ok"),
            "retries": args.retries,
            "temperature": args.temperature,
        },
        "decisions": decisions,
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(
        f"Wrote {args.output} — ok={payload['meta']['n_ok']} "
        f"failed={payload['meta']['n_failed']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
