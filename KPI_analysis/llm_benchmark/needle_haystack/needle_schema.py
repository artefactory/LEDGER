"""Pydantic schema for the needle-in-a-haystack KPI benchmark.

Each query asks for ONE KPI value for ONE fiscal year. The answer is a single,
short, structured JSON object — deliberately minimal so it is cheap to decode
(short completions keep the prefix-cached document the dominant cost) and
trivial to score.

The schema is passed verbatim to vLLM via
``response_format={"type": "json_schema", ...}`` so the xgrammar guided-decoding
backend constrains the model to exactly this shape. Even so, the client still
validates with this model before trusting the output.

Fields
------
- ``found``          — did the model locate the requested figure in the report?
- ``value``          — the figure in **raw single units** (single dollars / the
                       reporting currency; a per-share figure for EPS; a share
                       count for shares_outstanding). This is the only field the
                       scorer compares against ground truth.
- ``value_verbatim`` — the exact substring as printed in the report (e.g.
                       ``"$ (1,505)"`` or ``"3,400,300"``). Audit aid: lets us
                       check the model read the right cell before scaling.
- ``unit_scale``     — the scale the model applied to get ``value`` from
                       ``value_verbatim``. Audit aid: ``value`` should equal
                       ``value_verbatim`` parsed × the scale (×1 / ×1e3 / ×1e6 /
                       ×1e9), except for ``per_share`` where no scaling applies.
- ``page``           — the ``[Page N]`` marker of the page the figure was read
                       from. Provenance for spot-checking against the qrels.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


UnitScale = Literal[
    "units",       # already in single units; emit as printed
    "thousands",   # statement headed "in thousands"; value = printed × 1_000
    "millions",    # statement headed "in millions"; value = printed × 1_000_000
    "billions",    # statement headed "in billions"; value = printed × 1_000_000_000
    "per_share",   # EPS / per-share figure — NOT scaled
    "unknown",     # scale could not be determined
]


class NeedleAnswer(BaseModel):
    """The model's answer to a single (company, fiscal-year, KPI) lookup."""

    found: bool = Field(
        description=(
            "True if the requested metric for the requested fiscal year is "
            "stated in the report and you are reporting its value. False if the "
            "metric is genuinely not present for that year — in which case "
            "`value` must be null. Never set found=true with a guessed value."
        )
    )
    value: float | None = Field(
        default=None,
        description=(
            "The figure in RAW SINGLE UNITS. For monetary metrics this is single "
            "units of the reporting currency: apply the statement's scale "
            "('in thousands' -> x1000, 'in millions' -> x1e6, 'in billions' -> "
            "x1e9) before emitting. For EPS this is the per-share figure exactly "
            "as printed (do NOT scale). For shares_outstanding this is the share "
            "count in single shares. Use a leading minus sign for negatives "
            "(convert accounting parentheses '(1,505)' to -1505...). Null iff "
            "found is false."
        )
    )
    value_verbatim: str | None = Field(
        default=None,
        description=(
            "The exact figure as printed in the report, copied character-for-"
            "character (e.g. '$ (1,505)', '3,400,300', '1.08'). Do not scale or "
            "reformat it. Null iff found is false."
        )
    )
    unit_scale: UnitScale | None = Field(
        default=None,
        description=(
            "The scale you applied to turn value_verbatim into value: 'units', "
            "'thousands', 'millions', 'billions', 'per_share' (EPS — no scaling), "
            "or 'unknown'. Null iff found is false."
        )
    )
    page: int | None = Field(
        default=None,
        ge=1,
        description=(
            "The [Page N] marker of the page you read the figure from "
            "(1-indexed). Null if unknown or not found."
        ),
    )
