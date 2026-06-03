# LLM Benchmark Analysis: Multi-KPI vs Needle-in-a-Haystack

## Overview

Two complementary benchmarks evaluate LLMs on KPI extraction from OCR'd annual reports (494 reports, 2017–2022):

- **Multi-KPI extraction** — the model must extract ~30 KPIs from each report in a single pass (13,455 ground-truth cells).
- **Needle-in-a-haystack** — the model is asked for a single, specific KPI value per query (10,000 queries, one KPI at a time).

Both benchmarks share the same test-set reports and ±1% match tolerance.

## Results at a glance

| Model | Multi-KPI Recall | Multi-KPI Precision | Needle Accuracy | Needle Precision |
|---|---|---|---|---|
| Qwen3.6-27B | **79.3** | **88.0** | **93.6** | **95.8** |
| gpt-oss-20b | 68.1 | 76.0 | 87.9 | 89.4 |
| Nemotron-3-Nano-30B | 67.6 | 80.5 | 15.8 | 16.1 |
| Ministral-3-14B | 42.6 | 44.8 | 90.7 | 91.4 |

## Key observations

### Qwen3.6-27B leads on both tasks

Qwen is the most balanced model: it ranks first on the single-value lookup (93.6% accuracy) and maintains a strong lead on the harder multi-KPI extraction task (79.3% recall, 88.0% precision). Its precision gap over the next-best model (gpt-oss-20b) widens from ~6pp in needle to ~12pp in multi-KPI, suggesting more reliable structured output.

### Ministral-3-14B: strong on needle, weak on multi-KPI

The most dramatic ranking shift. Ministral ranks second on needle-in-a-haystack (90.7%) but drops to last on multi-KPI extraction (42.6% recall, 44.8% precision). This suggests the model can locate a value when prompted for a single fact, but struggles to reliably produce structured, multi-field JSON output — it generates the most predictions (14,178) yet matches the fewest (5,732), indicating a high hallucination rate when tasked with extracting many KPIs at once.

### Nemotron-3-Nano-30B: opposite pattern

Nemotron shows the inverse behaviour: it collapses on needle-in-a-haystack (15.8% accuracy, overwhelmed by 1000× scale errors) but performs competitively on multi-KPI extraction (67.6% recall, 80.5% precision). When extracting multiple KPIs, the structured schema and guided decoding likely constrain the model's output format, preventing the unit-scaling failures that dominate its needle performance. Its multi-KPI weakness is coverage (2,156 missing — the highest miss count), not precision.

### gpt-oss-20b: consistent mid-tier

The most stable model across both tasks: third on needle (87.9%) and second on multi-KPI (68.1% recall). It trades precision for coverage — it produces fewer misses than Nemotron (1,384 vs 2,156) but more wrong answers (2,903 vs 2,203), and its extra-prediction count (1,272) is the second highest.

## Implications

- **Single-value lookup is not a proxy for multi-KPI extraction.** Needle-in-a-haystack accuracy does not predict multi-KPI ranking (Ministral and Nemotron swap positions). Benchmarks that only test one-at-a-time queries may overstate a model's readiness for production extraction.
- **Structured output discipline matters.** The multi-KPI task penalises models that hallucinate extra KPIs or mis-map values to the wrong field — failure modes invisible in the needle benchmark.
- **Scale errors are model-specific, not task-specific.** Nemotron's 1000× scaling bug dominates needle performance but is partially masked in multi-KPI by guided decoding constraints. Fixing this single failure mode would likely boost Nemotron's needle accuracy to competitive levels.
