# OCR Annotation Interface

Browser interface for reviewing OCR table extraction quality. The app now
defaults to table-level items extracted from `*_det.mmd`, shows the isolated
HTML table in the extracted-content pane, and auto-centers the raw page image
on the detected table region while still allowing manual zoom-out for more
context.

Annotations are stored under `annotation_OCR/sessions/` so quality labels can
later be joined to downstream benchmark outputs.

## Run

### Headless mode (recommended for multi-user)

Start the server with no session arguments — annotators create/resume sessions
from the browser landing page. If `annotation_OCR/manifests/tables_5000.json`
exists, the server uses it automatically for fast session creation. Otherwise
it falls back to building a sampled table queue directly from the OCR corpus.

```bash
uv run python annotation_OCR/server.py --host 0.0.0.0 --port 5050
```

Then open `http://HOST:5050`. The landing page lets each user enter their name,
create a new session, or resume an existing one. No CLI or Python knowledge
needed on the annotator side.

### Pre-created session (single-user / scripted)

From the repository root:

```bash
uv run python annotation_OCR/server.py \
  --session-name "table QA smoke" \
  --annotator "your-name" \
  --queue-mode tables \
  --sample-size 100 \
  --host 127.0.0.1 \
  --port 5050
```

For a small smoke run:

```bash
uv run python annotation_OCR/server.py \
  --session-name smoke \
  --annotator test \
  --queue-mode tables \
  --sample-size 20 \
  --limit-reports 2 \
  --host 127.0.0.1 \
  --port 5050
```

To force the server to use an explicit precomputed manifest:

```bash
uv run python annotation_OCR/server.py \
  --manifest-path annotation_OCR/manifests/tables_5000.json \
  --host 127.0.0.1 \
  --port 5050
```

To use precomputed study-session bundles for a paper annotation round:

```bash
uv run python annotation_OCR/server.py \
  --study-bundle annotation_OCR/manifests/study_sessions_15.json \
  --host 127.0.0.1 \
  --port 5050
```

Each new session created from the landing page then receives the next fixed
session queue from that bundle, so the progress bar tracks a real per-annotator
target rather than the whole table pool.

Resume an existing session:

```bash
uv run python annotation_OCR/server.py --session-id SESSION_ID --host 127.0.0.1 --port 5050
```

SSH port forwarding from a laptop:

```bash
ssh -L 5050:127.0.0.1:5050 USER@SERVER
```

Then open `http://127.0.0.1:5050` locally.

For table sessions, the extracted-content pane shows only the isolated table and
the raw-image pane auto-refocuses on the detected bounding box. Use `Refocus`
or press `F` to jump back to the table after manual exploration.

## Precompute A Reusable 5,000-Table Manifest

Build the reusable subset once offline:

```bash
mkdir -p annotation_OCR/manifests

uv run python annotation_OCR/ocr_index.py \
  --queue-mode tables \
  --sample-size 5000 \
  --seed 42 \
  --output annotation_OCR/manifests/tables_5000.json
```

That manifest can then be reused by the server so new annotation sessions do
not need to rescan the OCR corpus.

## Build Study Session Bundles

For hybrid annotation rounds, build one bundle for each possible annotator
count. The generated bundles already keep each session inside the target range
of 120 to 140 items:

```bash
uv run python annotation_OCR/study_sessions.py \
  --source-manifest annotation_OCR/manifests/tables_5000.json \
  --output-dir annotation_OCR/manifests \
  --annotators 14 15 16 \
  --seed 42
```

This writes:

- `annotation_OCR/manifests/study_sessions_14.json`
- `annotation_OCR/manifests/study_sessions_15.json`
- `annotation_OCR/manifests/study_sessions_16.json`

The 15- and 16-annotator bundles use 1500 unique tables with 300 triple-coded
agreement tables. The 14-annotator bundle lowers the agreement subset to 220 so
all session quotas still stay within the 120 to 140 target range.

## Compute Agreement After Annotation

After the study round, compute overlap agreement plus accept/reject ratios with:

```bash
uv run python annotation_OCR/study_agreement.py \
  --study-bundle annotation_OCR/manifests/study_sessions_15.json
```

By default this writes analysis artifacts under:

- `annotation_OCR/sessions/study_analysis/study_sessions_15/summary.md`
- `annotation_OCR/sessions/study_analysis/study_sessions_15/summary.json`
- `annotation_OCR/sessions/study_analysis/study_sessions_15/session_metrics.csv`
- `annotation_OCR/sessions/study_analysis/study_sessions_15/item_metrics.csv`

The script auto-discovers sessions created from that bundle via their stored
`study_bundle_path` and `study_slot`. It reports exact agreement, pairwise
agreement, Fleiss' kappa, and accept/reject ratios both at the raw vote level
and at the final table-decision level.

## Data Sources

Defaults:

- OCR Markdown root: `DeepSeekOCR_Ardian_pruned_1k/`
- Raw image root: `/data/workspace/charles/pdf_ocr_deepseek/DeepSeekOCR_Ardian_raw_3kdocs/`
- Default reusable manifest path: `annotation_OCR/manifests/tables_5000.json`

Each queued table item maps back to the raw PNG page with the same zero-based
page index, for example page index `12` maps to `pages/page_0012.png`. Table
items carry the `_det.mmd` bounding box used by the UI to center the preview.
The manifest records mapping warnings such as missing raw images or page-count
mismatches.

## Queue Modes

- `tables`: default. Queues table-level items from `*_det.mmd`. Use `--sample-size` for deterministic random sampling.
- `table-candidates`: legacy page-level mode. Keeps pages with table-like signals, dense numeric rows, financial statement headings, or KPI aliases.
- `all`: legacy page-level mode that queues every page.
- `sample`: legacy seeded random sample across all discovered pages.

Indexer smoke check:

```bash
uv run python annotation_OCR/ocr_index.py \
  --ocr-root DeepSeekOCR_Ardian_pruned_1k \
  --raw-root /data/workspace/charles/pdf_ocr_deepseek/DeepSeekOCR_Ardian_raw_3kdocs \
  --queue-mode tables \
  --sample-size 20 \
  --limit-reports 2 \
  --check
```

## Keyboard

- `a`: mark Yes, save, advance
- `r`: mark No, save, advance
- `u`: mark Uncertain, save, advance
- `j` / right arrow: next page
- `k` / left arrow: previous page
- `+`, `-`, `0`: zoom / reset
- `f`: refocus on the detected table
- `?`: shortcut dialog

Shortcuts are disabled while typing in notes or editing form controls.

## Outputs

Each session writes to `annotation_OCR/sessions/{session_id}/`:

- `metadata.json`: session name, annotator, configuration, counts, timestamps.
- `manifest.json`: queued items and mapping diagnostics.
- `annotations.jsonl`: append-only event log, one saved annotation per line.
- `current_annotations.json`: latest annotation per item, written atomically.
- `summary.csv`: one row per queued item, including unreviewed items.
- `summary.md`: status-count overview.

Regenerate summaries:

```bash
uv run python annotation_OCR/summarize.py --session-id SESSION_ID
uv run python annotation_OCR/summarize.py --all
```

## Annotation Schema

Primary fields:

- `overall_status`: `ok`, `not_ok`, `uncertain`, or `unreviewed`
- `notes`: optional free text

Identity fields include `industry_slug`, `report_name`, `exchange`, `ticker`, `year`, `page_index`, `page_number`, `mmd_path`, `raw_png_path`, and `page_text_sha256`.

For table sessions, summary rows also include `item_kind`, `table_index`,
`table_row_count`, `table_col_count`, `det_mmd_path`, and `focus_bbox`.

## Downstream Joins

For table-level filtering, join annotation summaries on:

```text
exchange, ticker, year, page_index, table_index
```

For report-level benchmark filtering, aggregate page labels to:

```text
exchange, ticker, year
```

A conservative report-level rule is to exclude a report when any reviewed table
item is `not_ok`, or when the share of `uncertain` table items exceeds a
threshold chosen for the benchmark run.