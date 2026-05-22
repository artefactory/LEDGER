# OCR Annotation Interface

Browser interface for comparing raw OCR page images with the corresponding Markdown page extracted by DeepSeekOCR. The app stores page-level annotations under `annotation_OCR/sessions/` so quality labels can later be joined to LLM benchmark outputs.

## Run

### Headless mode (recommended for multi-user)

Start the server with no session arguments — annotators create/resume sessions
from the browser landing page:

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
  --queue-mode table-candidates \
  --host 127.0.0.1 \
  --port 5050
```

For a small smoke run:

```bash
uv run python annotation_OCR/server.py \
  --session-name smoke \
  --annotator test \
  --queue-mode table-candidates \
  --limit-reports 2 \
  --limit 20 \
  --host 127.0.0.1 \
  --port 5050
```

Resume an existing session:

```bash
uv run python annotation_OCR/server.py --session-id SESSION_ID --host 127.0.0.1 --port 5050
```

SSH port forwarding from a laptop:

```bash
ssh -L 5050:127.0.0.1:5050 USER@SERVER
```

Then open `http://127.0.0.1:5050` locally.

The extracted-content pane shows inline OCR images by default. Turn off `Inline images` if you want a lighter placeholder-only Markdown preview.

## Data Sources

Defaults:

- OCR Markdown root: `DeepSeekOCR_Ardian_pruned_1k/`
- Raw image root: `/data/workspace/charles/pdf_ocr_deepseek/DeepSeekOCR_Ardian_raw_3kdocs/`

Each queued item maps one `.mmd` page split to the raw PNG with the same zero-based page index, for example page index `12` maps to `pages/page_0012.png`. The manifest records mapping warnings such as missing raw images or page-count mismatches.

## Queue Modes

- `table-candidates`: default. Keeps pages with table-like signals, dense numeric rows, financial statement headings, or KPI aliases.
- `all`: queues every page.
- `sample`: seeded random sample across all discovered pages. Use `--sample-size` and `--seed`.

Indexer smoke check:

```bash
uv run python annotation_OCR/ocr_index.py \
  --ocr-root DeepSeekOCR_Ardian_pruned_1k \
  --raw-root /data/workspace/charles/pdf_ocr_deepseek/DeepSeekOCR_Ardian_raw_3kdocs \
  --queue-mode table-candidates \
  --limit-reports 2 \
  --limit 20 \
  --check
```

## Keyboard

- `a`: mark Yes, save, advance
- `r`: mark No, save, advance
- `u`: mark Uncertain, save, advance
- `j` / right arrow: next page
- `k` / left arrow: previous page
- `+`, `-`, `0`: zoom controls
- `?`: shortcut dialog

Shortcuts are disabled while typing in notes or editing form controls.

## Outputs

Each session writes to `annotation_OCR/sessions/{session_id}/`:

- `metadata.json`: session name, annotator, configuration, counts, timestamps.
- `manifest.json`: queued pages and mapping diagnostics.
- `annotations.jsonl`: append-only event log, one saved annotation per line.
- `current_annotations.json`: latest annotation per item, written atomically.
- `summary.csv`: one row per queued page, including unreviewed pages.
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

## Downstream Joins

For page-level filtering, join annotation summaries on:

```text
exchange, ticker, year, page_index
```

For report-level benchmark filtering, aggregate page labels to:

```text
exchange, ticker, year
```

A conservative report-level rule is to exclude a report when any reviewed table-candidate page is `not_ok`, or when the share of `uncertain` pages exceeds a threshold chosen for the benchmark run.