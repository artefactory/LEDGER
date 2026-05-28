# Table Manifests

Place reusable sampled table manifests here.

Recommended default:

```bash
uv run python annotation_OCR/ocr_index.py \
  --queue-mode tables \
  --sample-size 5000 \
  --seed 42 \
  --output annotation_OCR/manifests/tables_5000.json
```

When `tables_5000.json` exists, `annotation_OCR/server.py` will use it by default for new sessions.

## Study Session Bundles

For paper annotation rounds, also build the headcount-specific session bundles:

```bash
uv run python annotation_OCR/study_sessions.py \
  --source-manifest annotation_OCR/manifests/tables_5000.json \
  --output-dir annotation_OCR/manifests \
  --annotators 14 15 16 \
  --seed 42
```

This creates:

- `study_sessions_14.json`
- `study_sessions_15.json`
- `study_sessions_16.json`

Use the bundle matching the final annotator count when starting the server:

```bash
uv run python annotation_OCR/server.py \
  --study-bundle annotation_OCR/manifests/study_sessions_15.json
```

Why the 14-annotator bundle differs:

- `1500 unique + 300 triple-coded` requires `2100` total annotations.
- That fits 15 or 16 annotators while keeping each session in the `120–140` range.
- For 14 annotators, the bundle uses `220` agreement tables instead, for `1940` total annotations and per-session targets of `138–139`.
