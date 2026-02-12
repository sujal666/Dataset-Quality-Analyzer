# Dataset Quality Analyzer (Hugging Face Powered)

Tooling to inspect a dataset *before* training by flagging duplicates, label issues, toxicity, domain mixing, and leakage; then producing a quality score (0–100) and a JSON report.

## Quickstart

### 1) Python env

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

### 2) Run backend (FastAPI)

```bash
uvicorn api.main:app --reload --port 8000
```

### 3) Run frontend

```bash
cd frontend
npm install
npm run dev
```

Set `NEXT_PUBLIC_API_URL` in `frontend/.env.local` if your API isn’t on `http://localhost:8000`.

## CLI (local report)

```bash
python -m reports.report_generator --csv data/sample.csv --out reports/sample.report.json
```

If you want a fast run without downloading models:

```bash
python -m reports.report_generator --csv data/sample.csv --out reports/sample.report.json --no-hf-models
```

## Notes

- Default columns are normalized to `text` and `label` (Phase 1).
- Models download on first run (embedding + toxicity) and are cached by Hugging Face.
