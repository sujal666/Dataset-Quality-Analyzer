# Dataset Quality Analyzer (Hugging Face Powered)

Tooling to inspect a dataset before training by flagging duplicates, label issues, toxicity, domain mixing, and leakage, then producing a quality score (0-100) and JSON report.

## Quickstart

### 1) Python env

```bash
python -m venv .venv
.venv\Scripts\activate
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

Set `NEXT_PUBLIC_API_URL` in `frontend/.env.local` if your API is not `http://localhost:8000`.

## CLI (local report)

```bash
python -m reports.report_generator --csv data/sample.csv --out reports/sample.report.json
```

Fast mode without downloading HF models:

```bash
python -m reports.report_generator --csv data/sample.csv --out reports/sample.report.json --no-hf-models
```

## What changed in this refactor

- Modality-aware routing (`text`, `tabular`, `mixed`) now runs only relevant analyzers.
- Tabular datasets use key-based duplicate logic instead of semantic embedding duplicates.
- Domain-mixing logic is split by modality:
  - text: embedding clustering
  - tabular: cluster/category entropy alignment
- Label outlier detection now uses class-distance Z-scores.
- Embeddings are cached on disk (`embeddings/cache`) using dataset signature keys.
- Issue output includes plain explanation, why-it-matters, and concrete row examples.

