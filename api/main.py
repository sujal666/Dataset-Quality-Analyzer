from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from data.loader import load_csv_bytes_dataset
from data.preprocess import preprocess_dataset
from reports.report_generator import analyze_hf, generate_report


app = FastAPI(title="Dataset Quality Analyzer", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


class HFAnalyzeRequest(BaseModel):
    dataset: str = Field(..., description="Hugging Face dataset name, e.g. 'imdb'")
    split: str = Field("train", description="Split name, e.g. 'train'")
    subset: Optional[str] = Field(None, description="Subset/config name if applicable")
    text_col: Optional[str] = Field(None, description="Text column name (optional)")
    label_col: Optional[str] = Field(None, description="Label column name (optional)")
    max_rows: Optional[int] = Field(None, description="Optional row limit")
    no_hf_models: bool = Field(False, description="Disable HF models (use lightweight fallbacks).")


@app.post("/analyze/hf")
def analyze_hf_endpoint(req: HFAnalyzeRequest) -> dict:
    try:
        return analyze_hf(
            req.dataset,
            split=req.split,
            subset=req.subset,
            text_col=req.text_col,
            label_col=req.label_col,
            max_rows=req.max_rows,
            prefer_hf_models=not req.no_hf_models,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)}) from e


@app.post("/analyze/csv")
async def analyze_csv_endpoint(
    file: UploadFile = File(...),
    text_col: Optional[str] = Form(None),
    label_col: Optional[str] = Form(None),
    no_hf_models: bool = Form(False),
) -> dict:
    content = await file.read()
    try:
        loaded = load_csv_bytes_dataset(content, filename=file.filename, text_col=text_col, label_col=label_col)
        pre = preprocess_dataset(loaded.df, text_col=loaded.text_col, label_col=loaded.label_col)
        report = generate_report(
            pre.df, text_col=loaded.text_col, label_col=loaded.label_col, prefer_hf_models=not no_hf_models
        )
        report["meta"].update(
            {
                "input": {"type": "csv_upload", "filename": file.filename, "content_type": file.content_type},
                "loader": loaded.meta,
                "preprocess": pre.meta,
            }
        )
        return report
    except ValueError as e:
        raise HTTPException(status_code=400, detail={"error": str(e)}) from e
