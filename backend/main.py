"""
main.py
=======
FastAPI application: serves the QEM Lab API (live experiment runs,
precomputed benchmark results, technique/benchmark catalogs, figure
images) and the static frontend.

Run with:
    cd qem_project
    uvicorn backend.main:app --host 0.0.0.0 --port 8000

Then open http://localhost:8000 in a browser.
"""

import csv
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend import service, llm_client, explain_service
from backend.rag.chat_service import answer_question
from backend.ml import predictor as ml_predictor

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(ROOT_DIR, "frontend-ng", "dist", "qem-frontend", "browser")
RESULTS_CSV = os.path.join(ROOT_DIR, "results", "data", "benchmark_results.csv")
FIGURES_DIR = os.path.join(ROOT_DIR, "results", "figures")

app = FastAPI(title="QEM Lab API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    technique: str
    noise_strength: float = 0.15
    shots: int = 1500
    benchmark: Optional[str] = None
    n_qubits: Optional[int] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = None


class ExplainRequest(BaseModel):
    result: dict


class PredictRequest(BaseModel):
    noise_strength: float = 0.15
    n_qubits: Optional[int] = None


@app.get("/api/health")
def health():
    return {"status": "ok", "llm_available": llm_client.is_available()}


@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        history = [h.dict() for h in req.history] if req.history else None
        return answer_question(req.message, history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")


@app.post("/api/explain")
def explain(req: ExplainRequest):
    try:
        return explain_service.explain_result(req.result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explain failed: {e}")


@app.post("/api/predict")
def predict(req: PredictRequest):
    try:
        return ml_predictor.rank_techniques(req.noise_strength, req.n_qubits)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/api/techniques")
def get_techniques():
    return service.get_techniques_with_compatibility()


@app.get("/api/benchmarks")
def get_benchmarks():
    return service.BENCHMARKS


@app.post("/api/run")
def run_experiment(req: RunRequest):
    try:
        return service.run_technique(
            req.technique, req.noise_strength, req.shots, req.benchmark, req.n_qubits
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {e}")


@app.get("/api/results")
def get_results(experiment: str = Query(default=None)):
    """Returns the precomputed benchmark_results.csv as JSON rows."""
    if not os.path.exists(RESULTS_CSV):
        raise HTTPException(status_code=404, detail="No precomputed results found. Run the benchmark suite first.")
    rows = []
    with open(RESULTS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if experiment and row.get("experiment") != experiment:
                continue
            rows.append(row)
    return {"count": len(rows), "rows": rows}


@app.get("/api/figures")
def list_figures():
    if not os.path.isdir(FIGURES_DIR):
        return {"figures": []}
    files = sorted(f for f in os.listdir(FIGURES_DIR) if f.lower().endswith(".png"))
    return {"figures": files}


@app.get("/api/figures/{filename}")
def get_figure(filename: str):
    path = os.path.join(FIGURES_DIR, filename)
    if not os.path.isfile(path) or not filename.lower().endswith(".png"):
        raise HTTPException(status_code=404, detail="Figure not found")
    return FileResponse(path, media_type="image/png")


@app.get("/{full_path:path}")
def serve_frontend(full_path: str):
    """Serves the Angular production build. Any path that matches a real
    built asset (JS/CSS/favicon) is returned directly; anything else
    (client-side routes like /live, /results) falls back to index.html so
    Angular's router can take over, including on a hard refresh."""
    requested = os.path.join(FRONTEND_DIR, full_path)
    if full_path and os.path.isfile(requested):
        return FileResponse(requested)
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.isfile(index_path):
        raise HTTPException(
            status_code=404,
            detail="Frontend build not found. Run 'npm run build' in frontend-ng/ first.",
        )
    return FileResponse(index_path)
