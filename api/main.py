from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from nlp_engine.orchestration.orchestrator import NLPOrchestrator

app = FastAPI(title="NLP Intelligence Engine", version="1.0.0")
engine = NLPOrchestrator()


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1)


class SimilarityRequest(BaseModel):
    text: str = Field(..., min_length=1)
    candidate_texts: list[str] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    documents: list[str] = Field(default_factory=list)


@app.get("/health")
def health_check() -> dict[str, Any]:
    return {"status": "ok", "service": "nlp-intelligence-engine"}


@app.post("/api/v1/analyze")
def analyze(request: TextRequest) -> dict[str, Any]:
    return engine.analyze(request.text, task="classification")


@app.post("/api/v1/classify")
def classify(request: TextRequest) -> dict[str, Any]:
    return engine.predict(request.text, task="classification")


@app.post("/api/v1/sentiment")
def sentiment(request: TextRequest) -> dict[str, Any]:
    return engine.predict(request.text, task="sentiment")


@app.post("/api/v1/ner")
def ner(request: TextRequest) -> dict[str, Any]:
    return engine.analyze(request.text, task="ner")


@app.post("/api/v1/keywords")
def keywords(request: TextRequest) -> dict[str, Any]:
    return engine.analyze(request.text, task="keywords")


@app.post("/api/v1/similarity")
def similarity(request: SimilarityRequest) -> dict[str, Any]:
    return {"results": engine.similarity(request.text, request.candidate_texts)}


@app.post("/api/v1/search")
def search(request: SearchRequest) -> dict[str, Any]:
    results = engine.similarity(request.query, request.documents)
    return {"query": request.query, "results": results}


@app.get("/api/v1/models")
def models() -> dict[str, Any]:
    return {"models": [{"name": "tfidf-logistic-regression", "type": "classical"}]}


@app.get("/api/v1/experiments")
def experiments() -> dict[str, Any]:
    return {"experiments": [{"name": "baseline-demo", "model": "tfidf-logistic-regression", "status": "completed"}]}
