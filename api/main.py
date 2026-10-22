"""智慧司法判決預測與可解釋案例分析系統 — FastAPI 後端。

整合：
1. 判決要點規則擷取（parsing.extract_rules）
2. 訓練模型預測與 SHAP 可解釋分析 + 參考專案 HAN 句子注意力（model.predict）
3. 歷史相似判決檢索（retrieval.index）
4. 本地端 Gemma-4-E4B（Ollama / llama-server）RAG 法律分析生成（支援串流與同步模式）

啟動方式：
    uvicorn main:app --reload
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from llm.client import chat, stream_chat  # noqa: E402
from llm.prompts import ANALYZE_SYSTEM, build_rag_prompt  # noqa: E402
from model.predict import VerdictPredictor  # noqa: E402
from retrieval.index import CaseRetriever  # noqa: E402

app = FastAPI(title="台灣民事車禍判決預測與可解釋案例分析系統")
WEB_DIR = config.ROOT / "web"

_predictor: VerdictPredictor | None = None
_retriever: CaseRetriever | None = None


def get_predictor() -> VerdictPredictor:
    global _predictor
    if _predictor is None:
        _predictor = VerdictPredictor()
    return _predictor


def get_retriever() -> CaseRetriever:
    global _retriever
    if _retriever is None:
        _retriever = CaseRetriever()
    return _retriever


class PredictRequest(BaseModel):
    text: str = Field(..., description="使用者輸入的車禍案件事實描述")
    # 結構化傷勢與請求明細（可選填；未填時自動由 text 規則擷取）
    injury_level: int | None = Field(default=None, ge=0, le=4, description="0=無傷,1=輕傷,2=骨折中傷,3=重傷,4=死亡")
    injury_terms: list[str] | str | None = None
    hospital_days: int | None = Field(default=None, ge=0)
    surgery: bool | None = None
    disability: bool | None = None
    has_criminal: bool | None = None
    has_appraisal: bool | None = None
    claims_contributory: bool | None = None
    v_red_light: bool | None = None
    v_drunk: bool | None = None
    v_speeding: bool | None = None
    v_no_yield: bool | None = None
    claim_medical: int | None = Field(default=None, ge=0)
    claim_care: int | None = Field(default=None, ge=0)
    claim_work_loss: int | None = Field(default=None, ge=0)
    claim_labor_loss: int | None = Field(default=None, ge=0)
    claim_vehicle: int | None = Field(default=None, ge=0)
    claim_consolation: int | None = Field(default=None, ge=0)
    claim_amount: int | None = Field(default=None, ge=0)
    sync_llm: bool = Field(default=False, description="是否於 /api/predict 同步等待 Gemma 生成（預設前端改用串流端點）")


class Factor(BaseModel):
    feature: str = ""
    name: str = ""
    term: str
    effect: int
    weight: float


class SentenceWeight(BaseModel):
    sentence: str
    score: float


class SimilarCase(BaseModel):
    case_id: str
    similarity: float
    verdict: str
    claim_amount: int = 0
    awarded_amount: int = 0
    plaintiff_fault: float = 0.0
    summary: str


class PredictResponse(BaseModel):
    label: str
    confidence: float
    class_probs: dict[str, float]
    predicted_ratio: float
    predicted_award: int
    predicted_fault: float
    predicted_consolation: int
    extracted: dict
    reason: str
    factors: list[Factor]
    sentence_scores: list[SentenceWeight]
    similar: list[SimilarCase]
    metrics: dict
    elapsed_ms: int
    source: str
    llm_model: str
    error: str | None = None


def _overrides_from_req(req: PredictRequest) -> dict:
    keys = (
        "injury_level", "injury_terms", "hospital_days", "surgery", "disability",
        "has_criminal", "has_appraisal", "claims_contributory",
        "v_red_light", "v_drunk", "v_speeding", "v_no_yield",
        "claim_medical", "claim_care", "claim_work_loss", "claim_labor_loss",
        "claim_vehicle", "claim_consolation", "claim_amount",
    )
    return {k: getattr(req, k) for k in keys if getattr(req, k) is not None}


def _rule_summary(ml: dict) -> str:
    st = ml["structured"]
    return (
        f"依訓練模型（LightGBM）比照過去民事車禍判決分析：本案傷勢屬於「{st.get('injury_level_name')}」，"
        f"原告請求總額 {int(st.get('claim_amount') or 0):,} 元。"
        f"模型預測判決結果為「{ml['label']}」（信心度 {ml['confidence']:.0%}），"
        f"預估原告與有過失比例約 {ml['predicted_fault']:.0%}，"
        f"預估法院核定精神慰撫金約 {ml['predicted_consolation']:,} 元，"
        f"預估總判賠金額約新臺幣 {ml['predicted_award']:,} 元（占請求金額 {ml['predicted_ratio']:.0%}）。"
    )


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    overrides = _overrides_from_req(req)
    pred = get_predictor().predict_one(req.text, overrides)
    sim_list = get_retriever().search(req.text, overrides, top_k=5)

    reason = _rule_summary(pred)
    source = "ml"
    error = None

    if req.sync_llm:
        prompt = build_rag_prompt(req.text, pred["structured"], pred, pred["factors"], sim_list)
        try:
            reason = chat(
                [{"role": "system", "content": ANALYZE_SYSTEM}, {"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=420,
            )
            source = "ml+llm"
        except Exception as e:
            error = f"Gemma 語言模型呼叫失敗（{type(e).__name__}: {e}），已顯示訓練模型結構化分析結果。"

    st = pred["structured"]
    extracted_view = {
        "injury_level": st.get("injury_level", 0),
        "injury_level_name": st.get("injury_level_name", ""),
        "injury_terms": st.get("injury_terms", []),
        "hospital_days": st.get("hospital_days", 0),
        "surgery": bool(st.get("surgery")),
        "disability": bool(st.get("disability")),
        "claim_amount": int(st.get("claim_amount") or 0),
        "claim_medical": int(st.get("claim_medical") or 0),
        "claim_care": int(st.get("claim_care") or 0),
        "claim_work_loss": int(st.get("claim_work_loss") or 0),
        "claim_vehicle": int(st.get("claim_vehicle") or 0),
        "claim_consolation": int(st.get("claim_consolation") or 0),
    }

    ms = int((time.perf_counter() - t0) * 1000)
    return PredictResponse(
        label=pred["label"],
        confidence=pred["confidence"],
        class_probs=pred["class_probs"],
        predicted_ratio=pred["predicted_ratio"],
        predicted_award=pred["predicted_award"],
        predicted_fault=pred["predicted_fault"],
        predicted_consolation=pred["predicted_consolation"],
        extracted=extracted_view,
        reason=reason,
        factors=[Factor(**f) for f in pred["factors"]],
        sentence_scores=[SentenceWeight(sentence=s["sentence"], score=s["score"]) for s in pred["sentence_scores"]],
        similar=[SimilarCase(**c) for c in sim_list],
        metrics=pred["metrics"],
        elapsed_ms=ms,
        source=source,
        llm_model=config.LLM_MODEL,
        error=error,
    )


@app.post("/api/explain/stream")
def explain_stream(req: PredictRequest):
    """SSE 串流端點：將訓練模型預測結果與相似判決餵給 Gemma-4-E4B，即時串流生成白話法律分析。"""
    overrides = _overrides_from_req(req)
    pred = get_predictor().predict_one(req.text, overrides)
    sim_list = get_retriever().search(req.text, overrides, top_k=4)
    prompt = build_rag_prompt(req.text, pred["structured"], pred, pred["factors"], sim_list)
    messages = [
        {"role": "system", "content": ANALYZE_SYSTEM},
        {"role": "user", "content": prompt},
    ]

    def event_gen():
        try:
            for piece in stream_chat(messages, temperature=0.2, max_tokens=420):
                yield f"data: {json.dumps({'delta': piece}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            err_msg = f"語言模型（{config.LLM_MODEL}）連線失敗：{type(e).__name__}: {e}"
            yield f"data: {json.dumps({'error': err_msg}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/api/reload")
def reload_models():
    """熱更新：當匯入新判決並重訓模型與索引後，不需重啟伺服器即可重新載入。"""
    global _predictor, _retriever
    _predictor = VerdictPredictor()
    _retriever = CaseRetriever()
    return {
        "reloaded": True,
        "indexed_cases": len(_retriever.cases),
        "metrics": _predictor.metrics,
    }


@app.get("/api/status")
def status():
    p = get_predictor()
    r = get_retriever()
    return {
        "model_ready": True,
        "indexed_cases": len(r.cases),
        "metrics": p.metrics,
        "llm_url": config.LLM_URL,
        "llm_model": config.LLM_MODEL,
    }


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/{asset_name}.js")
def serve_js(asset_name: str):
    return FileResponse(WEB_DIR / f"{asset_name}.js", media_type="application/javascript")


@app.get("/{asset_name}.json")
def serve_json(asset_name: str):
    return FileResponse(WEB_DIR / f"{asset_name}.json", media_type="application/json")
