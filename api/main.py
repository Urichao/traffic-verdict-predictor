"""智慧司法判決預測與可解釋案例分析系統 — FastAPI 後端。

整合：
1. 案件文字檔上傳解析（.txt / .docx / .pdf）與自動擷取預填（POST /api/upload-text）
2. 台灣民事法院標準「車輛修復零件折舊」試算（api.depreciation）
3. 判決要點規則擷取（parsing.extract_rules）
4. 訓練模型預測與 SHAP 可解釋分析 + 參考專案 HAN 句子注意力（model.predict）
5. 歷史相似判決檢索（retrieval.index）
6. 本地端 Gemma-4-E4B（Ollama / llama-server）RAG 法律分析生成（支援串流與同步模式）

啟動方式：
    uvicorn main:app --reload
"""
from __future__ import annotations

import io
import json
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from api.depreciation import calculate_depreciation, extract_vehicle_parts_labor  # noqa: E402
from llm.client import chat, stream_chat  # noqa: E402
from llm.prompts import ANALYZE_SYSTEM, build_rag_prompt  # noqa: E402
from model.predict import VerdictPredictor  # noqa: E402
from parsing.extract_rules import INJURY_LEVEL_NAMES, extract_features  # noqa: E402
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


class DepreciationRequest(BaseModel):
    vehicle_type: str = Field(default="car", description="car | motorcycle | commercial_car | truck")
    vehicle_age_years: float = Field(default=0.0, ge=0.0, le=50.0)
    parts_cost: int = Field(default=0, ge=0)
    labor_cost: int = Field(default=0, ge=0)
    method: str = Field(default="declining_balance", description="declining_balance | straight_line")


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
    # 基本與擴充請求金額欄位
    claim_medical: int | None = Field(default=None, ge=0)
    claim_future_medical: int | None = Field(default=None, ge=0)
    claim_care: int | None = Field(default=None, ge=0)
    claim_transport: int | None = Field(default=None, ge=0)
    claim_work_loss: int | None = Field(default=None, ge=0)
    claim_labor_loss: int | None = Field(default=None, ge=0)
    claim_vehicle: int | None = Field(default=None, ge=0)
    claim_funeral: int | None = Field(default=None, ge=0)
    claim_support: int | None = Field(default=None, ge=0)
    claim_consolation: int | None = Field(default=None, ge=0)
    claim_amount: int | None = Field(default=None, ge=0)
    insurance_deduction: int | None = Field(default=None, ge=0, description="已領強制險／任意險理賠扣除額")
    # 車輛修復折舊試算欄位
    vehicle_type: str | None = Field(default=None, description="car | motorcycle | commercial_car | truck")
    vehicle_age_years: float | None = Field(default=None, ge=0.0, le=50.0)
    depreciation_method: str | None = Field(default="declining_balance")
    claim_vehicle_parts: int | None = Field(default=None, ge=0)
    claim_vehicle_labor: int | None = Field(default=None, ge=0)
    # 當事人學經歷與休養情形（對齊 parsing/extract_rules.py 新欄位）
    plaintiff_education: str | None = None
    plaintiff_occupation: str | None = None
    plaintiff_monthly_income: int | None = Field(default=None, ge=0)
    plaintiff_rest_months: float | None = Field(default=None, ge=0.0)
    plaintiff_labor_loss_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
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
    depreciation: dict | None = None
    insurance_deduction: int = 0
    net_award_after_insurance: int = 0
    reason: str
    factors: list[Factor]
    sentence_scores: list[SentenceWeight]
    similar: list[SimilarCase]
    metrics: dict
    elapsed_ms: int
    source: str
    llm_model: str
    error: str | None = None


def _decode_text_bytes(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp950", "big5", "utf-16"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _extract_docx_bytes(raw: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        xml_bytes = zf.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paras = []
    for p in root.findall(".//w:p", ns):
        texts = [t.text for t in p.findall(".//w:t", ns) if t.text]
        line = "".join(texts).strip()
        if line:
            paras.append(line)
    return "\n".join(paras)


def _extract_pdf_bytes(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise HTTPException(status_code=400, detail="伺服器尚未安裝 pypdf，暫時無法解析 PDF 檔。") from exc
    reader = PdfReader(io.BytesIO(raw))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n".join(p for p in pages if p)


def _overrides_from_req(req: PredictRequest) -> dict[str, Any]:
    keys = (
        "injury_level", "injury_terms", "hospital_days", "surgery", "disability",
        "has_criminal", "has_appraisal", "claims_contributory",
        "v_red_light", "v_drunk", "v_speeding", "v_no_yield",
        "claim_medical", "claim_future_medical", "claim_care", "claim_transport",
        "claim_work_loss", "claim_labor_loss", "claim_vehicle",
        "claim_funeral", "claim_support", "claim_consolation", "claim_amount",
    )
    ov: dict[str, Any] = {k: getattr(req, k) for k in keys if getattr(req, k) is not None}
    # 若使用者填寫了修車零件與工資，但未填修車總額，自動加總
    if ov.get("claim_vehicle") is None and ((req.claim_vehicle_parts or 0) > 0 or (req.claim_vehicle_labor or 0) > 0):
        ov["claim_vehicle"] = int(req.claim_vehicle_parts or 0) + int(req.claim_vehicle_labor or 0)
    if (req.insurance_deduction or 0) > 0:
        ov["insurance_paid"] = True
    return ov


def _compute_depreciation_for_req(req: PredictRequest) -> dict[str, Any] | None:
    parts = int(req.claim_vehicle_parts or 0)
    labor = int(req.claim_vehicle_labor or 0)
    if parts <= 0 and labor <= 0:
        return None
    return calculate_depreciation(
        vehicle_type=req.vehicle_type or "car",
        vehicle_age_years=float(req.vehicle_age_years or 0.0),
        parts_cost=parts,
        labor_cost=labor,
        method=req.depreciation_method or "declining_balance",
    )


def _build_enriched_prompt(
    req: PredictRequest,
    pred: dict[str, Any],
    sim_list: list[dict[str, Any]],
    depreciation: dict[str, Any] | None,
) -> str:
    base = build_rag_prompt(req.text, pred["structured"], pred, pred["factors"], sim_list)
    extra_lines: list[str] = []
    if depreciation and depreciation.get("applied"):
        extra_lines.append(
            f"【車損折舊試算】{depreciation['formula_note']}"
            f"原請求修車 {depreciation['original_total']:,} 元（零件 {depreciation['parts_original']:,}＋工資 {depreciation['labor_original']:,}），"
            f"折舊後預估核准 {depreciation['approved_total']:,} 元（扣除折舊 {depreciation['deducted_amount']:,} 元）。"
        )
    if (req.insurance_deduction or 0) > 0:
        extra_lines.append(f"【保險理賠扣除】已領強制險／保險理賠 {int(req.insurance_deduction):,} 元（依法應自總判賠金額中扣除）。")

    st = pred["structured"]
    edu = req.plaintiff_education or st.get("plaintiff_education")
    occ = req.plaintiff_occupation or st.get("plaintiff_occupation")
    inc = req.plaintiff_monthly_income if req.plaintiff_monthly_income is not None else st.get("plaintiff_monthly_income")
    rest = req.plaintiff_rest_months if req.plaintiff_rest_months is not None else st.get("plaintiff_rest_months")
    bg_parts = []
    if edu:
        bg_parts.append(f"學歷：{edu}")
    if occ:
        bg_parts.append(f"職業：{occ}")
    if inc:
        bg_parts.append(f"月收入：{int(inc):,}元")
    if rest:
        bg_parts.append(f"醫囑休養：{rest}個月")
    if bg_parts:
        extra_lines.append("【原告背景與休養情形（供慰撫金審酌）】" + "、".join(bg_parts))

    if not extra_lines:
        return base
    return base + "\n" + "\n".join(extra_lines)


def _rule_summary(ml: dict, depreciation: dict | None = None, ins_deduct: int = 0) -> str:
    st = ml["structured"]
    dep_txt = ""
    if depreciation and depreciation.get("applied"):
        dep_txt = (
            f"另依{depreciation['method_name']}試算車輛零件折舊，修車費由 {depreciation['original_total']:,} 元"
            f"核減為 {depreciation['approved_total']:,} 元（扣除折舊 {depreciation['deducted_amount']:,} 元）。"
        )
    ins_txt = ""
    if ins_deduct > 0:
        net = max(0, int(ml["predicted_award"]) - ins_deduct)
        ins_txt = f"若再扣除已領強制險／保險理賠 {ins_deduct:,} 元，被告實際尚須給付淨額約 {net:,} 元。"
    return (
        f"依訓練模型（LightGBM）比照過去民事車禍判決分析：本案傷勢屬於「{st.get('injury_level_name')}」，"
        f"原告請求總額 {int(st.get('claim_amount') or 0):,} 元。"
        f"模型預測判決結果為「{ml['label']}」（信心度 {ml['confidence']:.0%}），"
        f"預估原告與有過失比例約 {ml['predicted_fault']:.0%}，"
        f"預估法院核定精神慰撫金約 {ml['predicted_consolation']:,} 元，"
        f"預估總判賠金額約新臺幣 {ml['predicted_award']:,} 元（占請求金額 {ml['predicted_ratio']:.0%}）。"
        f"{dep_txt}{ins_txt}"
    )


@app.post("/api/upload-text")
async def upload_text(file: UploadFile = File(...)):
    """上傳案情文字檔（.txt / .docx / .pdf / .md），回傳純文字與自動擷取之結構化欄位供前端預填。"""
    filename = file.filename or "case.txt"
    suffix = Path(filename).suffix.lower()
    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="檔案過大，請上傳 10 MB 以下之文字或文件檔。")

    if suffix in (".txt", ".md", ".csv", ""):
        text = _decode_text_bytes(raw)
    elif suffix == ".docx":
        try:
            text = _extract_docx_bytes(raw)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Word (.docx) 檔案解析失敗：{exc}") from exc
    elif suffix == ".pdf":
        try:
            text = _extract_pdf_bytes(raw)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"PDF 檔案解析失敗：{exc}") from exc
    else:
        raise HTTPException(status_code=400, detail=f"不支援的檔案格式（{suffix}），請上傳 .txt、.docx 或 .pdf。")

    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="檔案中未讀取到任何文字內容。")

    feats = extract_features(text)
    veh_parts = extract_vehicle_parts_labor(text)
    extracted = {
        "injury_level": feats.get("injury_level", 0),
        "injury_level_name": INJURY_LEVEL_NAMES.get(int(feats.get("injury_level") or 0), ""),
        "injury_terms": feats.get("injury_terms", []),
        "hospital_days": feats.get("hospital_days", 0),
        "surgery": bool(feats.get("surgery")),
        "disability": bool(feats.get("disability")),
        "has_criminal": bool(feats.get("has_criminal")),
        "has_appraisal": bool(feats.get("has_appraisal")),
        "claims_contributory": bool(feats.get("claims_contributory")),
        "v_red_light": bool(feats.get("v_red_light")),
        "v_drunk": bool(feats.get("v_drunk")),
        "v_no_yield": bool(feats.get("v_no_yield")),
        "claim_medical": int(feats.get("claim_medical") or 0),
        "claim_future_medical": int(feats.get("claim_future_medical") or 0),
        "claim_care": int(feats.get("claim_care") or 0),
        "claim_transport": int(feats.get("claim_transport") or 0),
        "claim_work_loss": int(feats.get("claim_work_loss") or 0),
        "claim_labor_loss": int(feats.get("claim_labor_loss") or 0),
        "claim_vehicle": int(feats.get("claim_vehicle") or 0),
        "claim_funeral": int(feats.get("claim_funeral") or 0),
        "claim_support": int(feats.get("claim_support") or 0),
        "claim_consolation": int(feats.get("claim_consolation") or 0),
        "claim_amount": int(feats.get("claim_amount") or 0),
        "claim_vehicle_parts": veh_parts["claim_vehicle_parts"],
        "claim_vehicle_labor": veh_parts["claim_vehicle_labor"],
        "vehicle_age_years": veh_parts["vehicle_age_years"],
        "vehicle_type": veh_parts["vehicle_type"],
        "plaintiff_education": feats.get("plaintiff_education"),
        "plaintiff_occupation": feats.get("plaintiff_occupation"),
        "plaintiff_monthly_income": feats.get("plaintiff_monthly_income"),
        "plaintiff_rest_months": feats.get("plaintiff_rest_months"),
        "plaintiff_labor_loss_ratio": feats.get("plaintiff_labor_loss_ratio"),
    }
    return {"filename": filename, "text": text, "extracted": extracted}


@app.post("/api/depreciation")
def depreciation_preview(req: DepreciationRequest):
    """即時試算車輛修復零件折舊（汽車 5 年、機車 3 年，工資不折舊）。"""
    return calculate_depreciation(
        vehicle_type=req.vehicle_type,
        vehicle_age_years=req.vehicle_age_years,
        parts_cost=req.parts_cost,
        labor_cost=req.labor_cost,
        method=req.method,
    )


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    overrides = _overrides_from_req(req)
    depreciation = _compute_depreciation_for_req(req)
    pred = get_predictor().predict_one(req.text, overrides)
    sim_list = get_retriever().search(req.text, overrides, top_k=5)

    ins_deduct = int(req.insurance_deduction or 0)
    net_award = max(0, int(pred["predicted_award"]) - ins_deduct)
    reason = _rule_summary(pred, depreciation, ins_deduct)
    source = "ml"
    error = None

    if req.sync_llm:
        prompt = _build_enriched_prompt(req, pred, sim_list, depreciation)
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
        "claim_future_medical": int(st.get("claim_future_medical") or 0),
        "claim_care": int(st.get("claim_care") or 0),
        "claim_transport": int(st.get("claim_transport") or 0),
        "claim_work_loss": int(st.get("claim_work_loss") or 0),
        "claim_labor_loss": int(st.get("claim_labor_loss") or 0),
        "claim_vehicle": int(st.get("claim_vehicle") or 0),
        "claim_funeral": int(st.get("claim_funeral") or 0),
        "claim_support": int(st.get("claim_support") or 0),
        "claim_consolation": int(st.get("claim_consolation") or 0),
        "plaintiff_education": req.plaintiff_education or st.get("plaintiff_education"),
        "plaintiff_occupation": req.plaintiff_occupation or st.get("plaintiff_occupation"),
        "plaintiff_monthly_income": (
            req.plaintiff_monthly_income
            if req.plaintiff_monthly_income is not None
            else st.get("plaintiff_monthly_income")
        ),
        "plaintiff_rest_months": (
            req.plaintiff_rest_months
            if req.plaintiff_rest_months is not None
            else st.get("plaintiff_rest_months")
        ),
        "plaintiff_labor_loss_ratio": (
            req.plaintiff_labor_loss_ratio
            if req.plaintiff_labor_loss_ratio is not None
            else st.get("plaintiff_labor_loss_ratio")
        ),
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
        depreciation=depreciation,
        insurance_deduction=ins_deduct,
        net_award_after_insurance=net_award,
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
    depreciation = _compute_depreciation_for_req(req)
    pred = get_predictor().predict_one(req.text, overrides)
    sim_list = get_retriever().search(req.text, overrides, top_k=4)
    prompt = _build_enriched_prompt(req, pred, sim_list, depreciation)
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
