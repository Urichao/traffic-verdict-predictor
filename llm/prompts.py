"""Gemma-4-E4B 專用的系統提示詞與 RAG 模板（針對 CPU 推論速度精簡 Prompt 長度）。"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------- 1. 判決要點補齊（離線資料處理用）
EXTRACT_MISSING_SYSTEM = """你是台灣民事車禍判決書結構化擷取助手。請僅輸出一個 JSON 物件：
{"outcome":"原告勝訴"|"部分勝訴"|"原告敗訴"|null,"claim_amount":整數或null,"awarded_amount":整數或null,"plaintiff_fault":0到1小數或null,"awarded_consolation":整數或null}"""

# ---------------------------------------------------------------- 2. 線上預測與可解釋分析（RAG）
ANALYZE_SYSTEM = (
    "你是台灣民事車禍判決預測系統的法律分析助理。"
    "請僅依據提供的案件事實、模型預測數據與相似判例，用繁體中文撰寫約 160 字的精煉白話分析。"
    "切勿自行捏造數字或法條；最後一句請加註「本分析僅供學術研究參考，不構成正式法律意見。」"
)


def build_rag_prompt(
    user_text: str,
    structured_info: dict[str, Any],
    ml_pred: dict[str, Any],
    shap_factors: list[dict[str, Any]],
    similar_cases: list[dict[str, Any]],
) -> str:
    """精簡版 RAG Prompt，大幅降低無獨顯 CPU 之 Prefill 延遲。"""
    fac_str = "、".join(
        f"{f['name']}({'有利' if f.get('effect', 1) > 0 else '不利'})" for f in shap_factors[:4]
    )
    sim_str = "；".join(
        f"{c.get('case_id')}({c.get('verdict')}，請求{c.get('claim_amount', 0):,}→判准{c.get('awarded_amount', 0):,}元，原告過失{int(round((c.get('plaintiff_fault') or 0)*100))}%)"
        for c in similar_cases[:2]
    )
    return (
        f"【案情】{user_text[:350]}\n"
        f"【傷勢與請求】{structured_info.get('injury_level_name')}（{structured_info.get('injury_terms_str')}），"
        f"住院{structured_info.get('hospital_days', 0)}日，請求總額{structured_info.get('claim_amount', 0):,}元"
        f"（含醫療{structured_info.get('claim_medical', 0):,}、看護{structured_info.get('claim_care', 0):,}、"
        f"工作損失{structured_info.get('claim_work_loss', 0):,}、慰撫金{structured_info.get('claim_consolation', 0):,}元）。\n"
        f"【模型預測】{ml_pred.get('label')}（信心{ml_pred.get('confidence', 0):.0%}），"
        f"預估判賠{ml_pred.get('predicted_award', 0):,}元（{ml_pred.get('predicted_ratio', 0):.0%}），"
        f"預估原告與有過失{ml_pred.get('predicted_fault', 0):.0%}，預估核定慰撫金{ml_pred.get('predicted_consolation', 0):,}元。\n"
        f"【SHAP關鍵因子】{fac_str}\n"
        f"【相似判例】{sim_str}\n"
        "請分三段簡明回答：1.預測結果與判賠金額 2.傷勢與過失相抵依據 3.相似判例比照與舉證建議。"
    )
