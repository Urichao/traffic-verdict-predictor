"""將訓練好的 LightGBM 模型、HAN 注意力權重與 12,174 筆真實判決索引匯出為前端可直接執行的精簡 JSON。

這讓本系統能直接部署於全免費的 Hugging Face Static Space / GitHub Pages，
即使不需付費購買雲端容器或開著個人電腦，也能 24 小時在瀏覽器端完成：
1. 台灣民事車禍案件特徵擷取
2. LightGBM 勝敗訴分類、判賠比例、過失比例、慰撫金迴歸推論
3. 特徵邊際貢獻（SHAP 近似）與 HAN 句子 Attention 權重計算
4. 12,174 筆真實民事車禍判決混合檢索
5. 結構化白話法律分析報告生成（亦可隨時連接本地/雲端 Gemma-4-E4B）
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from model.features import FEATURE_META  # noqa: E402
from parsing.extract_rules import INJURY_LEVEL_NAMES, NUMERIC_FEATURES  # noqa: E402
from retrieval.index import INDEX_PATH, format_case_id, summarize_case  # noqa: E402

WEB_DIR = config.ROOT / "web"


def _compact_tree(node: dict):
    """將 LightGBM dump_model 的樹節點轉為極精簡陣列：
    內部節點: [split_feature, threshold, left_subtree, right_subtree]
    葉節點:   round(leaf_value, 6)
    """
    if "leaf_value" in node:
        return round(float(node["leaf_value"]), 6)
    return [
        int(node["split_feature"]),
        round(float(node["threshold"]), 5),
        _compact_tree(node["left_child"]),
        _compact_tree(node["right_child"]),
    ]


def _dump_booster(model) -> list:
    info = model.booster_.dump_model()["tree_info"]
    return [_compact_tree(t["tree_structure"]) for t in info]


def export_all(cloud_key: str = "") -> None:
    bundle = joblib.load(config.ARTIFACT_DIR / "predictor.joblib")
    idx_data = joblib.load(INDEX_PATH)

    clf = bundle["clf"]
    reg_ratio = bundle["reg_ratio"]
    reg_fault = bundle["reg_fault"]
    reg_consol = bundle["reg_consol"]
    han = bundle["han"]
    metrics = bundle.get("metrics", {})

    # 匯出 HAN 詞級注意力分數
    vocab = han.word_tfidf.vocabulary_
    idf = han.word_tfidf.idf_
    comp = han.svd.components_
    u_w = han.word_context_u_
    word_scores: dict[str, float] = {}
    for w, idx in vocab.items():
        if len(w) < 2:
            continue
        vec = comp[:, idx] * idf[idx]
        score = float(abs(np.tanh(vec @ u_w))) if u_w is not None else float(np.linalg.norm(vec))
        if score > 0.05:
            word_scores[w] = round(score, 4)

    model_path = WEB_DIR / "model_bundle.json"
    prev_b64 = ""
    if model_path.exists():
        try:
            prev_b64 = json.loads(model_path.read_text(encoding="utf-8")).get("cloud_key_b64", "")
        except Exception:
            pass
    raw_key = (cloud_key or os.getenv("GEMINI_API_KEY", "")).strip()
    key_b64 = base64.b64encode(raw_key.encode("utf-8")).decode("ascii") if raw_key else prev_b64

    model_json = {
        "metrics": metrics,
        "classes": list(clf.classes_),
        "numeric_features": list(NUMERIC_FEATURES),
        "injury_level_names": {str(k): v for k, v in INJURY_LEVEL_NAMES.items()},
        "feature_meta": {k: [v[0], list(v[1])] for k, v in FEATURE_META.items()},
        "clf_trees": _dump_booster(clf),
        "ratio_trees": _dump_booster(reg_ratio),
        "fault_trees": _dump_booster(reg_fault),
        "consol_trees": _dump_booster(reg_consol),
        "word_scores": word_scores,
        "cloud_key_b64": key_b64,
    }

    model_path.write_text(json.dumps(model_json, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    # 匯出 12,174 筆真實案例精簡檢索庫
    cases_df = idx_data["cases"]
    outcome_map = {"原告勝訴": 0, "部分勝訴": 1, "原告敗訴": 2}
    compact_cases = []
    for _, row in cases_df.iterrows():
        cid = format_case_id(row)
        out_code = outcome_map.get(str(row.get("outcome", "")), 1)
        claim = int(row.get("claim_amount") or 0)
        award = int(row.get("awarded_amount") or 0)
        fault = round(float(row["plaintiff_fault"]), 2) if row.get("plaintiff_fault") is not None and not np.isnan(row["plaintiff_fault"]) else 0.0
        lv = int(row.get("injury_level") or 0)
        terms = str(row.get("injury_terms") or "")
        summary = summarize_case(row)
        compact_cases.append([cid, out_code, claim, award, fault, lv, terms, summary])

    cases_path = WEB_DIR / "cases_bundle.json"
    cases_path.write_text(json.dumps(compact_cases, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    print(
        f"已匯出前端靜態推論包：\n"
        f"  - {model_path} ({model_path.stat().st_size / 1024:.1f} KB)\n"
        f"  - {cases_path} ({cases_path.stat().st_size / 1024 / 1024:.2f} MB，共 {len(compact_cases)} 筆真實判決)"
    )


if __name__ == "__main__":
    export_all()
