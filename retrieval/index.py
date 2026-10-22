"""建立並查詢相似歷史車禍判決索引（混合文字 TF-IDF 餘弦相似度 + 傷勢/違規結構化特徵相似度）。

用法：
    python -m retrieval.index
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from model.features import cut_text, merge_user_case  # noqa: E402
from parsing.extract_rules import INJURY_LEVEL_NAMES, LABELS, VEHICLE_KW, VIOLATION_KW  # noqa: E402

INDEX_PATH = config.ARTIFACT_DIR / "case_index.joblib"

COURT_SHORT = {
    "TPDV": "臺北地院", "PCDV": "新北地院", "SLDV": "士林地院", "TYDV": "桃園地院",
    "SCDV": "新竹地院", "MLDV": "苗栗地院", "TCDV": "臺中地院", "CHDV": "彰化地院",
    "NTDV": "南投地院", "ULDV": "雲林地院", "CYDV": "嘉義地院", "TNDV": "臺南地院",
    "KSDV": "高雄地院", "CTDV": "橋頭地院", "PTDV": "屏東地院", "ILDV": "宜蘭地院",
    "HLDV": "花蓮地院", "TTDV": "臺東地院", "KLDV": "基隆地院",
}

STRUCT_COLS = ["injury_level", "hospital_days", "surgery", "disability", "claims_contributory"] + list(VEHICLE_KW) + list(VIOLATION_KW)


def format_case_id(row: pd.Series | dict) -> str:
    jid = str(row.get("jid", ""))
    parts = jid.split(",")
    if len(parts) >= 4:
        court = COURT_SHORT.get(parts[0], parts[0])
        return f"{court} {parts[1]}年度{parts[2]}字第{parts[3]}號"
    return jid or "歷史判決"


def summarize_case(row: pd.Series | dict) -> str:
    lv = INJURY_LEVEL_NAMES.get(int(row.get("injury_level") or 0), "車禍傷害")
    terms = row.get("injury_terms") or ""
    inj = f"{lv}（{terms}）" if terms else lv
    claim = int(row.get("claim_amount") or 0)
    award = int(row.get("awarded_amount") or 0)
    fault = row.get("plaintiff_fault")
    fault_str = f"，法院認原告與有過失 {int(round(float(fault) * 100))}%" if pd.notna(fault) and float(fault) > 0 else "，原告無過失"
    consol = row.get("awarded_consolation")
    consol_str = f"（含慰撫金 {int(consol):,} 元）" if pd.notna(consol) and float(consol) > 0 else ""
    if row.get("outcome") == "原告敗訴":
        facts = str(row.get("facts_text") or "")[:70]
        return f"{facts}… 請求 {claim:,} 元，因舉證不足或已和解賠償，法院駁回全部請求。"
    return f"傷勢：{inj}。原告請求 {claim:,} 元{fault_str}，法院判准 {award:,} 元{consol_str}。"


class CaseRetriever:
    def __init__(self, index_path: Path = INDEX_PATH):
        data = joblib.load(index_path)
        self.tfidf: TfidfVectorizer = data["tfidf"]
        self.tfidf_mat = data["tfidf_mat"]
        self.scaler: StandardScaler = data["scaler"]
        self.struct_mat = data["struct_mat"]
        self.cases: pd.DataFrame = data["cases"]

    def search(self, text: str, overrides: dict[str, Any] | None = None, top_k: int = 5) -> list[dict[str, Any]]:
        feats = merge_user_case(text, overrides)
        q_txt = self.tfidf.transform([cut_text(text or "")])
        sim_txt = cosine_similarity(q_txt, self.tfidf_mat)[0]

        vec = np.array([[float(feats.get(c, 0) or 0) for c in STRUCT_COLS] + [math.log1p(max(0, float(feats.get("claim_amount") or 0)))]])
        q_st = self.scaler.transform(vec)
        sim_st = cosine_similarity(q_st, self.struct_mat)[0]
        sim_st = (sim_st + 1.0) / 2.0  # 轉至 0~1

        score = 0.55 * sim_txt + 0.45 * sim_st
        idxs = np.argsort(-score)[:top_k]
        out = []
        for idx in idxs:
            row = self.cases.iloc[int(idx)]
            out.append({
                "jid": str(row["jid"]),
                "case_id": format_case_id(row),
                "similarity": round(float(min(0.99, max(0.35, score[idx]))), 2),
                "verdict": str(row["outcome"]),
                "claim_amount": int(row.get("claim_amount") or 0),
                "awarded_amount": int(row.get("awarded_amount") or 0),
                "plaintiff_fault": float(row["plaintiff_fault"]) if pd.notna(row.get("plaintiff_fault")) else 0.0,
                "summary": summarize_case(row),
            })
        return out


def build_index(cases_path: Path = config.CASES_PATH, out_path: Path = INDEX_PATH):
    df = pd.read_parquet(cases_path)
    df = df[df["first_instance"] & df["outcome"].isin(LABELS)].reset_index(drop=True)
    corpus = [cut_text(t) for t in df["facts_text"].fillna("")]
    tfidf = TfidfVectorizer(max_features=4000, ngram_range=(1, 2))
    tfidf_mat = tfidf.fit_transform(corpus)

    raw_struct = np.column_stack(
        [pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy() for c in STRUCT_COLS]
        + [np.log1p(np.maximum(0.0, pd.to_numeric(df["claim_amount"], errors="coerce").fillna(0.0).to_numpy()))]
    )
    scaler = StandardScaler()
    struct_mat = scaler.fit_transform(raw_struct)

    keep_cols = [
        "jid", "outcome", "claim_amount", "awarded_amount", "award_ratio",
        "plaintiff_fault", "awarded_consolation", "injury_level", "injury_terms", "facts_text",
    ]
    cases_df = df[keep_cols].copy()
    cases_df["facts_text"] = cases_df["facts_text"].fillna("").astype(str).str.slice(0, 160)
    joblib.dump(
        {"tfidf": tfidf, "tfidf_mat": tfidf_mat, "scaler": scaler, "struct_mat": struct_mat, "cases": cases_df},
        out_path,
        compress=3,
    )
    print(f"已建立相似判決索引（共 {len(df)} 筆）→ {out_path}")


if __name__ == "__main__":
    build_index()
