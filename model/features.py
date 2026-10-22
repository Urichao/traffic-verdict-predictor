"""特徵工程與中文法律斷詞（結合結構化傷勢/金額特徵 + jieba TF-IDF/SVD 文字特徵）。"""
from __future__ import annotations

import re
from typing import Any

import jieba
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from parsing.extract_rules import NUMERIC_FEATURES, extract_features

# 擴充台灣民事車禍判決常用專有名詞字典（對應參考專案 JudgmentPrediction 的 special_v3.txt 概念）
LEGAL_TERMS = [
    "侵權行為", "損害賠償", "過失相抵", "與有過失", "精神慰撫金", "慰撫金",
    "勞動能力減損", "不能工作損失", "醫療費用", "看護費用", "強制汽車責任保險",
    "強制險", "行車事故鑑定委員會", "車鑑會", "肇事主因", "肇事次因", "無肇事因素",
    "闖紅燈", "酒後駕車", "未注意車前狀況", "未保持安全距離", "未依規定讓車",
    "轉彎車應讓直行車", "支線道車應暫停讓幹線道", "行人穿越道", "相當因果關係",
    "診斷證明書", "開放性骨折", "粉碎性骨折", "顱內出血", "硬腦膜下出血", "十字韌帶",
    "擦挫傷", "腦震盪", "零件折舊", "一造辯論", "過失傷害",
]
for _w in LEGAL_TERMS:
    jieba.add_word(_w, freq=2000)

# 特徵中文名稱與對應原文關鍵詞（供 SHAP 圖表與原文螢光筆標註使用）
FEATURE_META: dict[str, tuple[str, tuple[str, ...]]] = {
    "claim_amount": ("請求總金額", ("共計", "合計", "請求")),
    "injury_level": ("傷勢嚴重度", ("骨折", "顱內出血", "腦出血", "腦震盪", "撕裂傷", "擦挫傷", "挫傷", "死亡")),
    "hospital_days": ("住院天數", ("住院",)),
    "surgery": ("接受手術治療", ("手術", "開刀")),
    "disability": ("勞動能力減損/失能", ("勞動能力減損", "減少勞動能力", "失能")),
    "death": ("死亡事故", ("死亡", "不治", "殯葬費")),
    "has_criminal": ("具刑事有罪判決", ("刑事判決", "過失傷害", "過失致死", "刑事庭")),
    "has_appraisal": ("車鑑會鑑定意見", ("鑑定意見", "鑑定委員會", "車鑑會", "肇事主因")),
    "defendant_absent": ("被告未到庭爭執", ("未到庭", "一造辯論")),
    "claims_contributory": ("被告抗辯與有過失", ("與有過失", "過失相抵", "超速", "亦未注意")),
    "insurance_paid": ("已領強制險/保險理賠", ("強制汽車責任保險", "強制險", "理賠")),
    "settled_or_no_fault": ("已和解賠償或無肇事因素", ("達成和解", "和解", "已賠償", "無肇事因素", "無法證明")),
    "claim_medical": ("醫療費用請求", ("醫療費用", "醫藥費", "醫療費")),
    "claim_future_medical": ("將來醫療費", ("將來醫療", "後續醫療")),
    "claim_care": ("看護費用請求", ("看護費", "看護費用", "照護費")),
    "claim_transport": ("就醫交通費", ("交通費",)),
    "claim_work_loss": ("不能工作薪資損失", ("不能工作損失", "工作損失", "薪資損失")),
    "claim_labor_loss": ("勞動能力減損金額", ("勞動能力減損", "減少勞動能力")),
    "claim_vehicle": ("車輛修復費用", ("車輛修理費", "修車費", "車損", "修復費用")),
    "claim_funeral": ("殯葬費", ("殯葬費", "喪葬費")),
    "claim_support": ("扶養費", ("扶養費",)),
    "claim_consolation": ("精神慰撫金請求", ("精神慰撫金", "慰撫金")),
    "veh_motorcycle": ("騎乘機車", ("機車", "重機")),
    "veh_car": ("汽車/自小客車", ("自小客車", "小客車", "汽車")),
    "veh_truck": ("大型車/貨車", ("大貨車", "貨車", "聯結車", "公車")),
    "veh_bicycle": ("自行車", ("腳踏車", "自行車")),
    "pedestrian": ("行人優先路權", ("行人", "行人穿越道", "斑馬線")),
    "v_red_light": ("闖紅燈違規", ("闖紅燈", "紅燈")),
    "v_drunk": ("酒後駕車", ("酒後駕車", "酒駕", "酒精濃度")),
    "v_speeding": ("超速行駛", ("超速", "超過速限")),
    "v_no_yield": ("未依規定禮讓", ("未禮讓", "未依規定禮讓", "未讓", "應暫停讓")),
    "v_inattention": ("未注意車前狀況", ("未注意車前狀況", "疏未注意")),
    "v_wrong_way": ("逆向/跨越雙黃線", ("逆向", "雙黃線")),
    "v_lane_change": ("變換車道/未保安全距離", ("變換車道", "安全距離", "追撞")),
    "v_turn": ("轉彎車違規", ("左轉", "右轉", "迴轉")),
    "v_no_license": ("無照駕駛", ("無照", "駕駛執照")),
    "v_no_helmet": ("未戴安全帽", ("未戴安全帽",)),
}


def cut_text(text: str) -> str:
    tokens = [w.strip() for w in jieba.cut(text or "", cut_all=False) if len(w.strip()) >= 2]
    return " ".join(tokens)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[。；！？\n]+", text or "") if len(s.strip()) >= 4]


class FeatureBuilder:
    """把 DataFrame（結構化欄位 + facts_text）轉成模型輸入矩陣 X。"""

    def __init__(self, n_svd: int = 16):
        self.n_svd = n_svd
        self.tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
        self.svd: TruncatedSVD | None = None
        self.feature_names_: list[str] = []

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        num = self._numeric_matrix(df)
        corpus = [cut_text(t) for t in df["facts_text"].fillna("")]
        mat = self.tfidf.fit_transform(corpus)
        k = min(self.n_svd, max(1, mat.shape[1] - 1), max(1, mat.shape[0] - 1))
        if k >= 2:
            self.svd = TruncatedSVD(n_components=k, random_state=42)
            txt = self.svd.fit_transform(mat)
        else:
            txt = np.zeros((len(df), 0))
        self.feature_names_ = list(NUMERIC_FEATURES) + [f"text_svd_{i}" for i in range(txt.shape[1])]
        return np.hstack([num, txt]).astype(np.float32)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        num = self._numeric_matrix(df)
        corpus = [cut_text(t) for t in df["facts_text"].fillna("")]
        mat = self.tfidf.transform(corpus)
        txt = self.svd.transform(mat) if self.svd is not None else np.zeros((len(df), 0))
        return np.hstack([num, txt]).astype(np.float32)

    @staticmethod
    def _numeric_matrix(df: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in NUMERIC_FEATURES:
            if c in df.columns:
                cols.append(pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy())
            else:
                cols.append(np.zeros(len(df)))
        return np.column_stack(cols)


def merge_user_case(text: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """從使用者輸入的案情文字擷取特徵，並以結構化表單欄位（傷勢、各項請求金額等）覆寫或補充。"""
    feats = extract_features(text or "")
    overrides = overrides or {}
    for k, v in overrides.items():
        if v is None:
            continue
        if k in feats and isinstance(v, (int, float, bool)):
            if isinstance(v, bool):
                feats[k] = int(v)
            elif v > 0 or k == "injury_level":
                feats[k] = v
    # 若使用者分項填寫了醫療、看護、工作損失、車損、慰撫金，但沒填總金額或分項加總大於總金額，自動校正總額
    item_sum = sum(
        int(feats.get(c, 0) or 0)
        for c in (
            "claim_medical", "claim_future_medical", "claim_care", "claim_transport",
            "claim_work_loss", "claim_labor_loss", "claim_vehicle", "claim_funeral",
            "claim_support", "claim_consolation",
        )
    )
    if item_sum > feats.get("claim_amount", 0):
        feats["claim_amount"] = item_sum
    if overrides.get("injury_terms"):
        terms = overrides["injury_terms"]
        if isinstance(terms, str):
            terms = [t.strip() for t in re.split(r"[、,，\s]+", terms) if t.strip()]
        feats["injury_terms"] = sorted(set(list(feats.get("injury_terms", [])) + list(terms)))
    feats["facts_text"] = text or ""
    return feats
