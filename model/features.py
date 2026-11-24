"""特徵工程與中文法律斷詞（結合結構化傷勢/金額特徵 + jieba TF-IDF/SVD 文字特徵）。"""
from __future__ import annotations

import math
import re
from typing import Any

import jieba
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from api.depreciation import VEHICLE_SPECS, calculate_depreciation, extract_vehicle_parts_labor
from parsing.extract_rules import (
    CLAIM_ITEMS, DEEP_FEATURE_FIELDS, NUMERIC_FEATURES, extract_deep_features, extract_features,
)
from parsing.numbers import NUM_CHARS, cn2int, normalize_width

# 既有欄位順序保留，舊模型及檢索索引仍可讀取。
EXTRA_NUMERIC_FEATURES = (
    *DEEP_FEATURE_FIELDS, "insurance_deduction", "claim_vehicle_parts",
    "claim_vehicle_labor", "vehicle_age_years", "claim_vehicle_depreciated",
)
MODEL_NUMERIC_FEATURES = (*NUMERIC_FEATURES, *EXTRA_NUMERIC_FEATURES)
EDUCATION_LEVELS = {
    "小學": 1, "國小": 1, "國中": 1, "高中": 2, "高職": 2,
    "專科": 3, "大專": 3, "大學": 3, "研究所": 4, "碩士": 4, "博士": 4,
}
OCCUPATION_LEVELS = {
    "無業": 0, "待業": 0, "學生": 0, "退休": 0,
    "司機": 1, "計程車司機": 1, "貨車司機": 1, "工人": 1,
    "服務生": 1, "店員": 1, "廚師": 1, "公務員": 1,
    "設計師": 2, "工程師": 2, "教師": 2, "老師": 2, "護理師": 2, "護士": 2, "醫師": 2,
    "經理": 3, "主管": 3, "高階管理": 3,
}
INSURANCE_AMOUNT_RE = re.compile(
    rf"(?:已領取?|已獲領|已受領|已收取)\s*(?:強制汽車責任保險|強制險|任意險|保險)"
    rf"(?:理賠金?|給付|金)?\s*(?:共計|計|為)?\s*(?:新[臺台]幣)?\s*([{NUM_CHARS}]+)\s*元"
)

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
    "plaintiff_education": ("原告學歷", ("學歷", "畢業", "肄業")),
    "defendant_education": ("被告學歷", ("學歷", "畢業", "肄業")),
    "plaintiff_occupation": ("原告職業", ("職業", "從事", "擔任")),
    "defendant_occupation": ("被告職業", ("職業", "從事", "擔任")),
    "plaintiff_monthly_income": ("原告月收入", ("月收入", "月薪")),
    "defendant_monthly_income": ("被告月收入", ("月收入", "月薪")),
    "plaintiff_rest_months": ("醫囑休養月數", ("醫囑", "休養", "休息")),
    "plaintiff_labor_loss_ratio": ("勞動能力減損比例", ("勞動能力減損", "減少勞動能力")),
    "insurance_deduction": ("已領保險理賠金額", ("強制險", "任意險", "理賠")),
    "claim_vehicle_parts": ("車輛零件費", ("零件費", "零件")),
    "claim_vehicle_labor": ("車輛修復工資", ("工資", "工錢", "烤漆")),
    "vehicle_age_years": ("車齡", ("車齡", "使用期間")),
    "claim_vehicle_depreciated": ("折舊後車損試算", ("折舊", "零件", "車齡")),
}


def cut_text(text: str) -> str:
    tokens = [w.strip() for w in jieba.cut(text or "", cut_all=False) if len(w.strip()) >= 2]
    return " ".join(tokens)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[。；！？\n]+", text or "") if len(s.strip()) >= 4]


def encode_profile(value: Any, levels: dict[str, int]) -> float:
    if not isinstance(value, str):
        return np.nan
    name = normalize_width(value).strip().removesuffix("畢業").removesuffix("肄業")
    return levels.get(name, np.nan)


def _vehicle_depreciated(row: dict[str, Any]) -> int | None:
    parts = row.get("claim_vehicle_parts")
    labor = row.get("claim_vehicle_labor")
    age = row.get("vehicle_age_years")
    vehicle_type = row.get("vehicle_type")
    if not (pd.notna(parts) and parts > 0 and pd.notna(age) and age >= 0 and vehicle_type in VEHICLE_SPECS):
        return None
    labor = labor if pd.notna(labor) else 0
    total = row.get("claim_vehicle")
    if pd.notna(total) and total > parts + labor:
        return None
    method = row.get("depreciation_method")
    return calculate_depreciation(
        vehicle_type=vehicle_type, vehicle_age_years=age, parts_cost=parts, labor_cost=labor,
        method=method if isinstance(method, str) else "declining_balance",
    )["approved_total"]


def _insurance_amount(text: str) -> int | None:
    amounts = {cn2int(m.group(1)) for m in INSURANCE_AMOUNT_RE.finditer(normalize_width(text))}
    return next(iter(amounts)) if len(amounts) == 1 else None


def prepare_training_features(df: pd.DataFrame) -> pd.DataFrame:
    """以案情補齊舊 parquet 缺少的輸入欄位，保留既有結構化值。"""
    prepared = df.copy()
    texts = prepared["facts_text"].fillna("")
    for key, extracted in pd.DataFrame(
        [extract_deep_features(t) for t in texts], index=df.index
    ).items():
        prepared[key] = prepared[key].combine_first(extracted) if key in prepared else extracted
    for key, extracted in pd.DataFrame(
        [extract_vehicle_parts_labor(t) for t in texts], index=df.index
    ).items():
        prepared[key] = prepared[key].combine_first(extracted) if key in prepared else extracted
    insurance = texts.map(_insurance_amount)
    key = "insurance_deduction"
    prepared[key] = prepared[key].combine_first(insurance) if key in prepared else insurance
    insurance_paid = prepared.get("insurance_paid", pd.Series(0, index=df.index))
    prepared["insurance_paid"] = insurance_paid.astype(float).mask(prepared[key].gt(0), 1)
    vehicle = pd.Series([_vehicle_depreciated(row) for row in prepared.to_dict("records")], index=df.index)
    key = "claim_vehicle_depreciated"
    prepared[key] = prepared[key].combine_first(vehicle) if key in prepared else vehicle
    return prepared


class FeatureBuilder:
    """把結構化欄位與 facts_text 轉成模型輸入矩陣。"""

    def __init__(self, n_svd: int = 16, numeric_features=MODEL_NUMERIC_FEATURES):
        self.n_svd = n_svd
        self.numeric_features = tuple(numeric_features)
        self.tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
        self.svd: TruncatedSVD | None = None
        self.feature_names_: list[str] = []

    def numeric_feature_names(self) -> tuple[str, ...]:
        return getattr(self, "numeric_features", tuple(NUMERIC_FEATURES))

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        num = self._numeric_matrix(df)
        self.feature_names_ = list(self.numeric_feature_names())
        self.svd = None
        if len(df) < 2 or self.n_svd < 2:
            return num.astype(np.float32)

        corpus = [cut_text(t) for t in df["facts_text"].fillna("")]
        try:
            mat = self.tfidf.fit_transform(corpus)
        except ValueError as exc:
            if "empty vocabulary" not in str(exc) and "After pruning, no terms remain" not in str(exc):
                raise
            return num.astype(np.float32)

        k = min(self.n_svd, mat.shape[1] - 1, mat.shape[0] - 1)
        if k < 2:
            return num.astype(np.float32)
        self.svd = TruncatedSVD(n_components=k, random_state=42)
        txt = self.svd.fit_transform(mat)
        self.feature_names_.extend(f"text_svd_{i}" for i in range(k))
        return np.hstack([num, txt]).astype(np.float32)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        num = self._numeric_matrix(df)
        if self.svd is None:
            return num.astype(np.float32)
        corpus = [cut_text(t) for t in df["facts_text"].fillna("")]
        txt = self.svd.transform(self.tfidf.transform(corpus))
        return np.hstack([num, txt]).astype(np.float32)

    def _numeric_matrix(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for col in self.numeric_feature_names():
            missing = 0.0 if col in NUMERIC_FEATURES else np.nan
            if col not in df:
                cols.append(np.full(len(df), missing))
                continue
            values = df[col]
            if col.endswith("_education"):
                values = values.map(lambda value: encode_profile(value, EDUCATION_LEVELS))
            elif col.endswith("_occupation"):
                values = values.map(lambda value: encode_profile(value, OCCUPATION_LEVELS))
            values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
            if col in NUMERIC_FEATURES:
                values = values.fillna(0.0)
            cols.append(values.to_numpy(dtype=float, na_value=np.nan))
        return np.column_stack(cols)


def merge_user_case(text: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """從事實擷取特徵，並以結構化表單欄位補充。"""
    text = text or ""
    feats = extract_features(text)
    feats.update(extract_vehicle_parts_labor(text))
    feats["insurance_deduction"] = _insurance_amount(text)
    overrides = overrides or {}
    for key, value in overrides.items():
        if key in ("vehicle_type", "depreciation_method") or (
            key in DEEP_FEATURE_FIELDS and key.endswith(("_education", "_occupation"))
        ):
            if isinstance(value, str) and value.strip():
                feats[key] = value.strip()
        elif isinstance(value, (int, float, bool)) and math.isfinite(value):
            if key in EXTRA_NUMERIC_FEATURES and key != "claim_vehicle_depreciated" and value >= 0:
                feats[key] = value
            elif key in NUMERIC_FEATURES and (isinstance(value, bool) or value > 0 or key == "injury_level"):
                feats[key] = int(value) if isinstance(value, bool) else value

    repair_total = (feats["claim_vehicle_parts"] or 0) + (feats["claim_vehicle_labor"] or 0)
    if repair_total and (not feats["claim_vehicle"] or any(
        overrides.get(key) is not None for key in ("claim_vehicle_parts", "claim_vehicle_labor")
    )) and not overrides.get("claim_vehicle"):
        feats["claim_vehicle"] = repair_total
    feats["claim_vehicle_depreciated"] = _vehicle_depreciated(feats)
    if feats["insurance_deduction"]:
        feats["insurance_paid"] = 1

    item_sum = sum(int(feats.get(key, 0) or 0) for key in CLAIM_ITEMS)
    if item_sum > feats.get("claim_amount", 0):
        feats["claim_amount"] = item_sum
    if overrides.get("injury_terms"):
        terms = overrides["injury_terms"]
        if isinstance(terms, str):
            terms = [t.strip() for t in re.split(r"[、,，\s]+", terms) if t.strip()]
        feats["injury_terms"] = sorted(set(list(feats.get("injury_terms", [])) + list(terms)))
    feats["facts_text"] = text
    return feats
