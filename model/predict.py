"""線上推論與可解釋分析（SHAP 值 + 參考專案 HAN 句子重要度）。"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from model.features import FEATURE_META, merge_user_case  # noqa: E402
from model.train import MODEL_BUNDLE_PATH  # noqa: E402
from parsing.extract_rules import INJURY_LEVEL_NAMES, NUMERIC_FEATURES  # noqa: E402


class VerdictPredictor:
    def __init__(self, bundle_path: Path = MODEL_BUNDLE_PATH):
        self.bundle = joblib.load(bundle_path)
        self.fb = self.bundle["feature_builder"]
        self.clf = self.bundle["clf"]
        self.reg_ratio = self.bundle["reg_ratio"]
        self.reg_fault = self.bundle["reg_fault"]
        self.reg_consol = self.bundle["reg_consol"]
        self.han = self.bundle["han"]
        self.metrics = self.bundle.get("metrics", {})
        self.explainer = shap.TreeExplainer(self.reg_ratio)

    def predict_one(self, text: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        feats = merge_user_case(text, overrides)
        row_df = pd.DataFrame([feats])
        X = self.fb.transform(row_df)

        # 1. 勝敗訴機率與預測類別
        proba = self.clf.predict_proba(X)[0]
        classes = list(self.clf.classes_)
        class_probs = {c: round(float(p), 4) for c, p in zip(classes, proba)}
        top_idx = int(np.argmax(proba))
        label = classes[top_idx]
        confidence = round(float(proba[top_idx]), 2)

        # 2. 判賠比例、過失比例、精神慰撫金預測
        raw_ratio = float(np.clip(self.reg_ratio.predict(X)[0], 0.0, 1.0))
        raw_fault = float(np.clip(self.reg_fault.predict(X)[0], 0.0, 1.0))
        raw_consol = float(max(0.0, self.reg_consol.predict(X)[0]))

        if label == "原告敗訴":
            pred_ratio = 0.0
            pred_consol = 0
        else:
            pred_ratio = round(raw_ratio, 3)
            pred_consol = int(round(raw_consol / 5000) * 5000)
            if feats.get("claim_consolation", 0) > 0:
                pred_consol = min(pred_consol, int(feats["claim_consolation"]))

        claim_amt = int(feats.get("claim_amount") or 0)
        pred_award = int(round(claim_amt * pred_ratio / 100) * 100) if claim_amt > 0 else pred_consol
        pred_fault = round(raw_fault, 2)

        # 3. SHAP 可解釋分析（針對判賠比例模型計算各結構化特徵之 SHAP 貢獻）
        shap_vals = self.explainer.shap_values(X)[0]
        factors = self._build_shap_factors(text, feats, shap_vals)

        # 4. 參考專案 HAN 句子重要度（Attention 權重）
        _, sentence_scores = self.han.encode_with_attention(text)

        feats["injury_level_name"] = INJURY_LEVEL_NAMES.get(int(feats.get("injury_level", 0)), "未指定")
        feats["injury_terms_str"] = "、".join(feats.get("injury_terms") or []) or "無特定傷勢關鍵詞"

        return {
            "label": label,
            "confidence": confidence,
            "class_probs": class_probs,
            "predicted_ratio": pred_ratio,
            "predicted_award": pred_award,
            "predicted_fault": pred_fault,
            "predicted_consolation": pred_consol,
            "structured": feats,
            "factors": factors,
            "sentence_scores": sentence_scores,
            "metrics": self.metrics,
        }

    def _build_shap_factors(
        self, text: str, feats: dict[str, Any], shap_vals: np.ndarray
    ) -> list[dict[str, Any]]:
        """只挑出本案實際發生/填寫之特徵，依 |SHAP| 排序並轉為 0~1 相對權重與原文對應詞。"""
        active = []
        for i, col in enumerate(NUMERIC_FEATURES):
            val = feats.get(col, 0)
            if not val:
                continue
            sv = float(shap_vals[i])
            if abs(sv) < 1e-4:
                continue
            cn_name, candidate_terms = FEATURE_META.get(col, (col, ()))
            matched_term = next((kw for kw in candidate_terms if kw in (text or "")), cn_name)
            effect = 1 if sv >= 0 else -1
            active.append({
                "feature": col,
                "name": cn_name,
                "term": matched_term,
                "effect": effect,
                "raw_shap": sv,
            })

        active.sort(key=lambda d: abs(d["raw_shap"]), reverse=True)
        top = active[:6]
        max_abs = max((abs(d["raw_shap"]) for d in top), default=1.0)
        for d in top:
            d["weight"] = round(min(1.0, max(0.08, abs(d["raw_shap"]) / (max_abs * 1.1))), 2)
        return top
