"""訓練民事車禍判決預測模型（LightGBM 分類＋迴歸）與參考專案 HAN 對照基準。

用法：
    python -m model.train
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from model.features import FeatureBuilder  # noqa: E402
from model.han_baseline import HierarchicalAttentionModel  # noqa: E402
from parsing.extract_rules import LABELS  # noqa: E402

MODEL_BUNDLE_PATH = config.ARTIFACT_DIR / "predictor.joblib"
METRICS_PATH = config.ARTIFACT_DIR / "metrics.json"


def train(cases_path: Path = config.CASES_PATH) -> dict:
    if not cases_path.exists():
        raise FileNotFoundError(f"找不到 {cases_path}，請先執行 python -m parsing.build_dataset")
    df = pd.read_parquet(cases_path)
    df = df[df["first_instance"] & df["outcome"].isin(LABELS)].reset_index(drop=True)
    if len(df) < 20:
        raise ValueError(f"有效第一審案件數過少（{len(df)} 筆），無法訓練模型")

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["outcome"]
    )

    fb = FeatureBuilder(n_svd=16)
    X_tr = fb.fit_transform(train_df)
    X_te = fb.transform(test_df)

    # 1. 勝敗訴分類模型 (原告勝訴 / 部分勝訴 / 原告敗訴)
    clf = LGBMClassifier(
        n_estimators=120,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=5,
        random_state=42,
        verbose=-1,
    )
    clf.fit(X_tr, train_df["outcome"])
    y_pred = clf.predict(X_te)

    # 2. 參考專案對照組：階層式 Attention 模型 (HAN)
    han = HierarchicalAttentionModel(embed_dim=24, max_sents=25)
    han.fit(train_df["facts_text"].fillna("").tolist(), train_df["outcome"].tolist())
    y_han = han.predict(test_df["facts_text"].fillna("").tolist())

    # 3. 判賠比例迴歸模型 (award_ratio: 0 ~ 1)
    m_ratio_tr = train_df["award_ratio"].notna()
    m_ratio_te = test_df["award_ratio"].notna()
    reg_ratio = LGBMRegressor(
        n_estimators=120, learning_rate=0.05, num_leaves=15, min_child_samples=5, random_state=42, verbose=-1
    )
    reg_ratio.fit(X_tr[m_ratio_tr], train_df.loc[m_ratio_tr, "award_ratio"])
    pred_ratio = np.clip(reg_ratio.predict(X_te[m_ratio_te]), 0.0, 1.0)

    # 4. 原告與有過失比例迴歸模型 (plaintiff_fault: 0 ~ 1)
    m_fault_tr = train_df["plaintiff_fault"].notna()
    m_fault_te = test_df["plaintiff_fault"].notna()
    reg_fault = LGBMRegressor(
        n_estimators=100, learning_rate=0.05, num_leaves=15, min_child_samples=5, random_state=42, verbose=-1
    )
    reg_fault.fit(X_tr[m_fault_tr], train_df.loc[m_fault_tr, "plaintiff_fault"])
    pred_fault = np.clip(reg_fault.predict(X_te[m_fault_te]), 0.0, 1.0)

    # 5. 精神慰撫金核定金額迴歸模型
    m_con_tr = train_df["awarded_consolation"].notna() & (train_df["awarded_consolation"] > 0)
    m_con_te = test_df["awarded_consolation"].notna() & (test_df["awarded_consolation"] > 0)
    reg_consol = LGBMRegressor(
        n_estimators=100, learning_rate=0.05, num_leaves=15, min_child_samples=5, random_state=42, verbose=-1
    )
    reg_consol.fit(X_tr[m_con_tr], train_df.loc[m_con_tr, "awarded_consolation"])
    pred_con = np.maximum(0.0, reg_consol.predict(X_te[m_con_te]))

    metrics = {
        "total_cases": int(len(df)),
        "train_cases": int(len(train_df)),
        "test_cases": int(len(test_df)),
        "lgbm_accuracy": round(float(accuracy_score(test_df["outcome"], y_pred)), 4),
        "lgbm_macro_f1": round(float(f1_score(test_df["outcome"], y_pred, average="macro")), 4),
        "han_baseline_accuracy": round(float(accuracy_score(test_df["outcome"], y_han)), 4),
        "han_baseline_macro_f1": round(float(f1_score(test_df["outcome"], y_han, average="macro")), 4),
        "award_ratio_mae": round(float(mean_absolute_error(test_df.loc[m_ratio_te, "award_ratio"], pred_ratio)), 4),
        "fault_ratio_mae": round(float(mean_absolute_error(test_df.loc[m_fault_te, "plaintiff_fault"], pred_fault)), 4),
        "consolation_mae": round(float(mean_absolute_error(test_df.loc[m_con_te, "awarded_consolation"], pred_con)), 0),
    }

    bundle = {
        "feature_builder": fb,
        "clf": clf,
        "reg_ratio": reg_ratio,
        "reg_fault": reg_fault,
        "reg_consol": reg_consol,
        "han": han,
        "metrics": metrics,
    }
    joblib.dump(bundle, MODEL_BUNDLE_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("訓練完成！評估指標：")
    for k, v in metrics.items():
        print(f"  {k:24s}: {v}")
    return metrics


if __name__ == "__main__":
    train()
