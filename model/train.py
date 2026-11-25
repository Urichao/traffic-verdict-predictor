"""訓練 LightGBM 預測模型與 HAN，輸出相同測試集上的優化前後比較。

用法：
    python -m model.train
    python -m model.train --cases /path/to/cases.parquet --output-dir /path/to/artifacts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from model.features import EXTRA_NUMERIC_FEATURES, FeatureBuilder, prepare_training_features  # noqa: E402
from model.han_baseline import HierarchicalAttentionModel  # noqa: E402
from parsing.extract_rules import LABELS, NUMERIC_FEATURES  # noqa: E402

MODEL_BUNDLE_PATH = config.ARTIFACT_DIR / "predictor.joblib"
METRICS_PATH = config.ARTIFACT_DIR / "metrics.json"
SEED = 42
WEIGHT_EXPONENTS = (0.0, 0.25, 0.5, 0.75, 1.0)
TREE_PARAMS = dict(learning_rate=0.05, num_leaves=15, min_child_samples=5,
                   random_state=SEED, verbose=-1, n_jobs=2)


def restore_consolation(predictions: np.ndarray, transform: str) -> np.ndarray:
    """將慰撫金模型輸出還原為新臺幣，兼容以原始金額訓練的舊模型。"""
    if transform == "log1p":
        predictions = np.expm1(predictions)
    elif transform != "identity":
        raise ValueError(f"未知的慰撫金目標轉換 {transform!r}")
    return np.maximum(0.0, predictions)


def _classifier(X: np.ndarray, y: pd.Series, exponent: float) -> LGBMClassifier:
    counts = y.value_counts()
    weights = ((len(y) / (len(counts) * counts)) ** exponent).to_dict() if exponent else None
    return LGBMClassifier(n_estimators=120, class_weight=weights, **TREE_PARAMS).fit(X, y)


def _macro_f1(y: pd.Series, pred: np.ndarray) -> float:
    return float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0))


def _choose_class_weight(train_df: pd.DataFrame) -> tuple[float, dict[str, float]]:
    y = train_df["outcome"]
    folds = min(5, int(y.value_counts().min()))
    if folds < 2:
        return 0.0, {}
    scores = {exponent: [] for exponent in WEIGHT_EXPONENTS}
    for tr, val in StratifiedKFold(folds, shuffle=True, random_state=SEED).split(train_df, y):
        fb = FeatureBuilder()
        X_tr = fb.fit_transform(train_df.iloc[tr])
        X_val = fb.transform(train_df.iloc[val])
        for exponent in WEIGHT_EXPONENTS:
            clf = _classifier(X_tr, y.iloc[tr], exponent)
            scores[exponent].append(_macro_f1(y.iloc[val], clf.predict(X_val)))
    averages = {exponent: float(np.mean(results)) for exponent, results in scores.items()}
    best = max(averages, key=averages.__getitem__)
    return best, {str(exponent): round(score, 4) for exponent, score in averages.items()}


def _fit_regressor(X_tr, X_te, train_df, test_df, target, *, log=False):
    positive_only = target == "awarded_consolation"
    mask_tr = train_df[target].gt(0) if positive_only else train_df[target].notna()
    mask_te = test_df[target].gt(0) if positive_only else test_df[target].notna()
    y_tr = train_df.loc[mask_tr, target]
    if y_tr.empty:
        raise ValueError(f"訓練集缺少可用的 {target} 標籤")
    reg = LGBMRegressor(n_estimators=120 if target == "award_ratio" else 100, **TREE_PARAMS)
    reg.fit(X_tr[mask_tr], np.log1p(y_tr) if log else y_tr)
    if not mask_te.any():
        return reg, None, 0
    raw = reg.predict(X_te[mask_te])
    if positive_only:
        pred = restore_consolation(raw, "log1p" if log else "identity")
    else:
        pred = np.clip(raw, 0.0, 1.0)
    return reg, round(float(mean_absolute_error(test_df.loc[mask_te, target], pred)), 2 if positive_only else 4), int(mask_te.sum())


def _evaluate(train_df, test_df, *, optimized, exponent=0.0):
    fb = FeatureBuilder() if optimized else FeatureBuilder(numeric_features=NUMERIC_FEATURES)
    X_tr = fb.fit_transform(train_df)
    X_te = fb.transform(test_df)
    clf = _classifier(X_tr, train_df["outcome"], exponent)
    y = test_df["outcome"]
    pred = clf.predict(X_te)
    metrics = {
        "lgbm_accuracy": round(float(accuracy_score(y, pred)), 4),
        "lgbm_macro_f1": round(_macro_f1(y, pred), 4),
    }
    details = {
        "class_f1": {label: round(float(score), 4) for label, score in zip(
            LABELS, f1_score(y, pred, labels=LABELS, average=None, zero_division=0)
        )},
        "regression_test_cases": {},
    }
    bundle = {"feature_builder": fb, "clf": clf}
    for name, target, metric in (
        ("reg_ratio", "award_ratio", "award_ratio_mae"),
        ("reg_fault", "plaintiff_fault", "fault_ratio_mae"),
        ("reg_consol", "awarded_consolation", "consolation_mae"),
    ):
        model, score, n_test = _fit_regressor(
            X_tr, X_te, train_df, test_df, target, log=optimized and name == "reg_consol"
        )
        bundle[name] = model
        metrics[metric] = score
        details["regression_test_cases"][metric] = n_test
    return bundle, metrics, details


def train(cases_path: Path = config.CASES_PATH, output_dir: Path = config.ARTIFACT_DIR) -> dict:
    if not cases_path.exists():
        raise FileNotFoundError(f"找不到 {cases_path}，請先執行 python -m parsing.build_dataset")
    df = pd.read_parquet(cases_path)
    df = df[df["first_instance"] & df["outcome"].isin(LABELS)].reset_index(drop=True)
    counts = df["outcome"].value_counts().reindex(LABELS, fill_value=0)
    if len(df) < 20 or counts.min() < 2:
        raise ValueError(f"有效第一審樣本須至少 20 筆，且每個勝敗類別至少 2 筆。實際筆數 {dict(counts)}")
    baseline_train, baseline_test = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["outcome"]
    )
    for split in (baseline_train, baseline_test):
        if split["outcome"].nunique() != len(LABELS):
            raise ValueError("分層切分後有勝敗類別缺席，請補充稀少類別的樣本")

    prepared = prepare_training_features(df)
    train_df, test_df = prepared.loc[baseline_train.index], prepared.loc[baseline_test.index]
    _, before, before_details = _evaluate(baseline_train, baseline_test, optimized=False)
    exponent, cv_scores = _choose_class_weight(train_df)
    bundle, after, after_details = _evaluate(train_df, test_df, optimized=True, exponent=exponent)

    han = HierarchicalAttentionModel(embed_dim=24, max_sents=25)
    han.fit(train_df["facts_text"].fillna("").tolist(), train_df["outcome"].tolist())
    y_han = han.predict(test_df["facts_text"].fillna("").tolist())
    metrics = {
        "total_cases": len(df),
        "train_cases": len(train_df),
        "test_cases": len(test_df),
        **after,
        "han_baseline_accuracy": round(float(accuracy_score(test_df["outcome"], y_han)), 4),
        "han_baseline_macro_f1": round(_macro_f1(test_df["outcome"], y_han), 4),
    }
    # 與同一份新特徵分類器的無權重版本比較，額外觀察類別權重的影響。
    unweighted = after["lgbm_macro_f1"]
    if exponent:
        fb = bundle["feature_builder"]
        clf = _classifier(fb.transform(train_df), train_df["outcome"], 0.0)
        unweighted = round(_macro_f1(test_df["outcome"], clf.predict(fb.transform(test_df))), 4)
    # 以同一矩陣比較原始金額與 log1p，使目標轉換的影響可單獨觀察。
    fb = bundle["feature_builder"]
    _, raw_target_mae, _ = _fit_regressor(
        fb.transform(train_df), fb.transform(test_df), train_df, test_df, "awarded_consolation"
    )
    comparison = {
        "split": {
            "random_state": SEED, "test_size": 0.2,
            "train_outcomes": train_df["outcome"].value_counts().to_dict(),
            "test_outcomes": test_df["outcome"].value_counts().to_dict(),
        },
        "class_weight_exponent": exponent,
        "class_weight_cv_macro_f1": cv_scores,
        "before": before | before_details,
        "after": after | after_details,
        "without_class_weights_macro_f1": unweighted,
        "without_log_transform_consolation_mae": raw_target_mae,
        "new_feature_counts": {
            "train": {
                col: int(count) for col, count in zip(
                    fb.numeric_feature_names(), np.isfinite(fb._numeric_matrix(train_df)).sum(axis=0)
                ) if col in EXTRA_NUMERIC_FEATURES
            },
            "test": {
                col: int(count) for col, count in zip(
                    fb.numeric_feature_names(), np.isfinite(fb._numeric_matrix(test_df)).sum(axis=0)
                ) if col in EXTRA_NUMERIC_FEATURES
            },
        },
    }
    bundle.update(han=han, metrics=metrics, consolation_target_transform="log1p")
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_dir / MODEL_BUNDLE_PATH.name)
    for filename, content in ((METRICS_PATH.name, metrics), ("model_comparison.json", comparison)):
        (output_dir / filename).write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=config.CASES_PATH)
    parser.add_argument("--output-dir", type=Path, default=config.ARTIFACT_DIR)
    args = parser.parse_args()
    train(args.cases, args.output_dir)
