from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from api.main import PredictRequest, _overrides_from_req
from model.features import FeatureBuilder, merge_user_case, prepare_training_features
from model.predict import VerdictPredictor
from model.train import restore_consolation
from parsing.extract_rules import NUMERIC_FEATURES


def test_form_features_reach_model_and_estimate_vehicle_damage():
    req = PredictRequest(
        text="原告月收入五萬元，被告高中畢業。原告醫囑休養兩個月。",
        plaintiff_education="大學", plaintiff_occupation="退休", plaintiff_monthly_income=0,
        claim_future_medical=900, insurance_deduction=3000, vehicle_type="car",
        vehicle_age_years=1, claim_vehicle_parts=100000, claim_vehicle_labor=20000,
    )
    row = merge_user_case(req.text, _overrides_from_req(req))
    assert row["claim_vehicle_depreciated"] == 83100
    assert row["claim_amount"] == 120900
    assert row["insurance_paid"] == 1

    fb = FeatureBuilder()
    matrix = fb.fit_transform(pd.DataFrame([row, merge_user_case("原告主張發生車禍。")]))
    columns = dict(zip(fb.feature_names_, matrix[0]))
    assert columns["plaintiff_education"] == 3
    assert columns["defendant_education"] == 2
    assert columns["plaintiff_occupation"] == 0
    assert columns["plaintiff_monthly_income"] == 0
    assert columns["plaintiff_rest_months"] == 2
    assert columns["claim_vehicle_depreciated"] == 83100
    assert np.isnan(matrix[1, fb.feature_names_.index("plaintiff_occupation")])
    assert np.isnan(matrix[1, fb.feature_names_.index("claim_vehicle_depreciated")])


def test_training_backfills_from_facts_and_preserves_existing_values():
    text = "原告大學畢業，月收入五萬元。"
    df = pd.DataFrame({
        "facts_text": [text, "原告受傷。"], "insurance_paid": [0, 0],
        "plaintiff_monthly_income": [60000, None], "court_defendant_education": ["博士", "大學"],
        "claim_vehicle": [120000, 0], "claim_vehicle_parts": [100000, None],
        "claim_vehicle_labor": [20000, None], "vehicle_type": ["car", None],
        "vehicle_age_years": [1, None],
    })
    result = prepare_training_features(df)
    assert result.loc[0, "plaintiff_education"] == "大學"
    assert result.loc[0, "plaintiff_monthly_income"] == 60000
    assert result.loc[0, "claim_vehicle_depreciated"] == 83100
    assert pd.isna(result.loc[0, "defendant_education"])
    assert pd.isna(result.loc[1, "claim_vehicle_depreciated"])


def test_consolation_restore_and_legacy_feature_order():
    assert restore_consolation(np.log1p([0.0, 125000.0]), "log1p") == pytest.approx([0, 125000])
    assert restore_consolation(np.array([-1.0, 125000.0]), "identity").tolist() == [0, 125000]
    predictor = VerdictPredictor(Path(__file__).resolve().parents[1] / "artifacts" / "predictor.joblib")
    # 當 GitHub 舊模型日後被新模型取代，仍檢查舊版 builder 的相容路徑。
    legacy = FeatureBuilder()
    del legacy.numeric_features
    assert legacy.numeric_feature_names() == tuple(NUMERIC_FEATURES)
    result = predictor.predict_one("原告因車禍骨折，請求精神慰撫金200,000元。")
    assert 0 <= result["predicted_consolation"] <= 200000
