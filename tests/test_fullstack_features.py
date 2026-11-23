"""全端與 API 新功能測試：車輛零件折舊試算、文字檔上傳 (.txt/.docx)、擴充欄位與推論效能。"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.depreciation import calculate_depreciation, extract_vehicle_parts_labor  # noqa: E402
from api.main import app  # noqa: E402

client = TestClient(app)


def test_depreciation_car_and_motorcycle():
    # 自用小客車（耐用 5 年，定率遞減率 0.369），滿 1 年零件 100,000 → 63,100，工資 20,000 不折舊
    d1 = calculate_depreciation(vehicle_type="car", vehicle_age_years=1.0, parts_cost=100000, labor_cost=20000)
    assert d1["applied"] is True
    assert d1["parts_depreciated"] == 63100
    assert d1["approved_total"] == 83100
    assert d1["deducted_amount"] == 36900

    # 超過耐用年數（機車 3 年，車齡 5 年）→ 殘值下限 10%
    d2 = calculate_depreciation(vehicle_type="motorcycle", vehicle_age_years=5.0, parts_cost=30000, labor_cost=5000)
    assert d2["residual_ratio"] == 0.1
    assert d2["parts_depreciated"] == 3000
    assert d2["approved_total"] == 8000

    # 平均法（自用小客車 5 年，殘值下限 1/6，車齡 5 年）
    d3 = calculate_depreciation(
        vehicle_type="car", vehicle_age_years=5.0, parts_cost=60000, labor_cost=10000, method="straight_line"
    )
    assert d3["parts_depreciated"] == 10000
    assert d3["approved_total"] == 20000


def test_extract_vehicle_parts_labor():
    info = extract_vehicle_parts_labor("我騎機車車齡3年，修車零件費18,000元、工資6,000元。")
    assert info["claim_vehicle_parts"] == 18000
    assert info["claim_vehicle_labor"] == 6000
    assert info["vehicle_age_years"] == 3.0
    assert info["vehicle_type"] == "motorcycle"


def test_upload_txt_and_docx_api():
    txt_content = (
        "我大學畢業，從事工程師工作，每月收入約5萬元。"
        "我騎機車遭汽車未禮讓撞擊致右鎖骨骨折，住院4日並開刀，醫囑休養約2個月。"
        "支出醫療費用35,000元、計程車資3,000元、請假扣薪40,000元、修車零件12,000元、工資4,000元，車齡2年。"
    )
    r_txt = client.post(
        "/api/upload-text",
        files={"file": ("case_oral.txt", txt_content.encode("utf-8"), "text/plain")},
    )
    assert r_txt.status_code == 200, r_txt.text
    ex = r_txt.json()["extracted"]
    assert ex["injury_level"] == 2
    assert ex["hospital_days"] == 4
    assert ex["surgery"] is True
    assert ex["claim_medical"] == 35000
    assert ex["claim_transport"] == 3000
    assert ex["claim_work_loss"] == 40000
    assert ex["claim_vehicle_parts"] == 12000
    assert ex["claim_vehicle_labor"] == 4000
    assert ex["plaintiff_education"] == "大學"
    assert ex["plaintiff_occupation"] == "工程師"
    assert ex["plaintiff_monthly_income"] == 50000
    assert ex["plaintiff_rest_months"] == 2

    # 測試 .docx 壓縮 XML 解析
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "word/document.xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:body><w:p><w:r><w:t>我高中畢業，月收入約35,000元，計程車資2,000元。</w:t></w:r></w:p></w:body>'
            "</w:document>",
        )
    r_docx = client.post(
        "/api/upload-text",
        files={"file": ("draft.docx", buf.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )
    assert r_docx.status_code == 200, r_docx.text
    assert r_docx.json()["extracted"]["plaintiff_monthly_income"] == 35000
    assert r_docx.json()["extracted"]["claim_transport"] == 2000


def test_predict_with_depreciation_and_extra_fields():
    payload = {
        "text": "被告開車左轉未禮讓直行機車，致原告受有左橈骨骨折，住院5日並接受手術，請求醫療費45,000元、精神慰撫金200,000元。",
        "injury_level": 2,
        "hospital_days": 5,
        "surgery": True,
        "vehicle_type": "motorcycle",
        "vehicle_age_years": 2.0,
        "claim_vehicle_parts": 20000,
        "claim_vehicle_labor": 5000,
        "claim_future_medical": 15000,
        "claim_transport": 3000,
        "insurance_deduction": 30000,
        "plaintiff_education": "大學",
        "plaintiff_occupation": "工程師",
        "plaintiff_monthly_income": 55000,
        "plaintiff_rest_months": 3.0,
    }
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["elapsed_ms"] < 3000
    assert data["depreciation"]["applied"] is True
    assert data["depreciation"]["original_total"] == 25000
    assert data["depreciation"]["approved_total"] < 25000
    assert data["insurance_deduction"] == 30000
    assert data["net_award_after_insurance"] == max(0, data["predicted_award"] - 30000)
    assert data["extracted"]["claim_future_medical"] == 15000
    assert data["extracted"]["claim_transport"] == 3000
    assert data["extracted"]["plaintiff_monthly_income"] == 55000
