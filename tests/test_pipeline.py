"""單元與整合測試：數字與比例解析、主文與段落切分、要點擷取、模型推論、相似案例檢索與 API。"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api.main import app  # noqa: E402
from parsing.extract_rules import extract_case, extract_plaintiff_fault, parse_main  # noqa: E402
from parsing.numbers import cn2int, parse_ratio  # noqa: E402
from parsing.sections import split_sections  # noqa: E402


def test_cn2int_and_ratio():
    assert cn2int("壹佰貳拾萬參仟") == 1_203_000
    assert cn2int("40萬") == 400_000
    assert cn2int("125,600") == 125_600
    assert cn2int("三萬五千") == 35_000
    assert parse_ratio("百分之三十") == 0.3
    assert parse_ratio("三成") == 0.3
    assert parse_ratio("25%") == 0.25
    assert parse_ratio("十分之四") == 0.4


def test_parse_main_outcomes():
    m1 = parse_main("被告應給付原告新臺幣貳拾伍萬元，及自民國112年5月1日起至清償日止按週年利率百分之五計算之利息。原告其餘之訴駁回。")
    assert m1["outcome"] == "部分勝訴"
    assert m1["awarded_amount"] == 250_000

    m2 = parse_main("被告應給付原告新臺幣400,000元。訴訟費用由被告負擔。")
    assert m2["outcome"] == "原告勝訴"
    assert m2["awarded_amount"] == 400_000

    m3 = parse_main("原告之訴及其假執行之聲請均駁回。訴訟費用由原告負擔。")
    assert m3["outcome"] == "原告敗訴"
    assert m3["awarded_amount"] == 0


def test_split_sections_and_extract_case():
    sample_doc = {
        "JID": "TPDV,112,訴,888,20231015,1",
        "JYEAR": "112",
        "JCASE": "訴",
        "JNO": "888",
        "JDATE": "20231015",
        "JTITLE": "侵權行為損害賠償(交通)",
        "JFULL": (
            "臺灣臺北地方法院民事判決\n112年度訴字第888號\n"
            "主文\n被告應給付原告新臺幣210,000元。原告其餘之訴駁回。\n"
            "事實及理由\n"
            "一、原告起訴主張：被告駕駛自小客車闖紅燈撞擊原告機車，致原告受有左腳骨折，住院5日並接受手術。"
            "原告請求醫療費用50,000元、看護費用20,000元及精神慰撫金230,000元，共計300,000元。"
            "並聲明：被告應給付原告新臺幣300,000元。\n"
            "二、被告抗辯：原告行經路口超速，與有過失。\n"
            "三、本院之判斷：本院審酌精神慰撫金以150,000元為適當。又原告應負擔過失責任比例為百分之三十。\n"
            "中華民國 112 年 10 月 15 日"
        ),
    }
    sec = split_sections(sample_doc["JFULL"])
    assert "210,000" in sec.main
    assert "本院審酌" not in sec.facts  # 確認法院心證未洩漏至事實特徵
    rec = extract_case(sample_doc)
    assert rec["outcome"] == "部分勝訴"
    assert rec["claim_amount"] == 300_000
    assert rec["awarded_amount"] == 210_000
    assert rec["injury_level"] == 2
    assert rec["plaintiff_fault"] == 0.3
    assert rec["awarded_consolation"] == 150_000


def test_extract_plaintiff_fault():
    assert extract_plaintiff_fault("認原告應負擔過失責任比例為百分之二十") == 0.2
    assert extract_plaintiff_fault("被告雖抗辯原告與有過失，惟查原告並無任何違規，無過失相抵之適用") == 0.0


def test_api_predict_endpoint():
    client = TestClient(app)
    r = client.post(
        "/api/predict",
        json={
            "text": "原告騎乘機車直行，被告駕駛自小客車左轉未禮讓撞擊原告，致原告左腳骨折，住院5日並接受手術。原告請求醫療費用45,000元、看護費用25,000元、不能工作損失50,000元及精神慰撫金180,000元，共計300,000元。被告抗辯原告超速與有過失。",
            "injury_level": 2,
            "hospital_days": 5,
            "surgery": True,
            "claim_amount": 300000,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["label"] in ("原告勝訴", "部分勝訴", "原告敗訴")
    assert 0.0 <= data["predicted_ratio"] <= 1.0
    assert len(data["similar"]) > 0
    assert len(data["sentence_scores"]) > 0
