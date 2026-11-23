"""台灣民事車禍實務「車輛修復零件折舊」試算模組。

依據：
- 行政院《固定資產耐用年數表》與《固定資產折舊率表》
  - 自用小客車（非運輸業用客車）：耐用年數 5 年，定率遞減法年折舊率 0.369
  - 機車（機器腳踏車）：耐用年數 3 年，定率遞減法年折舊率 0.536
  - 營業用小客車（計程車／運輸業客車）：耐用年數 4 年，定率遞減法年折舊率 0.438
  - 貨車／大客車：耐用年數 5 年，定率遞減法年折舊率 0.369
- 營利事業所得稅查核準則第 95 條第 6 項：
  - 定率遞減法：最後 1 年（或超過耐用年數）之折舊後殘值不低於原價之 1/10（10%）
  - 平均法：預留殘值為 原價 / (耐用年數 + 1)
- 民事法院實務原則：
  - 「更換新零件」應扣除折舊；「工資／烤漆費」不予折舊（100% 計入）。
"""
from __future__ import annotations

import math
import re
from typing import Any

from parsing.numbers import NUM_CHARS, cn2int

VEHICLE_SPECS: dict[str, dict[str, Any]] = {
    "car": {"name": "自用小客車", "useful_life": 5, "declining_rate": 0.369},
    "motorcycle": {"name": "機車（機器腳踏車）", "useful_life": 3, "declining_rate": 0.536},
    "commercial_car": {"name": "營業用客車／計程車", "useful_life": 4, "declining_rate": 0.438},
    "truck": {"name": "貨車／大客車", "useful_life": 5, "declining_rate": 0.369},
}


def calculate_depreciation(
    *,
    vehicle_type: str = "car",
    vehicle_age_years: float = 0.0,
    parts_cost: int = 0,
    labor_cost: int = 0,
    method: str = "declining_balance",
) -> dict[str, Any]:
    """試算車輛修復費用（零件折舊 ＋ 工資不折舊）。"""
    spec = VEHICLE_SPECS.get(vehicle_type, VEHICLE_SPECS["car"])
    life = int(spec["useful_life"])
    rate = float(spec["declining_rate"])
    age = max(0.0, float(vehicle_age_years or 0.0))
    parts = max(0, int(parts_cost or 0))
    labor = max(0, int(labor_cost or 0))
    original_total = parts + labor

    if parts == 0:
        return {
            "applied": False,
            "vehicle_type": vehicle_type,
            "vehicle_type_name": spec["name"],
            "useful_life_years": life,
            "vehicle_age_years": round(age, 2),
            "method": method,
            "method_name": "定率遞減法" if method != "straight_line" else "平均法",
            "parts_original": 0,
            "labor_original": labor,
            "original_total": original_total,
            "residual_ratio": 1.0,
            "parts_depreciated": 0,
            "approved_total": labor,
            "deducted_amount": 0,
            "formula_note": "未填寫零件費，工資部分不予折舊。",
        }

    if method == "straight_line":
        # 平均法：殘值 = 原價 / (耐用年數 + 1)
        floor_ratio = 1.0 / (life + 1)
        eff_age = min(age, float(life))
        residual_ratio = 1.0 - (1.0 - floor_ratio) * (eff_age / life)
        residual_ratio = max(floor_ratio, min(1.0, residual_ratio))
        method_name = "平均法"
        note = (
            f"{spec['name']}耐用年數 {life} 年，車齡 {age:.1f} 年，採平均法（殘值下限 1/{life + 1}），"
            f"零件殘值比例 {residual_ratio:.1%}；工資不折舊。"
        )
    else:
        # 定率遞減法（台灣民事車禍實務最主流）：每年折舊率 rate，最後殘值最低為原價 1/10
        floor_ratio = 0.1
        if age >= life:
            residual_ratio = floor_ratio
        else:
            full_years = int(math.floor(age))
            rem_frac = age - full_years
            ratio = (1.0 - rate) ** full_years
            if rem_frac > 1e-6:
                # 實務上未滿 1 年之月數以按月折舊率 (1 - rate * 月/12) 計算
                months = min(12, max(1, int(math.ceil(rem_frac * 12))))
                ratio *= 1.0 - rate * (months / 12.0)
            residual_ratio = max(floor_ratio, min(1.0, ratio))
        method_name = "定率遞減法"
        note = (
            f"{spec['name']}耐用年數 {life} 年（年折舊率 {rate:.3f}），車齡 {age:.1f} 年，"
            f"採定率遞減法（殘值下限 10%），零件殘值比例 {residual_ratio:.1%}；工資不折舊。"
        )

    parts_depreciated = int(round(parts * residual_ratio))
    approved_total = parts_depreciated + labor
    deducted = max(0, original_total - approved_total)

    return {
        "applied": True,
        "vehicle_type": vehicle_type,
        "vehicle_type_name": spec["name"],
        "useful_life_years": life,
        "vehicle_age_years": round(age, 2),
        "method": method,
        "method_name": method_name,
        "parts_original": parts,
        "labor_original": labor,
        "original_total": original_total,
        "residual_ratio": round(residual_ratio, 4),
        "parts_depreciated": parts_depreciated,
        "approved_total": approved_total,
        "deducted_amount": deducted,
        "formula_note": note,
    }


def extract_vehicle_parts_labor(text: str) -> dict[str, Any]:
    """從使用者輸入或上傳文字中，分別擷取修車「零件費」、「工資費」與「車齡／車種」。"""
    t = text or ""
    amount = rf"(?:新[臺台]幣)?\s*([{NUM_CHARS}.．]+)\s*元"
    conn = r"\s*(?:為|共|計|花了|花費)?\s*[:：]?\s*"

    parts_sum = 0
    labor_sum = 0
    for m in re.finditer(rf"零件(?:費用|費)?{conn}{amount}", t):
        v = cn2int(m.group(1))
        if v:
            parts_sum += v
    for m in re.finditer(rf"(?:工資|工錢|工費|烤漆(?:費|費用)?){conn}{amount}", t):
        v = cn2int(m.group(1))
        if v:
            labor_sum += v

    age_years = None
    m_age = re.search(r"(?:車齡|使用(?:期間)?(?:已達|已逾|約|共)?)\s*([0-9０-９]+(?:\.[0-9０-９]+)?)\s*年(?:\s*([0-9０-９]+)\s*(?:個)?月)?", t)
    if m_age:
        try:
            yrs = float(m_age.group(1))
            mos = float(m_age.group(2)) if m_age.group(2) else 0.0
            age_years = round(yrs + mos / 12.0, 2)
        except ValueError:
            age_years = None

    veh_type = None
    if any(k in t for k in ("計程車", "營業小客車", "營業用小客車")):
        veh_type = "commercial_car"
    elif any(k in t for k in ("大貨車", "貨車", "大客車", "聯結車")):
        veh_type = "truck"
    elif any(k in t for k in ("機車", "重機", "機器腳踏車")) and not any(k in t for k in ("原告駕駛自小客車", "原告自小客車")):
        veh_type = "motorcycle"
    elif any(k in t for k in ("自小客車", "小客車", "自用小客車", "轎車", "汽車")):
        veh_type = "car"

    return {
        "claim_vehicle_parts": parts_sum or None,
        "claim_vehicle_labor": labor_sum or None,
        "vehicle_age_years": age_years,
        "vehicle_type": veh_type,
    }
