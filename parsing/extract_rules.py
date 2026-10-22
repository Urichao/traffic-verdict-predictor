"""以規則（Regex＋關鍵字）擷取車禍判決要點。

同一套函式也用來分析「使用者輸入的案情」，確保訓練資料與使用者輸入的特徵一致：
- 特徵（features）只從「案情事實」（原告主張＋被告抗辯）擷取
- 標籤（targets）只從「主文」與「本院之判斷」擷取
"""
from __future__ import annotations

import re
from typing import Any

from parsing.numbers import AMOUNT_RE, NUM_CHARS, RATIO_TOKEN, cn2int, find_amounts, parse_ratio
from parsing.sections import Sections, split_sections

LABELS = ("原告勝訴", "部分勝訴", "原告敗訴")

# ---------------------------------------------------------------- 傷勢
INJURY_LEVELS = [  # (等級, 關鍵字)；取命中的最高等級
    (4, ("死亡", "身故", "不治", "傷重不治", "殯葬費")),
    (3, ("顱內出血", "腦出血", "硬腦膜", "蜘蛛網膜下腔出血", "腦挫傷", "脊髓損傷", "截肢", "植物人", "癱瘓",
         "重傷害", "重大傷病", "氣血胸", "脾臟破裂", "肝臟撕裂")),
    (2, ("骨折", "粉碎性", "脫臼", "韌帶斷裂", "韌帶撕裂", "撕裂傷", "腦震盪", "半月板", "肋骨", "骨裂")),
    (1, ("擦挫傷", "挫傷", "擦傷", "扭傷", "拉傷", "瘀傷", "挫擦傷", "鈍傷", "頸部扭傷")),
]
INJURY_LEVEL_NAMES = {0: "無人傷／僅財損", 1: "輕傷（擦挫傷）", 2: "中度（骨折等）", 3: "重傷", 4: "死亡"}

# ---------------------------------------------------------------- 車種與違規
VEHICLE_KW = {
    "veh_motorcycle": ("機車", "重機", "機器腳踏車"),
    "veh_car": ("自小客車", "小客車", "自用小客車", "汽車", "轎車", "計程車"),
    "veh_truck": ("貨車", "大貨車", "聯結車", "曳引車", "砂石車", "大客車", "公車", "遊覽車"),
    "veh_bicycle": ("腳踏車", "自行車", "單車"),
    "pedestrian": ("行人", "步行", "穿越道路", "斑馬線"),
}
VIOLATION_KW = {
    "v_red_light": ("闖紅燈", "紅燈", "違反號誌"),
    "v_drunk": ("酒後駕車", "酒駕", "酒精濃度", "酒測", "吐氣所含酒精"),
    "v_speeding": ("超速", "超過速限", "速限"),
    "v_no_yield": ("未禮讓", "未讓", "支線道車應暫停讓幹線道", "轉彎車應讓直行車", "應暫停讓"),
    "v_inattention": ("未注意車前狀況", "未注意前方", "疏未注意"),
    "v_wrong_way": ("逆向", "跨越雙黃線", "跨越分向限制線"),
    "v_lane_change": ("變換車道", "未保持安全距離", "未保持行車安全距離", "追撞"),
    "v_turn": ("左轉", "右轉", "迴轉"),
    "v_no_license": ("無照", "未領有駕駛執照", "駕照"),
    "v_no_helmet": ("未戴安全帽", "未配戴安全帽"),
}

# ---------------------------------------------------------------- 請求項目
CLAIM_ITEMS = {
    "claim_medical": ("醫療費用", "醫藥費", "醫療費"),
    "claim_future_medical": ("將來醫療", "後續醫療", "未來醫療"),
    "claim_care": ("看護費", "看護費用", "照護費"),
    "claim_transport": ("交通費", "就醫交通"),
    "claim_work_loss": ("不能工作損失", "工作損失", "薪資損失", "不能工作之損失", "營業損失"),
    "claim_labor_loss": ("勞動能力減損", "減少勞動能力", "勞動能力之損失"),
    "claim_vehicle": ("車輛修理費", "修車費", "車損", "維修費", "修復費用", "車輛損害", "財物損失"),
    "claim_funeral": ("殯葬費", "喪葬費"),
    "claim_support": ("扶養費",),
    "claim_consolation": ("精神慰撫金", "慰撫金", "精神上損害", "非財產上損害"),
}

# 數字特徵（給模型與前端用）
NUMERIC_FEATURES = (["claim_amount", "injury_level", "hospital_days", "surgery", "disability", "death",
                     "has_criminal", "has_appraisal", "defendant_absent", "claims_contributory",
                     "insurance_paid", "settled_or_no_fault"]
                    + list(CLAIM_ITEMS) + list(VEHICLE_KW) + list(VIOLATION_KW))


def _has(text: str, kws) -> bool:
    return any(k in text for k in kws)


def _amount_after(text: str, kws, window: int = 40) -> int | None:
    """找關鍵字後 window 字內第一個金額（取全文中最早出現者）。"""
    hits = []
    for k in kws:
        for m in re.finditer(re.escape(k), text):
            am = AMOUNT_RE.search(text[m.end(): m.end() + window])
            v = cn2int(am.group(1)) if am else None
            if v:
                hits.append((m.start(), v))
                break
    return min(hits)[1] if hits else None


# ================================================================ 特徵（案情事實）
def extract_features(facts: str) -> dict[str, Any]:
    t = facts or ""
    f: dict[str, Any] = {}
    level, hits = 0, []
    for lv, kws in INJURY_LEVELS:
        found = [k for k in kws if k in t]
        if found:
            hits += found
            level = max(level, lv)
    f["injury_level"] = level
    f["injury_terms"] = sorted(set(hits))
    m = re.search(r"住院(?:治療)?(?:共|計|達)?\s*([0-9０-９]+|[一二三四五六七八九十百]+)\s*(?:日|天)", t)
    f["hospital_days"] = (cn2int(m.group(1)) or 0) if m else 0
    f["surgery"] = int(_has(t, ("手術", "開刀", "內固定")))
    f["disability"] = int(_has(t, ("勞動能力減損", "減少勞動能力", "失能", "殘廢")))
    f["death"] = int(level == 4)
    f["has_criminal"] = int(_has(t, ("刑事判決", "過失傷害", "過失致死", "刑事庭", "附帶民事")))
    f["has_appraisal"] = int(_has(t, ("鑑定意見", "鑑定委員會", "覆議", "肇事責任鑑定", "行車事故鑑定")))
    f["defendant_absent"] = int(_has(t, ("未於言詞辯論期日到場", "一造辯論", "未提出書狀作何聲明",
                                         "未到庭", "未提出任何書狀")))
    f["claims_contributory"] = int(_has(t, ("與有過失", "過失相抵", "亦有過失")))
    f["insurance_paid"] = int(_has(t, ("強制汽車責任保險", "強制險", "保險金", "理賠")))
    f["settled_or_no_fault"] = int(_has(t, ("達成和解", "已和解", "和解書", "已賠償", "無肇事因素", "無法證明")))
    for k, kws in VEHICLE_KW.items():
        f[k] = int(_has(t, kws))
    for k, kws in VIOLATION_KW.items():
        f[k] = int(_has(t, kws))
    for k, kws in CLAIM_ITEMS.items():
        f[k] = _amount_after(t, kws) or 0
    f["claim_amount"] = extract_claim_amount(t) or 0
    return f


_PAY_RE = re.compile(rf"應(?:連帶|各|共同|不真正連帶)?給付(?:原告)?[^。；]{{0,20}}?(?:新[臺台]幣)?\s*(?:[（(]下同[)）])?\s*"
                     rf"([{NUM_CHARS}]+)\s*元")


def extract_claim_amount(text: str) -> int | None:
    """原告請求總額：優先取「聲明」附近（若有減縮／擴張取最後一次）。"""
    cands = []
    for m in _PAY_RE.finditer(text or ""):
        v = cn2int(m.group(1))
        if not v:
            continue
        ctx = text[max(0, m.start() - 40): m.start()]
        score = 2 if ("減縮" in ctx or "擴張" in ctx or "變更" in ctx) else 1 if "聲明" in ctx else 0
        cands.append((score, m.start(), v))
    if cands:
        top = max(c[0] for c in cands)
        best = [c for c in cands if c[0] == top]
        return best[-1][2] if top == 2 else best[0][2]
    m = re.search(rf"(?:共計|合計|總計|合計請求)\s*(?:新[臺台]幣)?\s*([{NUM_CHARS}]+)\s*元", text or "")
    return cn2int(m.group(1)) if m else None


# ================================================================ 標籤（主文、法院判斷）
_SKIP_MAIN = ("訴訟費用", "擔保", "假執行", "裁判費")


def parse_main(main: str) -> dict[str, Any]:
    """解析主文 → 判決結果與判賠金額。"""
    m = main or ""
    sents = [s for s in re.split(r"[。；]|(?=[一二三四五六七八九十]+、)", m) if s]
    awarded = 0
    has_pay = False
    for s in sents:
        if any(k in s for k in _SKIP_MAIN) or "給付" not in s:
            continue
        has_pay = True
        s = re.split(r"及自|，及|及其中|及均自|暨自", s)[0]
        awarded += sum(v for v, _, _ in find_amounts(s))
    rejected_rest = bool(re.search(r"(其餘之訴|其餘請求|其餘部分|其他之訴)[^。]{0,10}駁回", m))
    rejected_all = bool(re.search(r"原告之訴(?:及其?假執行之聲請)?(?:均)?駁回", m))
    if has_pay and awarded > 0:
        outcome = "部分勝訴" if rejected_rest else "原告勝訴"
    elif rejected_all or (rejected_rest and not has_pay):
        outcome = "原告敗訴"
    else:
        outcome = None
    return {"outcome": outcome, "awarded_amount": awarded if has_pay else 0}


def extract_plaintiff_fault(court: str) -> float | None:
    """法院認定的原告（被害人）過失比例 0~1；未提及與有過失回傳 0；無法判斷回傳 None。"""
    t = court or ""
    R = RATIO_TOKEN
    pats = [
        rf"原告(?:應)?(?:負擔|負|承擔)(?:之)?(?:過失|肇事)?(?:責任)?(?:比例)?(?:為|約|應為)?\s*{R}",
        rf"(?:過失|肇事)(?:責任)?比例[^。]{{0,20}}?原告[^。，]{{0,6}}?{R}",
        rf"減輕被告[^。]{{0,15}}?{R}",
        rf"原告[^。]{{0,10}}?與有過失[^。]{{0,30}}?{R}",
    ]
    for p in pats:
        mm = re.search(p, t)
        if mm:
            v = parse_ratio(mm.group(1))
            if v is not None:
                return v
    mm = re.search(r"原告與被告[^。]{0,15}?([0-9０-９])\s*[:：比]\s*([0-9０-９])", t)
    if mm:
        a, b = int(mm.group(1).translate(str.maketrans("０１２３４５６７８９", "0123456789"))), \
            int(mm.group(2).translate(str.maketrans("０１２３４５６７８９", "0123456789")))
        if a + b:
            return a / (a + b)
    mm = re.search(rf"被告(?:應)?(?:負擔|負)[^。]{{0,6}}?{R}[^。]{{0,6}}?(?:過失|肇事)責任", t)
    if mm and "與有過失" in t:
        v = parse_ratio(mm.group(1))
        if v is not None:
            return round(1 - v, 4)
    if "與有過失" in t or "過失相抵" in t:
        if re.search(r"原告[^。]{0,20}(?:並無|無)(?:與有)?過失|無過失相抵", t):
            return 0.0
        return None
    return 0.0


def extract_awarded_consolation(court: str) -> int | None:
    t = court or ""
    for m in re.finditer(r"(?:精神)?慰撫金", t):
        seg = t[m.start(): m.start() + 120]
        mm = re.search(rf"(?:以|核定為?|酌定為?|應以|准許)\s*(?:新[臺台]幣)?\s*([0-9０-９,，零一二三四五六七八九十百千萬]+)\s*元\s*(?:為|始為)?(?:適當|相當|允當|妥適|合理|適宜|妥當)", seg)
        if mm:
            return cn2int(mm.group(1))
    return None


# ================================================================ 整合
def extract_case(doc: dict) -> dict[str, Any]:
    """doc: 開放資料 JSON（JID/JTITLE/JFULL…）→ 一筆案件紀錄。"""
    from crawler.filters import is_first_instance, parse_jid

    jid = doc.get("JID", "")
    meta = parse_jid(jid)
    sec: Sections = split_sections(doc.get("JFULL", ""))
    facts = sec.facts
    feats = extract_features(facts)
    # 請求金額以完整事實及理由搜尋較穩定（聲明段可能在原告主張之前）
    feats["claim_amount"] = extract_claim_amount(sec.body[: max(len(sec.body) - len(sec.court), 2000)]) \
        or feats["claim_amount"]
    main = parse_main(sec.main)
    claim, awarded = feats["claim_amount"], main["awarded_amount"]
    if main["outcome"] == "原告勝訴" and awarded > 0 and not claim:
        claim = awarded
        feats["claim_amount"] = claim
    ratio = None
    if claim and main["outcome"]:
        ratio = 0.0 if main["outcome"] == "原告敗訴" else awarded / claim
        if ratio > 1.05:  # 擷取錯誤（多位原告、多筆請求）→ 不作為迴歸標籤
            ratio = None
    return {
        "jid": jid,
        "court_code": meta["court_code"],
        "court_prefix": meta["court_code"][:3],
        "case_type": meta["case_type"],
        "year": meta["year"],
        "date": meta["date"],
        "first_instance": is_first_instance(meta["case_type"]),
        "title": doc.get("JTITLE", ""),
        **{k: v for k, v in feats.items()},
        "outcome": main["outcome"],
        "awarded_amount": awarded,
        "award_ratio": min(ratio, 1.0) if ratio is not None else None,
        "plaintiff_fault": extract_plaintiff_fault(sec.court),
        "awarded_consolation": extract_awarded_consolation(sec.court),
        "main_text": sec.main,
        "facts_text": facts[:6000],
        "court_text": sec.court[:6000],
    }
