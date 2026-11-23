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
    "claim_transport": ("交通費", "就醫交通", "計程車資", "計程車費", "計程車車資"),
    "claim_work_loss": ("不能工作損失", "工作損失", "薪資損失", "不能工作之損失", "營業損失", "請假扣薪", "被扣薪", "扣薪", "少賺"),
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
    """同一子句內找項目金額；不跨項目或否定／未知敘述取值。"""
    other_kws = [word for words in CLAIM_ITEMS.values() for word in words
                 if word not in kws]
    # 保留金額千分位中的逗號與小數點。
    clauses = re.split(r'[。；;\n，、]|,(?!\d{3}(?:\D|$))', text or '')
    for clause in clauses:
        matches = sorted(
            (match for word in kws for match in re.finditer(re.escape(word), clause)),
            key=lambda match: (match.start(), -len(match.group())),
        )
        for match in matches:
            before = clause[:match.start()]
            if re.search(r'沒有|並無|未有|無須|不需|未請求|不請求|否認|未發生', before):
                continue
            tail = clause[match.end():match.end() + window]
            boundaries = [tail.find(word) for word in other_kws if word in tail]
            if boundaries:
                tail = tail[:min(boundaries)]
            amount = AMOUNT_RE.search(tail)
            if not amount:
                continue
            context = tail[:amount.start()]
            if re.search(r'不詳|不明|不清楚|未知|未確定|待確認|沒有|不需|無須|不請求|未支出', context):
                continue
            value = cn2int(amount.group(1))
            if value is not None:
                return value
    return None


def _continue_profile_subject(text: str) -> str:
    """僅將緊接單一當事人背景句的月薪句補上主詞。

    換行、分號、其他人物、爭議或插入其他敘述均中止延續。
    只延續一次，不將整段文字都套用前一位當事人。
    """
    pieces = re.split(r'([。；;\n])', text)
    previous_role = None
    for i in range(0, len(pieces), 2):
        sentence = pieces[i]
        compact = re.sub(r'\s+', '', sentence)
        subjects = re.findall(r'原告|被告|(?<![\u4e00-\u9fff])我', compact)
        if not subjects and previous_role and re.match(r'^(?:每月收入|每月薪資|月收入|月薪)', compact):
            pieces[i] = previous_role + sentence
            previous_role = None
            continue
        previous_role = None
        if len(subjects) != 1:
            continue
        if re.search(r'父|母|妻|丈夫|兒子|女兒|配偶|律師|代理人|朋友|證人|主張|抗辯|否認|認為|表示|稱|「|」|"', compact):
            continue
        if not re.search(r'畢業|肄業|學歷|教育程度|從事|擔任|職業|月收入|月薪', compact):
            continue
        if i + 1 < len(pieces) and pieces[i + 1] == '。':
            previous_role = subjects[0]
    return ''.join(pieces)


# ================================================================ 特徵（案情事實）
def extract_vehicle_amount(text: str) -> int | None:
    """擷取同一句的修車明細；明確修車總額優先，避免重複加總。

    跨句明細、多筆報價與折舊計算仍需另外處理。
    """
    amount = rf"(?:新[臺台]幣)?\s*([{NUM_CHARS}.．]+)\s*元"
    connector = r"\s*(?:為|共|計|花了|花費)?\s*[:：]?\s*"
    part_pattern = re.compile(
        rf"(?:零件(?:費用|費)?|工資|工錢|工費){connector}{amount}"
    )
    total_pattern = re.compile(
        rf"(?:車輛修理費|修車費|維修費|修復費用|修車)"
        rf"\s*(?:共計|合計|總計|總共)?{connector}{amount}"
    )
    other_items = tuple(
        kw for field, keywords in CLAIM_ITEMS.items()
        if field != "claim_vehicle" for kw in keywords
    )
    for sentence in re.split(r"[。；;\n]", text or ""):
        repair = re.search(r"修車|修理|維修|修復|車損", sentence)
        if not repair:
            continue
        segment = sentence[repair.start():]
        # 停在下一個賠償項目，避免把整案合計當成修車總額。
        boundaries = [segment.find(kw) for kw in other_items if kw in segment]
        if boundaries:
            segment = segment[:min(boundaries)]
        explicit = total_pattern.search(segment)
        if explicit:
            value = cn2int(explicit.group(1))
            if value is not None:
                return value
        matches = list(part_pattern.finditer(segment))
        values = [cn2int(match.group(1)) for match in matches]
        if not values or any(value is None for value in values):
            continue
        # 只有明細後緊接的合計才視為修車合計。
        total = re.match(
            rf"[\s，,、]*(?:共計|合計|總計|總共)\s*{amount}",
            segment[matches[-1].end():],
        )
        if total:
            value = cn2int(total.group(1))
            if value is not None:
                return value
        return sum(values)
    return None


# 深度欄位暫不加入 NUMERIC_FEATURES：需驗證資料並重訓後才供模型使用。
DEEP_FEATURE_FIELDS = (
    'plaintiff_education', 'defendant_education',
    'plaintiff_occupation', 'defendant_occupation',
    'plaintiff_monthly_income', 'defendant_monthly_income',
    'plaintiff_rest_months', 'plaintiff_labor_loss_ratio',
)


def extract_deep_features(text: str) -> dict[str, Any]:
    """保守擷取明確單一當事人的資料。

    原告／被告或第一人稱「我」為主詞；僅延續緊接背景句的月薪句。
    職業保留原文名稱；不推論薪資、學歷或失能程度。
    多人、否定、爭議、範圍或衝突數值留為 None。
    休養僅接受醫囑且以月表示，不自行將天數換成月份。
    """
    from parsing.numbers import normalize_width

    values = {key: set() for key in DEEP_FEATURE_FIELDS}
    t = _continue_profile_subject(normalize_width(text or ''))
    # 目前的單人欄位無法表達多人分配，整段保守棄權。
    if re.search(r'原告(?:等|二人|兩人|三人)|被告(?:等|二人|兩人|三人)|兩造均|分別|各自', t):
        return {key: None for key in DEEP_FEATURE_FIELDS}

    number = rf'[{NUM_CHARS}.]+'
    edu = r'博士|碩士|研究所|大學|大專|專科|高中|高職|國中|國小|小學'
    jobs = r'設計師|工程師|計程車司機|貨車司機|司機|教師|老師|護理師|護士|醫師|工人|服務生|店員|廚師|公務員|學生|無業|待業|退休'

    def add(key, value):
        if key in values and value is not None:
            values[key].add(value)

    for sentence in re.split(r'[。；;\n]', t):
        sentence = re.sub(r'\s+', '', sentence)
        subjects = list(re.finditer(r'原告|被告|(?<![\u4e00-\u9fff])我', sentence))
        for i, subject in enumerate(subjects):
            role = 'defendant' if subject.group() == '被告' else 'plaintiff'
            end = subjects[i + 1].start() if i + 1 < len(subjects) else len(sentence)
            segment = sentence[subject.end():end]
            # 不把當事人的親屬或代理人資料歸到當事人。
            if re.match(r'(?:之|的)?(?:父|母|妻|夫|子|女|配偶|律師|代理人)', segment):
                continue
            for clause in re.split(r'[，、]|,(?!\d{3}(?:\D|$))', segment):
                if re.search(r'未提供|不詳|不明|否認|爭執|不是|並非|未曾|不需|無須|未經|沒有|無法|不認', clause):
                    continue
                for m in re.finditer(rf'(?:學歷(?:為|是|:)?|教育程度(?:為|是|:)?)({edu})(?!肄業|未畢業)', clause):
                    add(role + '_education', m.group(1))
                for m in re.finditer(rf'({edu})(?:畢業|肄業)', clause):
                    add(role + '_education', m.group(1) + ('肄業' if m.group().endswith('肄業') else ''))
                for m in re.finditer(rf'(?:從事|擔任|職業(?:為|是|:)|現為|是)({jobs})(?:工作|職務)?', clause):
                    add(role + '_occupation', m.group(1))
                # 範圍與約數不轉成精確單一數值。
                if re.search(r'約|至少|以上|以下|至|到|餘|多|不超過|~|～|－|-', clause):
                    continue
                for m in re.finditer(rf'(?:每月收入|每月薪資|月收入|月薪)(?:為|是|:)?(?:新[臺台]幣)?({number})元', clause):
                    add(role + '_monthly_income', cn2int(m.group(1)))
                if role != 'plaintiff':
                    continue
                m = re.search(rf'(?:醫囑|醫師建議|醫生建議|醫生說|醫生叫我)(?:應|需|須|宜|建議)?(?:休養|休息)(?:至少)?({number}|半)(?:個)?月(半)?', clause)
                if m:
                    raw = m.group(1)
                    months = 0.5 if raw == '半' else (float(raw) if re.fullmatch(r'\d+(?:\.\d+)?', raw) else cn2int(raw))
                    if months is not None:
                        add('plaintiff_rest_months', months + (0.5 if m.group(2) else 0))
                m = re.search(r'(?:勞動能力減損|勞動能力減少|減少勞動能力)(?:比例)?(?:為|達|:)?(百分之[零一二三四五六七八九十百\d]+|\d+(?:\.\d+)?%)', clause)
                if m:
                    add('plaintiff_labor_loss_ratio', parse_ratio(m.group(1)))
    return {key: next(iter(items)) if len(items) == 1 else None for key, items in values.items()}


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
    vehicle_amount = extract_vehicle_amount(t)
    if vehicle_amount is not None:
        f["claim_vehicle"] = vehicle_amount
    f["claim_amount"] = extract_claim_amount(t) or 0
    f.update(extract_deep_features(t))
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
    """擷取明確判准的慰撫金；沒有明確結論或金額有歧義時回傳 None。

    本規則處理單句明確結論；跨句、多人或混合費用仍需人工覆核。
    """
    amount = rf"(?:新[臺台]幣)?\s*([{NUM_CHARS}.．]+)\s*元"
    patterns = [
        rf"(?:應)?(?:酌減為|酌定為|核定為|核定|酌定)\s*{amount}",
        rf"(?:應以|以)\s*{amount}\s*(?:為|始為)"
        rf"(?:適當|相當|允當|妥適|合理|適宜|妥當)",
        rf"(?:請求)?\s*{amount}[，,]*(?:核屬相當|應予准許|應予准予)",
    ]
    candidates = set()
    for sentence in re.split(r"[。；;]", court or ""):
        sentence = re.sub(r"\s+", "", sentence)
        if "慰撫金" not in sentence:
            continue
        # 否定或當事人建議不直接作為法院判准標籤。
        if re.search(
            r"不應|不予|不得|不宜|尚非|並非|非屬|無理由|不相當|不予准許"
            r"|(?:原告|被告)(?:主張|認為|辯稱|抗辯|建議)", sentence
        ):
            continue
        start = sentence.find("慰撫金") + len("慰撫金")
        segment = sentence[start:]
        if re.search(
            r"醫療費|醫藥費|看護費|交通費|修車|車損|工作損失|薪資損失"
            r"|另(?:一)?(?:名|位)原告|各|分別", segment
        ):
            continue
        for pattern in patterns:
            matches = list(re.finditer(pattern, segment))
            if matches:
                for match in matches:
                    value = cn2int(match.group(1))
                    if value is not None:
                        candidates.add(value)
                break
    return next(iter(candidates)) if len(candidates) == 1 else None

# ================================================================ 整合
def extract_case(doc: dict) -> dict[str, Any]:
    """doc: 開放資料 JSON（JID/JTITLE/JFULL…）→ 一筆案件紀錄。"""
    from crawler.filters import is_first_instance, parse_jid

    jid = doc.get("JID", "")
    meta = parse_jid(jid)
    sec: Sections = split_sections(doc.get("JFULL", ""))
    facts = sec.facts
    feats = extract_features(facts)
    # 法院判斷中的背景資料獨立保存，避免回填成事前可知的模型輸入。
    court_background = {
        "court_" + key: value
        for key, value in extract_deep_features(sec.court).items()
    }
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
        **court_background,
        "outcome": main["outcome"],
        "awarded_amount": awarded,
        "award_ratio": min(ratio, 1.0) if ratio is not None else None,
        "plaintiff_fault": extract_plaintiff_fault(sec.court),
        "awarded_consolation": extract_awarded_consolation(sec.court),
        "main_text": sec.main,
        "facts_text": facts[:6000],
        "court_text": sec.court[:6000],
    }
