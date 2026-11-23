"""中文／阿拉伯混合金額與比例的轉換。

cn2int("壹佰貳拾萬參仟") -> 1203000
cn2int("120萬3,000")     -> 1203000
cn2int("三萬五")         -> 35000
cn2int("三萬零五")       -> 30005
cn2int("3.5萬")          -> 35000
parse_ratio("三成")      -> 0.3
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

_FW = str.maketrans(
    "０１２３４５６７８９，．％：",
    "0123456789,.%:",
)

_DIGIT = {
    "零": 0, "〇": 0, "○": 0,
    "一": 1, "壹": 1,
    "二": 2, "貳": 2, "兩": 2,
    "三": 3, "參": 3, "叁": 3,
    "四": 4, "肆": 4,
    "五": 5, "伍": 5,
    "六": 6, "陸": 6,
    "七": 7, "柒": 7,
    "八": 8, "捌": 8,
    "九": 9, "玖": 9,
}

_UNIT = {
    "十": 10, "拾": 10,
    "百": 100, "佰": 100,
    "千": 1000, "仟": 1000,
}

_BIG = {
    "萬": 10_000,
    "万": 10_000,
    "億": 100_000_000,
}

NUM_CHARS = (
    "0-9０-９,，"
    "零〇○一壹二貳兩三參叁四肆五伍六陸七柒八捌九玖"
    "十拾百佰千仟萬万億"
)

# 保留第一個捕捉群組為數字，供其他模組使用。
# 小數點必須包含在同一個金額中，避免把「3.5萬元」抓成「5萬元」。
AMOUNT_RE = re.compile(
    rf"(?:新[臺台]幣)?\s*"
    rf"(?:[（(]下同[)）])?\s*"
    rf"([{NUM_CHARS}.．]+)\s*元"
)


def normalize_width(s: str) -> str:
    """將全形數字與部分標點轉為半形。"""
    return s.translate(_FW)


def _parse_integer(s: str) -> int | None:
    """解析含明確單位的整數，不套用口語省略規則。"""
    if not s:
        return None

    # 優先拆大單位，支援「一億二千萬」。
    for unit_text, unit_value in (
        ("億", 100_000_000),
        ("萬", 10_000),
        ("万", 10_000),
    ):
        if unit_text in s:
            if s.count(unit_text) != 1:
                return None

            left, right = s.split(unit_text, 1)
            left_value = _parse_integer(left) if left else 1
            right_value = _parse_integer(right) if right else 0

            if left_value is None or right_value is None:
                return None

            return left_value * unit_value + right_value

    # 沒有十、百、千時，逐位讀取數字。
    if not any(ch in _UNIT for ch in s):
        digits = []
        for ch in s:
            if ch in "0123456789":
                digits.append(ch)
            elif ch in _DIGIT:
                digits.append(str(_DIGIT[ch]))
            else:
                return None

        return int("".join(digits)) if digits else None

    total = 0
    number = None
    last_unit = 10_000

    for ch in s:
        if ch in "0123456789":
            number = (number or 0) * 10 + int(ch)
        elif ch in _DIGIT:
            number = (number or 0) * 10 + _DIGIT[ch]
        elif ch in _UNIT:
            unit = _UNIT[ch]

            # 小單位應依序遞減，例如「三千五百二十」。
            if unit >= last_unit:
                return None

            total += (1 if number is None else number) * unit
            number = None
            last_unit = unit
        else:
            return None

    return total + (number or 0)


def cn2int(s: str) -> int | None:
    """將數字字串轉為整數；無法解析時回傳 None。

    不含「元」字，例如傳入「3.5萬」。
    口語「三萬五」按 35,000 解讀。
    """
    if not s:
        return None

    s = normalize_width(s).replace(",", "").strip()
    if not s:
        return None

    # 阿拉伯數字小數，可接萬、億。
    decimal_match = re.fullmatch(
        r"(\d+(?:\.\d+)?)([萬万億]?)",
        s,
    )
    if decimal_match:
        try:
            value = Decimal(decimal_match.group(1))
            unit = decimal_match.group(2)

            if unit:
                value *= _BIG[unit]

            return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        except InvalidOperation:
            return None

    # 口語省略：
    # 三萬五 → 三萬五千
    # 三千五 → 三千五百
    # 三百五 → 三百五十
    # 「三萬零五」不符合此規則，仍解析為 30,005。
    short = re.fullmatch(
        r"(.+[萬万千仟百佰])([一二兩三四五六七八九1-9])",
        s,
    )
    if short:
        base = _parse_integer(short.group(1))
        digit_text = short.group(2)
        digit = (
            int(digit_text)
            if digit_text in "123456789"
            else _DIGIT[digit_text]
        )
        unit = {
            "萬": 10_000,
            "万": 10_000,
            "千": 1000,
            "仟": 1000,
            "百": 100,
            "佰": 100,
        }[short.group(1)[-1]]

        if base is not None:
            return base + digit * (unit // 10)

    return _parse_integer(s)


def find_amounts(text: str) -> list[tuple[int, int, int]]:
    """回傳 [(金額, 起點, 終點)]；只保留大於零的金額。"""
    out = []

    for match in AMOUNT_RE.finditer(text or ""):
        value = cn2int(match.group(1))
        if value is not None and value > 0:
            out.append((value, match.start(), match.end()))

    return out


def _small_cn(s: str) -> int | None:
    return cn2int(normalize_width(s))


RATIO_TOKEN = (
    r"(百分之[零一二三四五六七八九十百\d]+"
    r"|\d{1,3}(?:\.\d+)?\s*%"
    r"|[一二三四五六七八九十\d](?:\.\d)?\s*成"
    r"|十分之[一二三四五六七八九\d]"
    r"|[一二三四五六七八九十\d]+分之"
    r"[一二三四五六七八九十\d]+)"
)


def parse_ratio(tok: str) -> float | None:
    """將百分比、成數與分數轉成 0～1；無法解析回傳 None。"""
    if not tok:
        return None

    t = re.sub(r"\s+", "", normalize_width(tok))

    try:
        if t.endswith("%"):
            value = float(t[:-1]) / 100

        elif t.startswith("百分之"):
            value = _small_cn(t[3:]) / 100

        elif t.endswith("成"):
            core = t[:-1]
            number = (
                float(core)
                if re.fullmatch(r"\d+(?:\.\d+)?", core)
                else _small_cn(core)
            )
            value = number / 10

        elif "分之" in t:
            denominator, numerator = t.split("分之", 1)
            value = _small_cn(numerator) / _small_cn(denominator)

        else:
            return None

    except (TypeError, ZeroDivisionError, ValueError):
        return None

    return value if 0 <= value <= 1 else None