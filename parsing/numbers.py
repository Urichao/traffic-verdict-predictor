"""中文/阿拉伯混合金額與比例的轉換。

    cn2int("壹佰貳拾萬參仟")  -> 1203000
    cn2int("120萬3,000")      -> 1203000
    parse_ratio("三成")        -> 0.3
"""
from __future__ import annotations

import re

_FW = str.maketrans("０１２３４５６７８９，．％：", "0123456789,.%:")
_DIGIT = {"零": 0, "〇": 0, "○": 0, "一": 1, "壹": 1, "二": 2, "貳": 2, "兩": 2, "三": 3, "參": 3, "叁": 3,
          "四": 4, "肆": 4, "五": 5, "伍": 5, "六": 6, "陸": 6, "七": 7, "柒": 7, "八": 8, "捌": 8,
          "九": 9, "玖": 9}
_UNIT = {"十": 10, "拾": 10, "百": 100, "佰": 100, "千": 1000, "仟": 1000}
_BIG = {"萬": 10_000, "万": 10_000, "億": 100_000_000}

NUM_CHARS = "0-9０-９,，零〇○一壹二貳兩三參叁四肆五伍六陸七柒八捌九玖十拾百佰千仟萬万億"
AMOUNT_RE = re.compile(rf"(?:新[臺台]幣)?\s*(?:[（(]下同[)）])?\s*([{NUM_CHARS}]+(?:\.\d+)?)\s*元")


def normalize_width(s: str) -> str:
    return s.translate(_FW)


def cn2int(s: str) -> int | None:
    """把金額字串轉成整數；無法解析回傳 None。"""
    if not s:
        return None
    s = normalize_width(s).replace(",", "").strip()
    if not s:
        return None
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([萬万億])", s)  # 1.5萬
    if m:
        return int(round(float(m.group(1)) * _BIG[m.group(2)]))
    if re.fullmatch(r"\d+", s):
        return int(s)
    total = section = number = 0
    seen = False
    for ch in s:
        if ch.isdigit():
            number = number * 10 + int(ch)
            seen = True
        elif ch in _DIGIT:
            number = _DIGIT[ch]
            seen = True
        elif ch in _UNIT:
            section += (number or 1) * _UNIT[ch]
            number = 0
            seen = True
        elif ch in _BIG:
            total += (section + number) * _BIG[ch]
            section = number = 0
            seen = True
        elif ch == ".":
            return None
        else:
            return None
    return total + section + number if seen else None


def find_amounts(text: str) -> list[tuple[int, int, int]]:
    """回傳 [(金額, 起點, 終點)]。"""
    out = []
    for m in AMOUNT_RE.finditer(text or ""):
        v = cn2int(m.group(1))
        if v is not None and v > 0:
            out.append((v, m.start(), m.end()))
    return out


def _small_cn(s: str) -> int | None:
    s = normalize_width(s)
    if s.isdigit():
        return int(s)
    return cn2int(s)


RATIO_TOKEN = (r"(百分之[零一二三四五六七八九十百\d]+|\d{1,3}(?:\.\d+)?\s*%|[一二三四五六七八九十\d](?:\.\d)?\s*成"
               r"|十分之[一二三四五六七八九\d]|[一二三四五六七八九十\d]+分之[一二三四五六七八九十\d]+)")


def parse_ratio(tok: str) -> float | None:
    """把 '30%'、'百分之三十'、'三成'、'十分之三'、'4分之1' 轉成 0~1。"""
    if not tok:
        return None
    t = normalize_width(tok).replace(" ", "")
    try:
        if t.endswith("%"):
            v = float(t[:-1]) / 100
        elif t.startswith("百分之"):
            v = _small_cn(t[3:]) / 100
        elif t.endswith("成"):
            core = t[:-1]
            v = (float(core) if re.fullmatch(r"\d+(?:\.\d+)?", core) else _small_cn(core)) / 10
        elif "分之" in t:
            den, num = t.split("分之", 1)
            v = _small_cn(num) / _small_cn(den)
        else:
            return None
    except (TypeError, ZeroDivisionError, ValueError):
        return None
    return v if 0 <= v <= 1 else None
