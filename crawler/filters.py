"""判斷一份裁判書是否為「民事車禍損害賠償判決」。"""
from __future__ import annotations

import re

# JID 形如 "TPDV,112,訴,1234,20231130,1"：第一段法院代碼最後一碼 V=民事、M=刑事、A=行政
CIVIL_COURT_RE = re.compile(r"^[A-Z]{3}V$")
# 上訴、抗告、再審等非第一審案件（簡上、小上、上易、重上、抗、再…）
APPEAL_CASE_RE = re.compile(r"上|抗|再|聲|救|調|促|司|全|執|消債|家")
TRAFFIC_TITLE_RE = re.compile(r"交通|車禍")
TRAFFIC_TEXT_RE = re.compile(r"車禍|交通事故|肇事|騎乘|駕駛|機車|自小客車|小客車|汽車|貨車|行人|路口")
DAMAGE_TITLE_RE = re.compile(r"損害賠償|侵權行為")


def parse_jid(jid: str) -> dict:
    parts = jid.split(",")
    if len(parts) < 5:
        return {"court_code": "", "year": None, "case_type": "", "case_no": "", "date": ""}
    return {
        "court_code": parts[0],
        "year": int(parts[1]) if parts[1].isdigit() else None,
        "case_type": parts[2],
        "case_no": parts[3],
        "date": parts[4],
    }


def is_civil_jid(jid: str) -> bool:
    return bool(CIVIL_COURT_RE.match(parse_jid(jid)["court_code"]))


def is_first_instance(case_type: str) -> bool:
    return not APPEAL_CASE_RE.search(case_type or "")


def is_traffic_damage_judgment(jid: str, title: str, full: str) -> bool:
    """民事 + 判決（非裁定）+ 損害賠償 + 與車禍相關。"""
    if not is_civil_jid(jid):
        return False
    title = title or ""
    if not DAMAGE_TITLE_RE.search(title):
        return False
    head = (full or "")[:300].replace(" ", "").replace("\u3000", "")
    if "判決" not in head:  # 排除裁定（移送、駁回起訴等程序性裁定）
        return False
    if TRAFFIC_TITLE_RE.search(title):  # 例如「損害賠償(交通)」
        return True
    hits = len(set(TRAFFIC_TEXT_RE.findall(full or "")))
    return hits >= 3
