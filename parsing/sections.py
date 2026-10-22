"""判決書段落切分。

裁判書全文 JFULL 為固定寬度斷行的純文字，結構大致為：
    案號／當事人 → 主文 → 事實及理由（原告主張、被告抗辯、本院之判斷）→ 日期、法官、書記官
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_WS = re.compile(r"[\s\u3000]+")
MAIN_HEADINGS = {"主文"}
BODY_HEADINGS = {"事實及理由", "事實與理由", "事實", "理由", "理由要領", "事實要領", "事實及理由要領", "判決要領", "犯罪事實及理由"}

PLAINTIFF_MARKERS = ("原告起訴主張", "原告主張", "原告起訴意旨", "原告起訴略以", "原告之主張", "原告方面",
                     "原告則以", "原告陳述")
DEFENDANT_MARKERS = ("被告則以", "被告抗辯", "被告答辯", "被告辯稱", "被告則辯稱", "被告方面", "被告則抗辯",
                     "被告經合法通知", "被告未於言詞辯論期日到場", "被告未到庭")
COURT_MARKERS = ("本院之判斷", "本院判斷", "得心證之理由", "得心證理由", "本院得心證", "兩造不爭執",
                 "不爭執事項", "法院之判斷", "爭點")
FOOTER_RE = re.compile(r"中\s*華\s*民\s*國\s*[0-9０-９一二三四五六七八九十百○〇零]+\s*年")


@dataclass
class Sections:
    header: str = ""
    main: str = ""        # 主文
    body: str = ""        # 事實及理由全文
    plaintiff: str = ""   # 原告主張
    defendant: str = ""   # 被告抗辯
    court: str = ""       # 本院之判斷
    extra: dict = field(default_factory=dict)

    @property
    def facts(self) -> str:
        """模型輸入用的「案情事實」：原告主張＋被告抗辯，不含法院判斷，避免答案洩漏。"""
        t = (self.plaintiff + self.defendant).strip()
        return t or self.body[: max(len(self.body) // 2, 1)]


def squash(s: str) -> str:
    return _WS.sub("", s or "")


def _first(text: str, markers, start: int = 0) -> int:
    idx = [i for i in (text.find(m, start) for m in markers) if i >= 0]
    return min(idx) if idx else -1


def split_sections(full: str) -> Sections:
    lines = (full or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    sec = Sections()
    buf = {"header": [], "main": [], "body": []}
    cur = "header"
    for ln in lines:
        k = squash(ln)
        if not k:
            continue
        k_head = k.rstrip(":：")
        if k_head in MAIN_HEADINGS and cur == "header":
            cur = "main"
            continue
        if k_head in BODY_HEADINGS and cur in ("header", "main"):
            cur = "body"
            continue
        buf[cur].append(k)
    sec.header = "".join(buf["header"])
    sec.main = "".join(buf["main"])
    body = "".join(buf["body"])
    m = FOOTER_RE.search(body)
    if m and m.start() > len(body) * 0.5:
        body = body[: m.start()]
    sec.body = body

    p = _first(body, PLAINTIFF_MARKERS)
    d = _first(body, DEFENDANT_MARKERS, max(p, 0))
    c = _first(body, COURT_MARKERS, max(d, p, 0))
    p0 = p if p >= 0 else 0
    if d >= 0:
        sec.plaintiff = body[p0:d]
        sec.defendant = body[d:c] if c > d else body[d:d + 1500]
    elif c >= 0:
        sec.plaintiff = body[p0:c]
    else:
        sec.plaintiff = body[p0: p0 + len(body) // 2]
    sec.court = body[c:] if c >= 0 else body[len(body) // 2:]
    return sec
