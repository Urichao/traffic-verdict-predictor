"""司法院「裁判書開放 API」增量抓取（data.judicial.gov.tw/jdg/api）。

- 服務時間：每日 00:00–06:00（其他時間會回「目前非本 API 服務時間」）
- JList 只提供近 7 日異動的裁判書 JID，適合每天排程補抓最新判決；歷史資料請用 crawler.opendata 下載 RAR。
- 帳密放在 .env：JUDICIAL_USER / JUDICIAL_PASSWORD

用法：
    python -m crawler.jdg_api              # 抓近 7 日的民事車禍判決 → data/interim/api_YYYYMMDD.jsonl
    python -m crawler.jdg_api --limit 200  # 測試用，最多抓 200 份民事第一審判決
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
import urllib3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from crawler.filters import is_civil_jid, is_first_instance, is_traffic_damage_judgment, parse_jid  # noqa: E402

urllib3.disable_warnings()


class JdgApi:
    def __init__(self, user: str = config.JUDICIAL_USER, password: str = config.JUDICIAL_PASSWORD):
        if not user or not password:
            raise ValueError("請在 .env 設定 JUDICIAL_USER / JUDICIAL_PASSWORD")
        self.s = requests.Session()
        self.s.verify = False
        self.user, self.password = user, password
        self.token = None

    def _post(self, path: str, body: dict):
        r = self.s.post(f"{config.JDG_API_BASE}/{path}", json=body, timeout=60)
        r.raise_for_status()
        d = r.json()
        if isinstance(d, dict) and d.get("error"):
            raise RuntimeError(d["error"])
        return d

    def auth(self) -> str:
        d = self._post("Auth", {"user": self.user, "password": self.password})
        self.token = d.get("Token") or d.get("token")
        return self.token

    def jlist(self) -> list[str]:
        d = self._post("JList", {"token": self.token})
        jids = []
        for day in d:
            jids += day.get("LIST") or day.get("list") or []
        return jids

    def jdoc(self, jid: str) -> dict:
        d = self._post("JDoc", {"token": self.token, "j": jid})
        fx = d.get("JFULLX") or {}
        return {
            "JID": d.get("JID", jid), "JYEAR": d.get("JYEAR"), "JCASE": d.get("JCASE"), "JNO": d.get("JNO"),
            "JDATE": d.get("JDATE"), "JTITLE": d.get("JTITLE"),
            "JFULL": fx.get("JFULLCONTENT") or d.get("JFULL") or "",
        }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="最多查詢幾份民事第一審 JDoc（0=不限）")
    ap.add_argument("--sleep", type=float, default=0.2, help="每次請求間隔秒數，避免造成伺服器負擔")
    a = ap.parse_args(argv)

    api = JdgApi()
    api.auth()
    jids = [j for j in api.jlist() if is_civil_jid(j) and is_first_instance(parse_jid(j)["case_type"])]
    print(f"近 7 日民事第一審裁判 {len(jids)} 份")
    out = config.INTERIM_DIR / f"api_{datetime.now():%Y%m%d}.jsonl"
    seen_file = config.INTERIM_DIR / "api_seen_jids.txt"
    done = set()
    if seen_file.exists():
        done = {x.strip() for x in seen_file.read_text(encoding="utf-8").splitlines() if x.strip()}
    if out.exists():
        done |= {json.loads(l)["JID"] for l in out.open(encoding="utf-8") if l.strip()}
    jids = [j for j in jids if j not in done]
    print(f"扣除已查詢案號後，尚有 {len(jids)} 份待查詢")
    if a.limit:
        jids = jids[: a.limit]
    kept = 0
    with open(out, "a", encoding="utf-8") as f, open(seen_file, "a", encoding="utf-8") as sf:
        for i, jid in enumerate(jids, 1):
            try:
                d = api.jdoc(jid)
                sf.write(jid + "\n")
                sf.flush()
            except Exception as e:
                print(f"[{i}] {jid} 失敗：{e}")
                if "服務時間" in str(e):
                    break
                time.sleep(2)
                continue
            if is_traffic_damage_judgment(d["JID"], d["JTITLE"] or "", d["JFULL"]):
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                kept += 1
            if i % 50 == 0:
                print(f"進度 {i}/{len(jids)}，車禍判決 {kept} 份")
            time.sleep(a.sleep)
    print(f"完成：新增 {kept} 份 → {out}")


if __name__ == "__main__":
    main()
