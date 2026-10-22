"""司法院資料開放平台（opendata.judicial.gov.tw）每月裁判書 RAR 下載器。

用法：
    python -m crawler.opendata list                 # 列出可下載的月份
    python -m crawler.opendata download --months 24 # 下載最近 24 個月
    python -m crawler.opendata download --only 202401 202402

注意：
- 裁判書 RAR 需要會員登入才能下載，而登入頁有 Cloudflare Turnstile 人機驗證，
  因此本程式「不」自動登入，而是使用你在瀏覽器登入後取得的 token（.env 的 JUDICIAL_TOKEN）。
- 每個 RAR 檔有數 GB，支援中斷續傳（.part 檔）。
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import requests
import urllib3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

urllib3.disable_warnings()  # 平台憑證鏈不完整，requests 預設驗證會失敗
LIST_URL = f"{config.OPENDATA_BASE}/data/api/rest/categories/051/resources"
FILE_URL = f"{config.OPENDATA_BASE}/api/FilesetLists/{{fid}}/file"
MONTH_RE = re.compile(r"^(\d{6})裁判書")


@dataclass
class MonthFile:
    month: str       # "202401"
    file_set_id: int
    title: str
    fmt: str


def session() -> requests.Session:
    s = requests.Session()
    s.verify = False
    s.headers["User-Agent"] = "Mozilla/5.0 (capstone research crawler)"
    if config.JUDICIAL_TOKEN:
        tok = config.JUDICIAL_TOKEN.strip()
        s.headers["Authorization"] = tok if tok.lower().startswith("bearer ") else f"Bearer {tok}"
    return s


def list_months(s: requests.Session | None = None) -> list[MonthFile]:
    s = s or session()
    items = s.get(LIST_URL, timeout=60).json()
    out = []
    for d in items:
        m = MONTH_RE.match(d["title"])
        if not m:
            continue
        for f in d["filesets"]:
            if f["resourceFormat"].upper() in ("RAR", "ZIP", "7Z"):
                out.append(MonthFile(m.group(1), f["fileSetId"], d["title"], f["resourceFormat"].lower()))
    out.sort(key=lambda x: x.month)
    return out


def download(mf: MonthFile, dest_dir: Path = config.RAW_DIR, s: requests.Session | None = None,
             retries: int = 5) -> Path:
    s = s or session()
    dest = dest_dir / f"{mf.month}.{mf.fmt}"
    if dest.exists():
        print(f"[skip] {dest.name} 已存在")
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    url = FILE_URL.format(fid=mf.file_set_id)
    for attempt in range(1, retries + 1):
        done = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={done}-"} if done else {}
        try:
            with s.get(url, headers=headers, stream=True, timeout=120) as r:
                ctype = r.headers.get("content-type", "")
                if r.status_code >= 400 or "json" in ctype:
                    msg = r.text[:200]
                    if "not found" in msg.lower() and not config.JUDICIAL_TOKEN:
                        raise PermissionError("伺服器拒絕下載：請先在 .env 設定 JUDICIAL_TOKEN（瀏覽器登入後複製）")
                    raise RuntimeError(f"HTTP {r.status_code}: {msg}")
                if done and r.status_code != 206:  # 伺服器不支援續傳 → 重頭下載
                    done = 0
                total = int(r.headers.get("content-length", 0)) + done
                mode = "ab" if done else "wb"
                t0, got = time.time(), done
                with open(part, mode) as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        got += len(chunk)
                        if total:
                            speed = (got - done) / max(time.time() - t0, 1e-6) / 1e6
                            print(f"\r{dest.name}: {got/1e9:.2f}/{total/1e9:.2f} GB  {speed:.1f} MB/s", end="")
                print()
            part.rename(dest)
            return dest
        except PermissionError:
            raise
        except Exception as e:  # 網路中斷 → 等待後續傳
            print(f"\n[retry {attempt}/{retries}] {e}")
            time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f"下載失敗：{mf.month}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    d = sub.add_parser("download")
    d.add_argument("--months", type=int, default=24, help="下載最近 N 個月")
    d.add_argument("--only", nargs="*", help="指定月份，如 202401 202402")
    d.add_argument("--extract", action="store_true", help="下載後立即篩選並刪除 RAR 以節省空間")
    a = ap.parse_args(argv)

    s = session()
    months = list_months(s)
    if a.cmd == "list":
        for m in months:
            print(m.month, m.file_set_id, m.title)
        print(f"共 {len(months)} 個月份；JUDICIAL_TOKEN {'已設定' if config.JUDICIAL_TOKEN else '未設定'}")
        return
    targets = [m for m in months if m.month in set(a.only)] if a.only else months[-a.months:]
    for m in targets:
        path = download(m, s=s)
        if a.extract:
            from crawler.extract import process_archive
            process_archive(path, delete_archive=True)


if __name__ == "__main__":
    main()
