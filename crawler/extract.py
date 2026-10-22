"""解壓每月 RAR 或掃描已解壓資料夾，只保留第一審民事車禍損害賠償判決，輸出成 data/interim/{月份}.jsonl。

用法：
    python -m crawler.extract data/raw/202401.rar [--delete]
    python -m crawler.extract --all            # 處理 data/raw 下所有壓縮檔
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from crawler.filters import is_traffic_damage_judgment  # noqa: E402

KEEP_FIELDS = ("JID", "JYEAR", "JCASE", "JNO", "JDATE", "JTITLE", "JFULL")
SKIP_JCASE_RE = re.compile(r"上|抗|再|聲|救|調|促|司|全|執|消債|家|補|移|選|仲|勞|國|智|商")


def _read_json(path: Path) -> dict | None:
    for enc in ("utf-8-sig", "utf-8", "cp950"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return None


def _iter_candidate_json_files(src: Path, skip_jids: set[str] | None = None):
    """優先只掃描第一審民事子資料夾與第一審訴訟字別檔名，並直接跳過已訓練過的案號 (skip_jids)。"""
    subdirs = [d for d in src.iterdir() if d.is_dir()]
    civil_subdirs = [d for d in subdirs if "民事" in d.name and "高等" not in d.name and "最高" not in d.name]
    search_roots = civil_subdirs if civil_subdirs else [src]
    for root in search_roots:
        for p in root.rglob("*.json"):
            if skip_jids and p.stem in skip_jids:
                continue
            parts = p.stem.split(",")
            if len(parts) >= 3:
                if not parts[0].endswith("V") or SKIP_JCASE_RE.search(parts[2]):
                    continue
            yield p


def filter_dir(src: Path, out_path: Path, skip_jids: set[str] | None = None) -> tuple[int, int]:
    seen = kept = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for p in _iter_candidate_json_files(src, skip_jids=skip_jids):
            seen += 1
            d = _read_json(p)
            if not d:
                continue
            jid = d.get("JID", "")
            if skip_jids and jid in skip_jids:
                continue
            if is_traffic_damage_judgment(jid, d.get("JTITLE", ""), d.get("JFULL", "")):
                out.write(json.dumps({k: d.get(k) for k in KEEP_FIELDS}, ensure_ascii=False) + "\n")
                kept += 1
    return seen, kept


def process_archive(archive: Path, delete_archive: bool = False) -> Path:
    archive = Path(archive)
    out_path = config.INTERIM_DIR / f"{archive.stem}.jsonl"
    if out_path.exists():
        print(f"[skip] {out_path.name} 已存在")
        return out_path
    tmp = Path(tempfile.mkdtemp(prefix="jud_", dir=config.RAW_DIR))
    try:
        cmd = [config.SEVEN_ZIP, "x", str(archive), f"-o{tmp}", "-y", "-r", "*V,*.json"]
        print("解壓中：", archive.name)
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        seen, kept = filter_dir(tmp, out_path)
        print(f"{archive.name}: 民事第一審候選 {seen} 份 → 車禍損害賠償 {kept} 份")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if delete_archive:
        archive.unlink(missing_ok=True)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archives", nargs="*", type=Path)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--delete", action="store_true", help="處理完刪除壓縮檔")
    a = ap.parse_args(argv)
    targets = a.archives or (sorted(p for p in config.RAW_DIR.iterdir()
                                    if p.suffix.lower() in (".rar", ".zip", ".7z")) if a.all else [])
    if not targets:
        ap.error("請指定壓縮檔或使用 --all")
    for t in targets:
        process_archive(t, a.delete)


if __name__ == "__main__":
    main()
