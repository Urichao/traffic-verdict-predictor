r"""一鍵匯入新判決資料、自動排除已訓練案件、篩選擷取、重訓模型並熱更新系統。

支援匯入任何格式的新資料：
1. 組員擷取的 CSV 檔（如 traffic_dataset_v2.csv）
2. 司法院開放資料平台下載的每月壓縮檔（.rar / .zip / .7z）
3. 解壓後的司法院裁判書 .json 資料夾或 .jsonl 檔
4. 凌晨 00:00–06:00 直接連線司法院 API 抓取最新判決（--from-api）

防重複機制：
- 匯入前先讀取 `cases.parquet` 中所有已訓練過的案號（`jid`），在掃描資料夾或讀取 CSV/JSONL 時直接略過已存在的案號，不重複讀取、不重複參與訓練。

用法範例：
    .\.venv\Scripts\python ingest.py "C:\path\to\new_data.csv"
    .\.venv\Scripts\python ingest.py "C:\path\to\202608.rar"
    .\.venv\Scripts\python ingest.py "C:\path\to\json_folder"
    .\.venv\Scripts\python ingest.py --from-api
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd
import requests

import config
from crawler.extract import filter_dir, process_archive
from model.train import train
from parsing.build_dataset import iter_docs
from parsing.clean_csv import clean_traffic_csv
from parsing.extract_rules import extract_case
from retrieval.index import build_index


def get_trained_jids() -> set[str]:
    """取得目前已存在於 cases.parquet 中的所有案號，並排除早期測試用合成樣本。"""
    if not config.CASES_PATH.exists():
        return set()
    df = pd.read_parquet(config.CASES_PATH, columns=["jid"])
    return set(df["jid"].dropna().astype(str))


def get_seed_jids() -> set[str]:
    seed_file = config.INTERIM_DIR / "seed_traffic_judgments.jsonl"
    if not seed_file.exists():
        return set()
    return {json.loads(line)["JID"] for line in seed_file.open(encoding="utf-8") if line.strip()}


def _rows_from_jsonl(jsonl_path: Path, skip_jids: set[str]) -> pd.DataFrame:
    rows = []
    skipped = 0
    for doc in iter_docs([jsonl_path]):
        jid = doc.get("JID", "")
        if jid in skip_jids:
            skipped += 1
            continue
        try:
            r = extract_case(doc)
            if isinstance(r.get("injury_terms"), list):
                r["injury_terms"] = "、".join(r["injury_terms"])
            if r.get("first_instance") and r.get("outcome") in ("原告勝訴", "部分勝訴", "原告敗訴"):
                rows.append(r)
                skip_jids.add(jid)
        except Exception as e:
            print(f"[略過] {jid}: {e}")
    if skipped:
        print(f"  -> 已自動排除 {skipped} 筆先前已訓練過的重複案件")
    return pd.DataFrame(rows)


def ingest_path(path: Path, skip_jids: set[str]) -> pd.DataFrame:
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"找不到檔案或目錄：{path}")

    # 1. CSV 檔（如 traffic_dataset_v1.csv / v2.csv）
    if path.is_file() and path.suffix.lower() == ".csv":
        out_clean = path.with_name(f"{path.stem}_cleaned.csv")
        out_pers = path.with_name(f"{path.stem}_personal_injury.csv")
        _, pq_df = clean_traffic_csv(path, out_clean, out_pers)
        before = len(pq_df)
        pq_df = pq_df[~pq_df["jid"].isin(skip_jids)].reset_index(drop=True)
        skipped = before - len(pq_df)
        if skipped:
            print(f"  -> CSV 中有 {skipped} 筆已存在於訓練集，已自動排除（淨新增 {len(pq_df)} 筆）")
        skip_jids.update(pq_df["jid"].astype(str))
        return pq_df

    # 2. RAR / ZIP / 7Z 壓縮檔
    if path.is_file() and path.suffix.lower() in (".rar", ".zip", ".7z"):
        jsonl_path = process_archive(path, delete_archive=False)
        return _rows_from_jsonl(jsonl_path, skip_jids)

    # 3. JSONL 檔
    if path.is_file() and path.suffix.lower() == ".jsonl":
        return _rows_from_jsonl(path, skip_jids)

    # 4. 含多個司法院 .json 檔的資料夾
    if path.is_dir():
        print(f"\n正在掃描資料夾：{path.name}（自動跳過已訓練過的 {len(skip_jids)} 筆案號）...")
        tmp_jsonl = Path(tempfile.mktemp(suffix=".jsonl", dir=config.INTERIM_DIR))
        seen, kept = filter_dir(path, tmp_jsonl, skip_jids=skip_jids)
        print(f"{path.name} 掃描完成：新掃描第一審民事候選 {seen} 份 → 新增車禍判決 {kept} 筆")
        df = _rows_from_jsonl(tmp_jsonl, skip_jids)
        tmp_jsonl.unlink(missing_ok=True)
        return df

    raise ValueError(f"不支援的檔案格式：{path}")


def merge_and_retrain(new_dfs: list[pd.DataFrame], reload_server: bool = True):
    valid_dfs = [d for d in new_dfs if d is not None and not d.empty]
    seed_jids = get_seed_jids()

    if config.CASES_PATH.exists():
        old_df = pd.read_parquet(config.CASES_PATH)
        if seed_jids:
            old_df = old_df[~old_df["jid"].isin(seed_jids)].reset_index(drop=True)
    else:
        old_df = pd.DataFrame()

    before_cnt = len(old_df)
    if not valid_dfs and before_cnt > 0:
        print("\n所有輸入案件皆已在訓練集中（新增 0 筆），無須重新訓練。")
        return

    merged = pd.concat([old_df] + valid_dfs, ignore_index=True) if not old_df.empty else pd.concat(valid_dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=["jid"], keep="first").reset_index(drop=True)
    merged.to_parquet(config.CASES_PATH, index=False)
    added = len(merged) - before_cnt
    print(f"\n[資料庫更新] 原有真實判決 {before_cnt} 筆 → 現有 {len(merged)} 筆（本次淨新增 {added} 筆，已嚴格排除重複案號）")

    print("\n[重新訓練模型]")
    train(config.CASES_PATH)
    print("\n[重建相似案例索引]")
    build_index(config.CASES_PATH)

    if reload_server:
        try:
            r = requests.post("http://127.0.0.1:8000/api/reload", timeout=10)
            if r.ok:
                print(f"[線上系統熱更新成功] 目前線上檢索庫筆數：{r.json().get('indexed_cases')} 筆")
        except Exception:
            pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", type=Path, help="要匯入的 .csv / .rar / .zip / .7z / .jsonl 或資料夾路徑")
    ap.add_argument("--from-api", action="store_true", help="從司法院開放 API (00:00-06:00) 抓取近 7 日最新判決並匯入")
    ap.add_argument("--api-limit", type=int, default=0, help="搭配 --from-api 時最多查詢幾筆（0=不限）")
    a = ap.parse_args(argv)

    if not a.inputs and not a.from_api:
        ap.error("請提供至少一個檔案/資料夾路徑，或使用 --from-api")

    seed_jids = get_seed_jids()
    skip_jids = get_trained_jids() - seed_jids

    dfs = []
    if a.from_api:
        from crawler.jdg_api import main as api_main
        args = ["--limit", str(a.api_limit)] if a.api_limit else []
        api_main(args)
        for p in sorted(config.INTERIM_DIR.glob("api_*.jsonl")):
            dfs.append(_rows_from_jsonl(p, skip_jids))

    for inp in a.inputs:
        dfs.append(ingest_path(inp, skip_jids))

    merge_and_retrain(dfs)


if __name__ == "__main__":
    main()
