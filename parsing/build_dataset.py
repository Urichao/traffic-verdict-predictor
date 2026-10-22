"""把 data/interim/*.jsonl（原始判決）轉成 data/processed/cases.parquet（結構化要點）。

用法：
    python -m parsing.build_dataset
    python -m parsing.build_dataset --llm-fill 200   # 再用 Gemma 補齊 200 筆規則抓不到過失比例的案件（很慢）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from parsing.extract_rules import extract_case  # noqa: E402


def iter_docs(paths):
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def build(paths=None, out: Path = config.CASES_PATH) -> pd.DataFrame:
    paths = paths or sorted(config.INTERIM_DIR.glob("*.jsonl"))
    rows, seen = [], set()
    for d in iter_docs(paths):
        jid = d.get("JID")
        if not jid or jid in seen:
            continue
        seen.add(jid)
        try:
            rows.append(extract_case(d))
        except Exception as e:  # 單一判決格式異常不影響整體
            print(f"[warn] {jid}: {e}")
    df = pd.DataFrame(rows)
    if df.empty:
        print("沒有資料：請先執行爬蟲（crawler.opendata / crawler.jdg_api）")
        return df
    df["injury_terms"] = df["injury_terms"].apply(lambda x: "、".join(x) if isinstance(x, list) else x)
    df.to_parquet(out, index=False)
    print(f"共 {len(df)} 筆 → {out}")
    print_stats(df)
    return df


def print_stats(df: pd.DataFrame):
    fi = df[df["first_instance"]]
    print(f"第一審 {len(fi)} 筆；判決結果分布：")
    print(fi["outcome"].value_counts(dropna=False).to_string())
    for col in ("claim_amount", "award_ratio", "plaintiff_fault", "awarded_consolation"):
        s = fi[col]
        ok = s.notna() & (s != 0) if col == "claim_amount" else s.notna()
        print(f"  {col:22s} 擷取成功率 {ok.mean():6.1%}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm-fill", type=int, default=0)
    a = ap.parse_args(argv)
    df = build()
    if a.llm_fill and not df.empty:
        from parsing.extract_llm import fill_missing
        df = fill_missing(df, limit=a.llm_fill)
        df.to_parquet(config.CASES_PATH, index=False)
        print_stats(df)


if __name__ == "__main__":
    main()
