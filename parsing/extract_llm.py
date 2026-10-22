"""以 Gemma-4-E4B 補齊規則（Regex）無法擷取出的判決要點。"""
from __future__ import annotations

import pandas as pd

from llm.client import chat, parse_json_object
from llm.prompts import EXTRACT_MISSING_SYSTEM
from parsing.extract_rules import LABELS


def llm_extract_one(main_text: str, facts_text: str, court_text: str) -> dict:
    user_prompt = (
        f"【主文】\n{(main_text or '')[:600]}\n\n"
        f"【事實摘錄】\n{(facts_text or '')[:800]}\n\n"
        f"【法院判斷摘錄】\n{(court_text or '')[:1200]}"
    )
    reply = chat(
        [
            {"role": "system", "content": EXTRACT_MISSING_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=300,
        json_mode=True,
    )
    return parse_json_object(reply)


def fill_missing(df: pd.DataFrame, limit: int = 100) -> pd.DataFrame:
    """針對缺少 outcome、claim_amount 或 plaintiff_fault 的第一審判決，呼叫 Gemma 補齊。"""
    df = df.copy()
    mask = df["first_instance"] & (
        df["outcome"].isna() | (df["claim_amount"] == 0) | df["plaintiff_fault"].isna()
    )
    idxs = df[mask].index[:limit]
    print(f"使用 Gemma 補齊 {len(idxs)} 筆缺漏判決要點...")
    for n, idx in enumerate(idxs, 1):
        row = df.loc[idx]
        try:
            d = llm_extract_one(row.get("main_text", ""), row.get("facts_text", ""), row.get("court_text", ""))
        except Exception as e:
            print(f"[{n}/{len(idxs)}] {row.get('jid')} LLM 擷取略過：{e}")
            continue
        if pd.isna(row["outcome"]) and d.get("outcome") in LABELS:
            df.at[idx, "outcome"] = d["outcome"]
        if (not row["claim_amount"]) and isinstance(d.get("claim_amount"), (int, float)) and d["claim_amount"] > 0:
            df.at[idx, "claim_amount"] = int(d["claim_amount"])
        if pd.isna(row["plaintiff_fault"]) and isinstance(d.get("plaintiff_fault"), (int, float)):
            v = float(d["plaintiff_fault"])
            if 0.0 <= v <= 1.0:
                df.at[idx, "plaintiff_fault"] = v
        if pd.isna(row["awarded_consolation"]) and isinstance(d.get("awarded_consolation"), (int, float)):
            df.at[idx, "awarded_consolation"] = int(d["awarded_consolation"])
        # 重新計算判賠比例
        c = df.at[idx, "claim_amount"]
        aw = df.at[idx, "awarded_amount"]
        out = df.at[idx, "outcome"]
        if c and out:
            r = 0.0 if out == "原告敗訴" else aw / c
            if 0.0 <= r <= 1.05:
                df.at[idx, "award_ratio"] = min(r, 1.0)
    return df
