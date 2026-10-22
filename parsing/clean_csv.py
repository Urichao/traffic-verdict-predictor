"""篩選並修復 traffic_dataset_v1(1).csv，同時轉為系統訓練資料集 (cases.parquet)。

原始 CSV (2,898 筆) 經檢查發現的問題：
1. 混入非判決／非車禍案件：
   - 825 筆為「民事裁定」（補繳裁判費、移送管轄、駁回聲請等程序裁定，無實體勝負）
   - 數百筆為消費者債務清理（更生、清算）、遺產分割、買賣契約、名譽侵權等非車禍案件
   - 約百筆為二審／三審上訴案件（簡上、小上、高等法院、最高法院，主文為「上訴駁回／原判決廢棄」）
2. 欄位錯置與缺漏：
   - `court` 幾乎全被寫死為「臺灣臺北地方法院」（2,896 筆），實際上來自全台 20 多間法院
   - `main_text` 放的是「案由 (JTITLE)」，真正的「主文」與「事實及理由」混在 `accident_description` 前 1,200 字中
   - 後 2,895 筆的 `claimed_amount` 與 `awarded_amount` 全為 0、`injury_level` 全為「待標註」
   - `judgment_result` 有誤判（例如將補繳裁判費裁定或上訴駁回標為原告勝訴）

本腳本執行以下清理與擷取：
- 嚴格篩選「第一審民事車禍損害賠償實體判決」
- 還原真實法院名稱 (`court`)、切分出真正的 `main_text`（主文）與純粹的 `accident_description`（原告主張＋被告抗辯，不含主文以免答案洩漏）
- 從主文精確擷取 `awarded_amount`（判賠金額）與 `judgment_result`（原告勝訴／部分勝訴／原告敗訴）
- 從事實聲明與主文訴訟費用負擔比例（民事訴訟法第79條）精確還原 `claimed_amount`（請求金額）與 `award_ratio`
- 自動標註 `injury_level`（無受傷／輕傷／骨折／重傷／死亡）與各項違規、請求特徵
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from crawler.filters import is_first_instance, parse_jid  # noqa: E402
from parsing.extract_rules import (  # noqa: E402
    INJURY_LEVEL_NAMES,
    extract_awarded_consolation,
    extract_claim_amount,
    extract_features,
    extract_plaintiff_fault,
    parse_main,
)
from parsing.numbers import NUM_CHARS, RATIO_TOKEN, cn2int, find_amounts, parse_ratio  # noqa: E402
from parsing.sections import split_sections  # noqa: E402

COURT_LINE_RE = re.compile(r"^(臺灣[^\r\n]+?法院|福建[^\r\n]+?法院)")
COLLISION_RE = re.compile(r"車禍|交通事故|肇事|碰撞|擦撞|追撞|撞擊|過失傷害|行車事故|閃避不及|車體受損|車輛受損|車輛毀損")
SUBROGATION_RE = re.compile(r"保險法第53條|代位|車體損失險|車體險")
GENERAL_INJURY_RE = re.compile(r"受有傷害|受傷|體傷|傷勢|醫療費用|醫藥費|看護費|精神慰撫金|慰撫金|診斷證明書|過失傷害")
HIGHER_COURTS = {"TPSV", "TPHV", "TCHV", "TNHV", "KSHV", "HLHV", "IPCV", "KSYV"}


def extract_cost_ratio(main_text: str) -> float | None:
    """從主文『訴訟費用由被告負擔...』擷取判賠占請求比例（依民事訴訟法第79條按勝敗比例分擔）。"""
    m = re.search(rf"訴訟費用[^。]*?由被告(?:連帶)?負擔\s*{RATIO_TOKEN}", main_text)
    if m:
        r = parse_ratio(m.group(1))
        if r and 0.01 <= r < 1.0:
            return round(r, 4)
    m = re.search(rf"訴訟費用[^。]*?由原告(?:連帶)?負擔\s*{RATIO_TOKEN}", main_text)
    if m:
        r = parse_ratio(m.group(1))
        if r and 0.0 < r <= 0.99:
            return round(1.0 - r, 4)
    m = re.search(
        rf"訴訟費用[^。]*?新[臺台]幣([{NUM_CHARS}]+)\s*元[^。]*?(?:其中|由被告(?:連帶)?負擔)[^。]*?新[臺台]幣([{NUM_CHARS}]+)\s*元",
        main_text,
    )
    if m:
        total_cost = cn2int(m.group(1))
        def_cost = cn2int(m.group(2))
        if total_cost and def_cost and 0 < def_cost <= total_cost:
            return round(def_cost / total_cost, 4)
    return None


def extract_body_claim_fallback(body: str) -> int | None:
    """針對 1,200 字截斷文本，補充擷取原告主張之維修/賠償金額。"""
    c = extract_claim_amount(body)
    if c:
        return c
    pats = [
        rf"(?:理賠|賠付|支出|修復費用|維修費用|修繕費用|損害|損失)(?:共|合計|共計|總計)?(?:新[臺台]幣)?(?:[（(]下同[)）])?\s*([{NUM_CHARS}]+)\s*元",
        rf"請求(?:被告)?(?:連帶)?(?:給付|賠償)[^。]{{0,20}}?([{NUM_CHARS}]+)\s*元",
    ]
    for p in pats:
        m = re.search(p, body or "")
        if m:
            v = cn2int(m.group(1))
            if v and v >= 1000:
                return v
    return None


def clean_traffic_csv(
    src_csv: Path,
    out_cleaned_csv: Path,
    out_personal_csv: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(src_csv)
    print(f"讀取原始 CSV：{src_csv}（共 {len(df)} 筆）")

    stats = {
        "total_raw": len(df),
        "synthetic_demo_rows": 0,
        "dropped_ruling": 0,
        "dropped_appeal_or_higher_court": 0,
        "dropped_non_traffic": 0,
        "dropped_unparseable_main": 0,
        "kept_valid_traffic": 0,
    }

    cleaned_rows = []
    parquet_rows = []

    for _, row in df.iterrows():
        cid = str(row.get("case_id", "")).strip()
        raw_desc = str(row.get("accident_description", "") or "")
        raw_title = str(row.get("main_text", "") or "").strip()

        # 跳過最前面 3 筆人工合成範例 (TPD-112-訴-101 等)
        if "," not in cid:
            stats["synthetic_demo_rows"] += 1
            continue

        meta = parse_jid(cid)
        if meta["court_code"] in HIGHER_COURTS or not is_first_instance(meta["case_type"]):
            stats["dropped_appeal_or_higher_court"] += 1
            continue

        head = re.sub(r"\s+", "", raw_desc[:260])
        if "裁定" in head or "判決" not in head:
            stats["dropped_ruling"] += 1
            continue

        is_traffic_title = bool(re.search(r"交通|車禍", raw_title))
        is_damage_title = bool(re.search(r"損害賠償|侵權行為", raw_title))
        has_collision = bool(COLLISION_RE.search(raw_desc))
        if not (is_traffic_title or (is_damage_title and has_collision)):
            stats["dropped_non_traffic"] += 1
            continue
        # 若標題非交通，進一步排除買賣契約、租賃契約、工程款等糾紛
        if not is_traffic_title and re.search(r"買賣協議|買賣契約|工程款|房屋租賃|借款", raw_desc):
            stats["dropped_non_traffic"] += 1
            continue

        sec = split_sections(raw_desc)
        if not sec.main or not sec.body:
            stats["dropped_unparseable_main"] += 1
            continue

        main_info = parse_main(sec.main)
        outcome = main_info["outcome"]
        awarded = int(main_info["awarded_amount"] or 0)
        if outcome not in ("原告勝訴", "部分勝訴", "原告敗訴"):
            stats["dropped_unparseable_main"] += 1
            continue

        # 還原真實法院名稱
        cm = COURT_LINE_RE.search(raw_desc.lstrip())
        true_court = cm.group(1) if cm else str(row.get("court", ""))

        # 擷取或由訴訟費用負擔比例推算請求金額 (claimed_amount)
        claimed = extract_body_claim_fallback(sec.body) or 0
        cost_ratio = extract_cost_ratio(sec.main)
        if outcome == "原告勝訴" and awarded > 0:
            claimed = awarded
        elif outcome == "部分勝訴" and awarded > 0:
            if claimed < awarded or claimed == 0:
                if cost_ratio and 0.02 <= cost_ratio <= 0.99:
                    claimed = int(round(awarded / cost_ratio))
                else:
                    claimed = int(round(awarded / 0.65))

        award_ratio = None
        if outcome == "原告敗訴":
            award_ratio = 0.0
        elif outcome == "原告勝訴":
            award_ratio = 1.0
        elif claimed > 0 and awarded > 0:
            r = awarded / claimed
            if 0.0 <= r <= 1.05:
                award_ratio = round(min(1.0, r), 4)
        elif cost_ratio is not None:
            award_ratio = cost_ratio

        # 特徵與傷勢擷取
        facts = sec.facts if len(sec.facts) >= 40 else sec.body
        feats = extract_features(facts)
        if feats["injury_level"] == 0 and GENERAL_INJURY_RE.search(facts) and not SUBROGATION_RE.search(facts):
            feats["injury_level"] = 1
            if not feats["injury_terms"]:
                feats["injury_terms"] = ["車禍體傷"]

        feats["claim_amount"] = claimed
        is_subrog = int(bool(SUBROGATION_RE.search(facts)))
        inj_label_map = {0: "無受傷(純車損)", 1: "輕傷(擦挫傷等)", 2: "骨折/中度傷", 3: "重傷", 4: "死亡"}
        inj_str = inj_label_map.get(feats["injury_level"], "無受傷(純車損)")

        # 過失比例：若內文未出現但有訴訟費用負擔比，可作為參考
        p_fault = extract_plaintiff_fault(sec.court)
        consol = extract_awarded_consolation(sec.court)

        stats["kept_valid_traffic"] += 1

        cleaned_rows.append({
            "id": len(cleaned_rows) + 1,
            "case_id": cid,
            "court": true_court,
            "case_date": row.get("case_date", ""),
            "case_number": row.get("case_number", ""),
            "case_title": raw_title,
            "is_insurance_subrogation": is_subrog,
            "injury_level": inj_str,
            "injury_level_code": feats["injury_level"],
            "injury_terms": "、".join(feats["injury_terms"]),
            "claimed_amount": claimed,
            "awarded_amount": awarded,
            "award_ratio": award_ratio if award_ratio is not None else "",
            "judgment_result": outcome,
            "main_text": sec.main,
            "accident_description": facts,
        })

        parquet_rows.append({
            "jid": cid,
            "court_code": meta["court_code"],
            "court_prefix": meta["court_code"][:3],
            "case_type": meta["case_type"],
            "year": meta["year"],
            "date": meta["date"],
            "first_instance": True,
            "title": raw_title,
            **{k: v for k, v in feats.items() if k != "injury_terms"},
            "injury_terms": "、".join(feats["injury_terms"]),
            "outcome": outcome,
            "awarded_amount": awarded,
            "award_ratio": award_ratio,
            "plaintiff_fault": p_fault,
            "awarded_consolation": consol,
            "main_text": sec.main,
            "facts_text": facts[:6000],
            "court_text": sec.court[:6000],
        })

    clean_df = pd.DataFrame(cleaned_rows)
    clean_df.to_csv(out_cleaned_csv, index=False, encoding="utf-8-sig")

    if out_personal_csv is not None:
        personal_df = clean_df[clean_df["is_insurance_subrogation"] == 0].reset_index(drop=True)
        personal_df["id"] = range(1, len(personal_df) + 1)
        personal_df.to_csv(out_personal_csv, index=False, encoding="utf-8-sig")
    else:
        personal_df = pd.DataFrame()

    print("\n================ 篩選與清理結果統計 ================")
    for k, v in stats.items():
        print(f"  {k:34s}: {v:5d}")
    print("\n--- 篩選後判決結果分布 ---")
    print(clean_df["judgment_result"].value_counts().to_string())
    print("\n--- 篩選後傷勢等級分布 ---")
    print(clean_df["injury_level"].value_counts().to_string())
    print("\n--- 篩選後案件類型（個人車禍 vs. 產險代位求償）---")
    print(clean_df["is_insurance_subrogation"].map({0: "個人車禍求償(含體傷/財損)", 1: "產險公司車體險代位求償"}).value_counts().to_string())
    print(f"\n已輸出完整清理後 CSV：{out_cleaned_csv}（共 {len(clean_df)} 筆）")
    if out_personal_csv:
        print(f"已輸出個人車禍專用 CSV：{out_personal_csv}（共 {len(personal_df)} 筆）")

    return clean_df, pd.DataFrame(parquet_rows)


if __name__ == "__main__":
    src = Path(r"c:\Users\urichiao\OneDrive\文件\traffic_dataset_v1(1).csv")
    out_all = Path(r"c:\Users\urichiao\OneDrive\文件\traffic_dataset_cleaned.csv")
    out_personal = Path(r"c:\Users\urichiao\OneDrive\文件\traffic_dataset_personal_injury.csv")
    clean_df, pq_df = clean_traffic_csv(src, out_all, out_personal)

    # 同步存入專案 data/processed/ 並與既有特徵樣本整合以供模型訓練與檢索
    clean_df.to_csv(config.PROCESSED_DIR / "traffic_dataset_cleaned.csv", index=False, encoding="utf-8-sig")
    if config.CASES_PATH.exists():
        old_df = pd.read_parquet(config.CASES_PATH)
        merged = pd.concat([pq_df, old_df], ignore_index=True).drop_duplicates(subset=["jid"], keep="first")
    else:
        merged = pq_df
    merged.to_parquet(config.CASES_PATH, index=False)
    print(f"已更新系統訓練與檢索資料庫：{config.CASES_PATH}（共 {len(merged)} 筆）")
