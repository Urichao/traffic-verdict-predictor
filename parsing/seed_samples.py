"""產生符合司法院開放資料格式（JID / JTITLE / JFULL）的模擬台灣民事車禍判決書樣本。

用途：
1. 在尚未下載數 GB 的司法院每月 RAR 壓縮檔前，提供完整格式的判決書（含主文、原告主張、被告抗辯、本院之判斷），
   讓擷取管線（parsing.build_dataset）、模型訓練（model.train）、相似判例檢索（retrieval.index）與前端系統能立即端對端運作與驗證。
2. 當 data/interim/ 放入真實爬蟲下載的 .jsonl 後，重跑 build_dataset 與 train 即可無縫替換為全量真實資料。
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

COURTS = [
    ("TPDV", "臺灣臺北地方法院"),
    ("PCDV", "臺灣新北地方法院"),
    ("SLDV", "臺灣士林地方法院"),
    ("TYDV", "臺灣桃園地方法院"),
    ("SCDV", "臺灣新竹地方法院"),
    ("TCDV", "臺灣臺中地方法院"),
    ("TNDV", "臺灣臺南地方法院"),
    ("KSDV", "臺灣高雄地方法院"),
]

SCENARIOS = [
    # 1. 闖紅燈 + 骨折（全勝或高比例勝訴）
    {
        "weight": 18,
        "p_veh": "普通重型機車",
        "d_veh": "自用小客車",
        "violation": "行經號誌管制交岔路口時闖紅燈直行",
        "injury": "左側脛骨及腓骨開放性骨折、多處擦挫傷",
        "hosp_days": (5, 14),
        "surgery": True,
        "disability": False,
        "criminal": True,
        "appraisal": True,
        "p_fault_choices": [0.0, 0.0, 0.0, 0.1],
        "def_claim": "原告車速過快且請求之精神慰撫金過高",
        "medical": (35000, 120000),
        "care": (12000, 66000),
        "work": (30000, 120000),
        "labor": (0, 0),
        "vehicle": (15000, 50000),
        "consol_claim": (150000, 300000),
        "consol_grant_ratio": (0.5, 1.0),
        "force_full_win_prob": 0.35,
    },
    # 2. 酒後駕車追撞（高判賠 / 全勝）
    {
        "weight": 12,
        "p_veh": "普通重型機車",
        "d_veh": "自用小客車",
        "violation": "酒後駕車（吐氣所含酒精濃度達每公升0.55毫克），未保持行車安全距離自後追撞",
        "injury": "右側鎖骨骨折、腦震盪、全身多處擦挫傷",
        "hosp_days": (3, 10),
        "surgery": True,
        "disability": False,
        "criminal": True,
        "appraisal": False,
        "p_fault_choices": [0.0],
        "def_claim": "對肇事責任不爭執，惟原告請求之精神慰撫金偏高",
        "medical": (25000, 90000),
        "care": (10000, 45000),
        "work": (28000, 90000),
        "labor": (0, 0),
        "vehicle": (12000, 45000),
        "consol_claim": (150000, 350000),
        "consol_grant_ratio": (0.7, 1.0),
        "force_full_win_prob": 0.55,
    },
    # 3. 轉彎車未禮讓直行車 + 原告超速（與有過失 2~3 成，部分勝訴）
    {
        "weight": 22,
        "p_veh": "普通重型機車",
        "d_veh": "自小客車",
        "violation": "左轉彎時未禮讓直行車先行，貿然左轉跨越對向車道",
        "injury": "左遠端橈骨骨折、左膝挫傷及韌帶撕裂傷",
        "hosp_days": (2, 7),
        "surgery": True,
        "disability": False,
        "criminal": True,
        "appraisal": True,
        "p_fault_choices": [0.2, 0.3, 0.3],
        "def_claim": "原告行經路口超速行駛且疏未注意車前狀況，應負與有過失責任",
        "medical": (20000, 80000),
        "care": (8000, 36000),
        "work": (25000, 90000),
        "labor": (0, 0),
        "vehicle": (10000, 40000),
        "consol_claim": (120000, 300000),
        "consol_grant_ratio": (0.4, 0.7),
        "force_full_win_prob": 0.0,
    },
    # 4. 輕傷擦挫傷（無骨折，請求慰撫金偏高，部分勝訴）
    {
        "weight": 18,
        "p_veh": "機車",
        "d_veh": "自小客車",
        "violation": "變換車道時疏未注意右側直行車輛且未保持安全間隔",
        "injury": "四肢多處擦挫傷、頸部扭傷及背部鈍傷",
        "hosp_days": (0, 1),
        "surgery": False,
        "disability": False,
        "criminal": True,
        "appraisal": False,
        "p_fault_choices": [0.0, 0.2, 0.3],
        "def_claim": "原告僅受輕微擦挫傷，請求精神慰撫金顯屬過高，且原告亦未注意車前狀況與有過失",
        "medical": (2500, 18000),
        "care": (0, 0),
        "work": (6000, 30000),
        "labor": (0, 0),
        "vehicle": (5000, 25000),
        "consol_claim": (60000, 200000),
        "consol_grant_ratio": (0.2, 0.5),
        "force_full_win_prob": 0.08,
    },
    # 5. 重傷（顱內出血 + 勞動能力減損）
    {
        "weight": 10,
        "p_veh": "普通重型機車",
        "d_veh": "營業大貨車",
        "violation": "行經閃光黃燈路口超速且未注意車前狀況，撞擊穿越路口之原告機車",
        "injury": "創傷性硬腦膜下出血、顱骨骨折、右側股骨粉碎性骨折，經手術治療後遺存勞動能力減損",
        "hosp_days": (18, 45),
        "surgery": True,
        "disability": True,
        "criminal": True,
        "appraisal": True,
        "p_fault_choices": [0.0, 0.2, 0.3],
        "def_claim": "原告支線道車未暫停讓幹線道車先行，與有過失，且看護費與精神慰撫金過高",
        "medical": (150000, 450000),
        "care": (90000, 240000),
        "work": (120000, 300000),
        "labor": (450000, 1500000),
        "vehicle": (20000, 50000),
        "consol_claim": (500000, 1200000),
        "consol_grant_ratio": (0.5, 0.8),
        "force_full_win_prob": 0.1,
    },
    # 6. 行人穿越斑馬線遭撞擊（高比例勝訴或全勝）
    {
        "weight": 8,
        "p_veh": "行人步行穿越行人穿越道",
        "d_veh": "自用小客車",
        "violation": "行近行人穿越道不依規定暫停讓行人優先通行，貿然右轉撞擊原告",
        "injury": "骨盆骨折、左膝挫傷及腦震盪",
        "hosp_days": (6, 15),
        "surgery": True,
        "disability": False,
        "criminal": True,
        "appraisal": True,
        "p_fault_choices": [0.0, 0.0, 0.1],
        "def_claim": "當時天色昏暗視線不佳，且原告請求之慰撫金過高",
        "medical": (40000, 130000),
        "care": (20000, 70000),
        "work": (35000, 110000),
        "labor": (0, 0),
        "vehicle": (0, 0),
        "consol_claim": (150000, 350000),
        "consol_grant_ratio": (0.65, 1.0),
        "force_full_win_prob": 0.45,
    },
    # 7. 原告敗訴（被告無過失／原告闖紅燈或逆向／未能舉證／已和解賠償）
    {
        "weight": 12,
        "lose": True,
        "lose_reasons": [
            (
                "原告騎乘機車闖紅燈穿越路口，被告綠燈直行見狀煞車不及發生碰撞。原告主張被告未注意車前狀況受有左腿擦挫傷，請求醫療費用15,000元及精神慰撫金80,000元，共計95,000元。",
                "被告則以：伊係遵守號誌綠燈直行，係原告違規闖紅燈突入車道，伊反應時間不足1秒無從閃避，經行車事故鑑定委員會鑑定結果亦認被告無肇事因素等語置辯。",
                "經查，本件事故經行車事故鑑定委員會鑑定結果，認原告闖紅燈為肇事主因，被告並無肇事因素。原告復未能舉證證明被告有何違反注意義務之過失，是原告依侵權行為法律關係請求被告負損害賠償責任，為無理由，應予駁回。",
                95000,
            ),
            (
                "原告主張被告駕駛自小客車於路口與原告機車發生擦撞，致原告受有手部挫傷及車輛損害，請求醫療費用8,000元、車輛修理費22,000元及精神慰撫金60,000元，共計90,000元。",
                "被告抗辯：兩造已於事故當日達成和解，伊已當場給付和解金新臺幣30,000元，原告並簽立和解書拋棄其餘民事請求權，且原告亦領取強制汽車責任保險理賠金。",
                "經查，兩造業就本件車禍達成和解，被告已依約給付已賠償金額完畢，原告不得再就同一事故另行請求損害賠償。從而，原告之訴為無理由，應予駁回。",
                90000,
            ),
            (
                "原告主張其駕駛自小客車停等紅燈時遭被告自後碰撞，受有頸部拉傷及車損，請求醫療費用6,000元、修車費44,000元及精神慰撫金50,000元，共計100,000元。",
                "被告則以：事故現場並無監視器畫面，原告所提診斷證明書係事發後兩週始就醫開立，無法證明與本件事故有相當因果關係，且原告已領取強制險醫療給付逾其支出等語抗辯。",
                "按侵權行為損害賠償責任，須原告就損害發生及相當因果關係負舉證責任。本件原告無法舉證其傷勢與事故間之因果關係，且車輛修復費用扣除零件折舊及已領取之強制汽車責任保險金後已無餘額，故原告之請求無理由，應予駁回。",
                100000,
            ),
        ],
    },
]


def _ratio_cn(r: float) -> str:
    m = {0.1: "百分之十", 0.2: "百分之二十", 0.3: "百分之三十", 0.4: "百分之四十", 0.5: "百分之五十"}
    return m.get(round(r, 1), f"{int(round(r * 100))}%")


def generate_one(idx: int, rng: random.Random) -> dict:
    court_code, court_name = rng.choice(COURTS)
    year = rng.choice([111, 112, 113])
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    case_type = rng.choice(["訴", "簡", "小"])
    case_no = 100 + idx
    date_str = f"{year + 1911}{month:02d}{day:02d}"
    jid = f"{court_code},{year},{case_type},{case_no},{date_str},1"

    weights = [s["weight"] for s in SCENARIOS]
    sc = rng.choices(SCENARIOS, weights=weights, k=1)[0]

    if sc.get("lose"):
        p_txt, d_txt, c_txt, claim_total = rng.choice(sc["lose_reasons"])
        main_txt = "原告之訴及其假執行之聲請均駁回。\n訴訟費用由原告負擔。"
        full = (
            f"{court_name}民事判決\n{year}年度{case_type}字第{case_no}號\n"
            f"原告 甲○○\n被告 乙○○\n"
            f"上列當事人間請求侵權行為損害賠償(交通)事件，本院於民國{year}年{month}月{day}日言詞辯論終結，判決如下：\n"
            f"主文\n{main_txt}\n"
            f"事實及理由\n"
            f"一、原告起訴主張：{p_txt}並聲明：被告應給付原告新臺幣{claim_total:,}元，及自起訴狀繕本送達翌日起至清償日止，按週年利率百分之五計算之利息。\n"
            f"二、{d_txt}並聲明：原告之訴駁回。\n"
            f"三、本院之判斷：{c_txt}\n"
            f"中華民國 {year} 年 {month} 月 {day} 日\n民事第一庭 法官 丙○○"
        )
        return {
            "JID": jid, "JYEAR": str(year), "JCASE": case_type, "JNO": str(case_no),
            "JDATE": date_str, "JTITLE": "侵權行為損害賠償(交通)", "JFULL": full,
        }

    med = rng.randrange(sc["medical"][0], sc["medical"][1] + 1, 500) if sc["medical"][1] else 0
    care = rng.randrange(sc["care"][0], sc["care"][1] + 1, 1000) if sc["care"][1] else 0
    work = rng.randrange(sc["work"][0], sc["work"][1] + 1, 1000) if sc["work"][1] else 0
    labor = rng.randrange(sc["labor"][0], sc["labor"][1] + 1, 5000) if sc["labor"][1] else 0
    veh = rng.randrange(sc["vehicle"][0], sc["vehicle"][1] + 1, 1000) if sc["vehicle"][1] else 0
    consol_claim = rng.randrange(sc["consol_claim"][0], sc["consol_claim"][1] + 1, 10000)
    hosp = rng.randint(sc["hosp_days"][0], sc["hosp_days"][1])

    p_fault = rng.choice(sc["p_fault_choices"])
    full_win = (p_fault == 0.0) and (rng.random() < sc["force_full_win_prob"])

    if full_win:
        consol_award = consol_claim
        veh_award = veh
    else:
        ratio_c = rng.uniform(*sc["consol_grant_ratio"])
        consol_award = max(10000, int(round(consol_claim * ratio_c / 5000)) * 5000)
        consol_award = min(consol_award, consol_claim)
        veh_award = int(round(veh * rng.uniform(0.5, 0.85)))

    claim_total = med + care + work + labor + veh + consol_claim
    raw_allow = med + care + work + labor + veh_award + consol_award
    awarded = int(round(raw_allow * (1.0 - p_fault)))
    if full_win:
        awarded = claim_total
    else:
        awarded = min(awarded, claim_total - 1000)

    parts = [f"醫療費用{med:,}元"]
    if care:
        parts.append(f"看護費用{care:,}元")
    if work:
        parts.append(f"不能工作損失{work:,}元")
    if labor:
        parts.append(f"勞動能力減損{labor:,}元")
    if veh:
        parts.append(f"車輛修理費{veh:,}元")
    parts.append(f"精神慰撫金{consol_claim:,}元")

    hosp_str = f"，住院治療共{hosp}日" if hosp > 0 else ""
    surg_str = "，並接受手術治療" if sc["surgery"] else ""
    has_crim = sc["criminal"] and (rng.random() < 0.45)
    has_appr = sc["appraisal"] and (rng.random() < 0.40)
    crim_str = "案經本院刑事庭判處被告過失傷害罪刑確定在案。" if has_crim else ""
    appr_str = "本件事故經行車事故鑑定委員會鑑定結果，認被告為肇事主因。" if has_appr else ""

    if full_win:
        main_txt = (
            f"被告應給付原告新臺幣{awarded:,}元，及自民國{year}年{month}月1日起至清償日止，按週年利率百分之五計算之利息。\n"
            f"訴訟費用由被告負擔。"
        )
    else:
        main_txt = (
            f"被告應給付原告新臺幣{awarded:,}元，及自民國{year}年{month}月1日起至清償日止，按週年利率百分之五計算之利息。\n"
            f"原告其餘之訴駁回。\n"
            f"訴訟費用由被告負擔十分之七，餘由原告負擔。"
        )

    fault_judge = (
        f"又查原告行車亦有疏失，與有過失，本院審酌雙方違規情節，認原告應負擔過失責任比例為{_ratio_cn(p_fault)}，依民法第217條第1項規定減輕被告賠償金額。"
        if p_fault > 0
        else "被告雖抗辯原告與有過失，惟查原告並無任何違規情形，難認原告有何與有過失，無過失相抵之適用。"
    )

    full = (
        f"{court_name}民事判決\n{year}年度{case_type}字第{case_no}號\n"
        f"原告 甲○○\n被告 乙○○\n"
        f"上列當事人間請求侵權行為損害賠償(交通)事件，本院於民國{year}年{month}月{day}日言詞辯論終結，判決如下：\n"
        f"主文\n{main_txt}\n"
        f"事實及理由\n"
        f"一、原告起訴主張：被告於民國{year}年1月15日駕駛{sc['d_veh']}，{sc['violation']}，"
        f"適原告騎乘／使用{sc['p_veh']}行經該處，遭被告撞擊，致原告受有{sc['injury']}之傷害{hosp_str}{surg_str}。"
        f"{crim_str}{appr_str}原告因此支出及受有{'、'.join(parts)}，共計新臺幣{claim_total:,}元。"
        f"爰依民法第184條第1項前段、第191條之2、第193條第1項及第195條第1項規定提起本訴。"
        f"並聲明：被告應給付原告新臺幣{claim_total:,}元，及自起訴狀繕本送達翌日起至清償日止，按週年利率百分之五計算之利息。\n"
        f"二、被告抗辯：{sc['def_claim']}。並聲明：原告之訴駁回。\n"
        f"三、本院之判斷：按因故意或過失，不法侵害他人之權利者，負損害賠償責任；汽車、機車或其他非依軌道行駛之動力車輛，在使用中加損害於他人者，駕駛人應賠償因此所生之損害。"
        f"經查，原告主張受有{sc['injury']}之事實，有診斷證明書及醫療費用收據可稽。"
        f"本院審酌兩造身分、資力與原告所受痛苦程度，認精神慰撫金以新臺幣{consol_award:,}元為適當。"
        f"{fault_judge}綜上，原告請求被告給付新臺幣{awarded:,}元為有理由，應予准許；逾此範圍之請求，為無理由，應予駁回。\n"
        f"中華民國 {year} 年 {month} 月 {day} 日\n民事第二庭 法官 丙○○"
    )
    return {
        "JID": jid, "JYEAR": str(year), "JCASE": case_type, "JNO": str(case_no),
        "JDATE": date_str, "JTITLE": "侵權行為損害賠償(交通)", "JFULL": full,
    }


def seed(n: int = 240, seed_val: int = 42) -> Path:
    rng = random.Random(seed_val)
    out = config.INTERIM_DIR / "seed_traffic_judgments.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for i in range(1, n + 1):
            doc = generate_one(i, rng)
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    print(f"已產生 {n} 筆台灣民事車禍判決書樣本 → {out}")
    return out


if __name__ == "__main__":
    seed()
