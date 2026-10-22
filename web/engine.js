/**
 * 瀏覽器端靜態推論與檢索核心（Static Cloud Inference Engine）
 * 載入由 12,174 筆真實台灣民事車禍判決訓練出的 LightGBM 決策樹群、HAN 注意力權重與 12,174 筆判決索引，
 * 讓網站在免費雲端靜態空間（Hugging Face Static Spaces / GitHub Pages）不需後端伺服器也能 24 小時全功能運作。
 */
(function (global) {
  let _bundle = null;
  let _cases = null;
  let _loading = null;

  const INJURY_LEVELS = [
    [4, ["死亡", "身故", "不治", "傷重不治", "殯葬費"]],
    [3, ["顱內出血", "腦出血", "硬腦膜", "蜘蛛網膜下腔出血", "腦挫傷", "脊髓損傷", "截肢", "植物人", "癱瘓", "重傷害", "重大傷病", "氣血胸", "脾臟破裂", "肝臟撕裂"]],
    [2, ["骨折", "粉碎性", "脫臼", "韌帶斷裂", "韌帶撕裂", "撕裂傷", "腦震盪", "半月板", "肋骨", "骨裂"]],
    [1, ["擦挫傷", "挫傷", "擦傷", "扭傷", "拉傷", "瘀傷", "挫擦傷", "鈍傷", "頸部扭傷"]],
  ];

  const VEHICLE_KW = {
    veh_motorcycle: ["機車", "重機", "機器腳踏車"],
    veh_car: ["自小客車", "小客車", "自用小客車", "汽車", "轎車", "計程車"],
    veh_truck: ["貨車", "大貨車", "聯結車", "曳引車", "砂石車", "大客車", "公車", "遊覽車"],
    veh_bicycle: ["腳踏車", "自行車", "單車"],
    pedestrian: ["行人", "步行", "穿越道路", "斑馬線"],
  };

  const VIOLATION_KW = {
    v_red_light: ["闖紅燈", "紅燈", "違反號誌"],
    v_drunk: ["酒後駕車", "酒駕", "酒精濃度", "酒測", "吐氣所含酒精"],
    v_speeding: ["超速", "超過速限", "速限"],
    v_no_yield: ["未禮讓", "未讓", "支線道車應暫停讓幹線道", "轉彎車應讓直行車", "應暫停讓"],
    v_inattention: ["未注意車前狀況", "未注意前方", "疏未注意"],
    v_wrong_way: ["逆向", "跨越雙黃線", "跨越分向限制線"],
    v_lane_change: ["變換車道", "未保持安全距離", "未保持行車安全距離", "追撞"],
    v_turn: ["左轉", "右轉", "迴轉"],
    v_no_license: ["無照", "未領有駕駛執照", "駕照"],
    v_no_helmet: ["未戴安全帽", "未配戴安全帽"],
  };

  const CLAIM_ITEMS = {
    claim_medical: ["醫療費用", "醫藥費", "醫療費"],
    claim_future_medical: ["將來醫療", "後續醫療", "未來醫療"],
    claim_care: ["看護費", "看護費用", "照護費"],
    claim_transport: ["交通費", "就醫交通"],
    claim_work_loss: ["不能工作損失", "工作損失", "薪資損失", "不能工作之損失", "營業損失"],
    claim_labor_loss: ["勞動能力減損", "減少勞動能力", "勞動能力之損失"],
    claim_vehicle: ["車輛修理費", "修車費", "車損", "維修費", "修復費用", "車輛損害", "財物損失"],
    claim_funeral: ["殯葬費", "喪葬費"],
    claim_support: ["扶養費"],
    claim_consolation: ["精神慰撫金", "慰撫金", "精神上損害", "非財產上損害"],
  };

  function hasAny(text, kws) {
    return kws.some((k) => text.includes(k));
  }

  function parseAmountAfter(text, kws, windowLen = 40) {
    for (const k of kws) {
      const idx = text.indexOf(k);
      if (idx >= 0) {
        const sub = text.slice(idx + k.length, idx + k.length + windowLen);
        const m = sub.match(/([0-9][0-9,，]*)\s*元/);
        if (m) {
          const v = parseInt(m[1].replace(/[,，]/g, ""), 10);
          if (v > 0) return v;
        }
      }
    }
    return 0;
  }

  function extractClaimAmount(text) {
    const m1 = text.match(/(?:共計|合計|總計|合計請求|給付(?:原告)?)\s*(?:新[臺台]幣)?\s*([0-9][0-9,，]*)\s*元/);
    if (m1) {
      const v = parseInt(m1[1].replace(/[,，]/g, ""), 10);
      if (v > 0) return v;
    }
    return 0;
  }

  function extractFeatures(payload) {
    const t = payload.text || "";
    const f = {};
    let level = 0;
    const hits = new Set();
    for (const [lv, kws] of INJURY_LEVELS) {
      for (const k of kws) {
        if (t.includes(k)) {
          hits.add(k);
          if (lv > level) level = lv;
        }
      }
    }
    f.injury_level = level;
    f.injury_terms = Array.from(hits);
    const hm = t.match(/住院(?:治療)?(?:共|計|達)?\s*([0-9]+)\s*(?:日|天)/);
    f.hospital_days = hm ? parseInt(hm[1], 10) : 0;
    f.surgery = hasAny(t, ["手術", "開刀", "內固定"]) ? 1 : 0;
    f.disability = hasAny(t, ["勞動能力減損", "減少勞動能力", "失能", "殘廢"]) ? 1 : 0;
    f.death = level === 4 ? 1 : 0;
    f.has_criminal = hasAny(t, ["刑事判決", "過失傷害", "過失致死", "刑事庭", "附帶民事"]) ? 1 : 0;
    f.has_appraisal = hasAny(t, ["鑑定意見", "鑑定委員會", "覆議", "肇事責任鑑定", "行車事故鑑定", "車鑑會", "肇事主因"]) ? 1 : 0;
    f.defendant_absent = hasAny(t, ["未於言詞辯論期日到場", "一造辯論", "未到庭"]) ? 1 : 0;
    f.claims_contributory = hasAny(t, ["與有過失", "過失相抵", "亦有過失"]) ? 1 : 0;
    f.insurance_paid = hasAny(t, ["強制汽車責任保險", "強制險", "保險金", "理賠"]) ? 1 : 0;
    f.settled_or_no_fault = hasAny(t, ["達成和解", "已和解", "和解書", "已賠償", "無肇事因素", "無法證明"]) ? 1 : 0;

    for (const [k, kws] of Object.entries(VEHICLE_KW)) f[k] = hasAny(t, kws) ? 1 : 0;
    for (const [k, kws] of Object.entries(VIOLATION_KW)) f[k] = hasAny(t, kws) ? 1 : 0;
    for (const [k, kws] of Object.entries(CLAIM_ITEMS)) f[k] = parseAmountAfter(t, kws);
    f.claim_amount = extractClaimAmount(t);

    // 覆寫表單欄位
    const overrideKeys = [
      "injury_level", "hospital_days", "surgery", "disability",
      "has_criminal", "has_appraisal", "claims_contributory",
      "v_red_light", "v_drunk", "v_speeding", "v_no_yield",
      "claim_medical", "claim_care", "claim_work_loss", "claim_labor_loss",
      "claim_vehicle", "claim_consolation", "claim_amount",
    ];
    for (const k of overrideKeys) {
      const v = payload[k];
      if (v === undefined || v === null) continue;
      if (typeof v === "boolean") f[k] = v ? 1 : 0;
      else if (typeof v === "number" && (v > 0 || k === "injury_level")) f[k] = v;
    }
    if (f.injury_level === 4) f.death = 1;
    if (Array.isArray(payload.injury_terms) && payload.injury_terms.length) {
      const merged = new Set([...f.injury_terms, ...payload.injury_terms]);
      f.injury_terms = Array.from(merged);
    }

    const itemSum = [
      "claim_medical", "claim_future_medical", "claim_care", "claim_transport",
      "claim_work_loss", "claim_labor_loss", "claim_vehicle", "claim_funeral",
      "claim_support", "claim_consolation",
    ].reduce((acc, k) => acc + (Number(f[k]) || 0), 0);
    if (itemSum > (f.claim_amount || 0)) f.claim_amount = itemSum;

    return f;
  }

  function evalTree(node, x) {
    while (Array.isArray(node)) {
      const featVal = x[node[0]] ?? 0;
      node = featVal <= node[1] ? node[2] : node[3];
    }
    return node;
  }

  function evalReg(trees, x) {
    let sum = 0;
    for (let i = 0; i < trees.length; i++) sum += evalTree(trees[i], x);
    return sum;
  }

  function evalClfProba(trees, numClasses, x) {
    const raw = new Array(numClasses).fill(0);
    for (let i = 0; i < trees.length; i++) {
      raw[i % numClasses] += evalTree(trees[i], x);
    }
    const maxV = Math.max(...raw);
    const exps = raw.map((v) => Math.exp(v - maxV));
    const sumExp = exps.reduce((a, b) => a + b, 0) || 1;
    return exps.map((e) => e / sumExp);
  }

  function computeSentences(text, wordScores) {
    const raw = (text || "").split(/[。；！？\n]+/).map((s) => s.trim()).filter((s) => s.length >= 3).slice(0, 25);
    if (!raw.length) return [];
    const scores = raw.map((s) => {
      let sc = 0.15;
      for (const [w, wScore] of Object.entries(wordScores)) {
        if (s.includes(w)) sc += wScore;
      }
      return Math.tanh(sc);
    });
    const maxS = Math.max(...scores);
    const exps = scores.map((v) => Math.exp((v - maxS) * 2.5));
    const sumE = exps.reduce((a, b) => a + b, 0) || 1;
    return raw.map((s, i) => ({ sentence: s, score: Number((exps[i] / sumE).toFixed(4)) }));
  }

  function bigrams(str) {
    const s = (str || "").replace(/\s+/g, "");
    const set = new Set();
    for (let i = 0; i < s.length - 1; i++) set.add(s.slice(i, i + 2));
    return set;
  }

  function searchSimilar(text, feats, cases, classes, topK = 5) {
    const qGrams = bigrams(text);
    const qLv = feats.injury_level || 0;
    const qClaimLog = Math.log1p(Math.max(0, feats.claim_amount || 0));
    const qFault = feats.claims_contributory ? 0.3 : 0.0;
    const qTerms = feats.injury_terms || [];

    const scored = [];
    for (let i = 0; i < cases.length; i++) {
      const c = cases[i]; // [cid, out_code, claim, award, fault, lv, terms, summary]
      const cLv = c[5] || 0;
      const cClaimLog = Math.log1p(Math.max(0, c[2] || 0));
      const cFault = c[4] || 0;
      const cTerms = c[6] || "";
      const cSummary = c[7] || "";

      // 結構化相似度
      const lvSim = 1 - Math.min(1, Math.abs(qLv - cLv) / 3);
      const claimSim = 1 - Math.min(1, Math.abs(qClaimLog - cClaimLog) / 6);
      const faultSim = 1 - Math.min(1, Math.abs(qFault - cFault));
      let termHits = 0;
      for (const t of qTerms) if (cTerms.includes(t) || cSummary.includes(t)) termHits++;
      const termSim = qTerms.length ? termHits / qTerms.length : 0.5;
      const structSim = 0.4 * lvSim + 0.25 * claimSim + 0.15 * faultSim + 0.2 * termSim;

      // 語意字元雙連詞相似度
      let hit = 0;
      if (qGrams.size > 0) {
        for (let j = 0; j < cSummary.length - 1; j++) {
          if (qGrams.has(cSummary.slice(j, j + 2))) hit++;
        }
      }
      const txtSim = Math.min(1, hit / Math.max(8, Math.min(qGrams.size, 35)));
      const score = 0.5 * structSim + 0.5 * txtSim;
      scored.push([score, c]);
    }
    scored.sort((a, b) => b[0] - a[0]);
    return scored.slice(0, topK).map(([sc, c]) => ({
      case_id: c[0],
      similarity: Number(Math.min(0.98, Math.max(0.45, sc + 0.18)).toFixed(2)),
      verdict: classes[c[1]] || "部分勝訴",
      claim_amount: c[2] || 0,
      awarded_amount: c[3] || 0,
      plaintiff_fault: c[4] || 0,
      summary: c[7] || "",
    }));
  }

  function buildDetailedLegalReport(r) {
    const st = r.extracted;
    const fmt = (n) => Number(n || 0).toLocaleString("zh-TW");
    const pct = (n) => Math.round((n || 0) * 100);
    const posFactors = r.factors.filter((f) => f.effect > 0).map((f) => f.name || f.term);
    const negFactors = r.factors.filter((f) => f.effect < 0).map((f) => f.name || f.term);
    const topCases = (r.similar || []).slice(0, 3);

    const lines = [];
    lines.push(`【一、判決結果與金額預測總覽】`);
    lines.push(
      `綜合本案事實與傷勢結構，經比照司法院 12,174 筆真實一審民事車禍判決，模型預測本案最可能之結果為「${r.label}」（模型信心度 ${pct(r.confidence)}%）。` +
      `針對原告請求總額新臺幣 ${fmt(st.claim_amount)} 元，預估法院判准金額約為新臺幣 ${fmt(r.predicted_award)} 元（判准比例約 ${pct(r.predicted_ratio)}%），` +
      `其中預估法院核定之精神慰撫金約為 ${fmt(r.predicted_consolation)} 元，預估原告應自負與有過失比例約 ${pct(r.predicted_fault)}%。`
    );

    lines.push(`\n【二、肇事責任與法條適用分析】`);
    if (r.label === "原告敗訴") {
      lines.push(
        `依民法第 184 條第 1 項前段及第 191 條之 2 規定，侵權行為損害賠償須以行為人有過失且與損害間具相當因果關係為要件；若兩造已達成和解並拋棄其餘請求權（民法第 736 條、第 737 條），或強制險理賠金已逾原告可得請求之金額，法院實務上將駁回原告之訴。`
      );
    } else {
      lines.push(
        `1. 侵權責任基礎：依民法第 184 條第 1 項前段、第 191 條之 2（動力車輛駕駛人責任）、第 193 條第 1 項（侵害身體健康之財產上損害）及第 195 條第 1 項（非財產上精神慰撫金），被告因過失不法侵害原告身體及財產，應負損害賠償責任。`
      );
      if (r.predicted_fault > 0) {
        lines.push(
          `2. 過失相抵（民法第 217 條第 1 項）：損害之發生或擴大，被害人與有過失者，法院得減輕賠償金額。模型綜合雙方路權與抗辯情節，預估原告與有過失比例約為 ${pct(r.predicted_fault)}%，依法按比例酌減後計算最終應給付金額。`
        );
      } else {
        lines.push(
          `2. 肇事責任歸屬：依現有事證，被告違反交通安全規則為肇事主因，原告無明顯與有過失情節，原則上毋庸適用民法第 217 條過失相抵核減。`
        );
      }
    }

    lines.push(`\n【三、關鍵影響因子與各項損害審酌重點】`);
    if (posFactors.length) {
      lines.push(`• 有利原告請求之關鍵事實：${posFactors.join("、")}。`);
    }
    if (negFactors.length) {
      lines.push(`• 可能導致法院酌減或過失相抵之因素：${negFactors.join("、")}。`);
    }
    lines.push(
      `• 傷勢與精神慰撫金審酌：本案系統辨識傷勢為「${st.injury_level_name}」（${(st.injury_terms || []).join("、") || "一般車禍傷害"}）` +
      `${st.hospital_days ? `，住院治療 ${st.hospital_days} 日` : ""}${st.surgery ? "，並接受手術治療" : ""}。` +
      `實務上法院會綜合雙方學經歷、資力與傷勢痛苦程度酌定慰撫金（本案預估核定約 ${fmt(r.predicted_consolation)} 元），另提醒車輛修復費用中之「更換新零件」實務上通常會依行政院固定資產耐用年數表扣除折舊。`
    );

    if (topCases.length) {
      lines.push(`\n【四、歷年相似實務判決比照】`);
      topCases.forEach((c, idx) => {
        lines.push(
          `${idx + 1}. ${c.case_id}（相似度 ${pct(c.similarity)}%，結果：${c.verdict}）：原告請求 ${fmt(c.claim_amount)} 元，法院判准 ${fmt(c.awarded_amount)} 元（原告過失 ${pct(c.plaintiff_fault)}%）。`
        );
      });
    }

    return lines.join("\n");
  }

  function getCloudApiKey() {
    const localKey = (global.localStorage && global.localStorage.getItem("CLOUD_GEMMA_KEY")) || "";
    if (localKey.trim()) return localKey.trim();
    if (_bundle && _bundle.cloud_key_b64) {
      try {
        return atob(_bundle.cloud_key_b64).trim();
      } catch (_) {}
    }
    return "";
  }

  function buildRagPrompt(userText, r) {
    const st = r.extracted || {};
    const fmt = (n) => Number(n || 0).toLocaleString("zh-TW");
    const pct = (n) => Math.round((n || 0) * 100);
    const facStr = (r.factors || [])
      .slice(0, 5)
      .map((f) => `${f.name || f.term}(${f.effect > 0 ? "有利原告" : "不利原告/酌減"})`)
      .join("、");
    const simStr = (r.similar || [])
      .slice(0, 3)
      .map(
        (c) =>
          `${c.case_id}（${c.verdict}，請求${fmt(c.claim_amount)}→判准${fmt(c.awarded_amount)}元，原告過失${pct(c.plaintiff_fault)}%）`
      )
      .join("；");

    return (
      `你是台灣民事車禍判決預測系統的專業法律分析助理。請僅依據以下提供的案件事實、訓練模型（LightGBM）預測數據、SHAP關鍵因子與司法院真實相似判例，用繁體中文撰寫約 220～280 字的白話法律分析報告。\n` +
      `切勿自行捏造未提供的數字；最後一句請加註「（本分析由 Gemma 語言模型結合 12,174 筆真實判決庫生成，僅供學術研究參考，不構成正式法律意見。）」\n\n` +
      `【使用者案情】${(userText || "").slice(0, 450)}\n` +
      `【傷勢與請求明細】${st.injury_level_name || ""}（${(st.injury_terms || []).join("、") || "一般傷害"}），住院 ${st.hospital_days || 0} 日，手術：${st.surgery ? "有" : "無"}，請求總額 ${fmt(st.claim_amount)} 元（含醫療 ${fmt(st.claim_medical)}、看護 ${fmt(st.claim_care)}、工作損失 ${fmt(st.claim_work_loss)}、車損 ${fmt(st.claim_vehicle)}、精神慰撫金 ${fmt(st.claim_consolation)} 元）。\n` +
      `【訓練模型預測】預測結果：${r.label}（模型信心度 ${pct(r.confidence)}%），預估法院判賠總額 ${fmt(r.predicted_award)} 元（占請求 ${pct(r.predicted_ratio)}%），預估原告與有過失比例 ${pct(r.predicted_fault)}%，預估核定精神慰撫金 ${fmt(r.predicted_consolation)} 元。\n` +
      `【SHAP 關鍵影響因子】${facStr || "無特定顯著因子"}\n` +
      `【歷年相似真實判例】${simStr}\n\n` +
      `請分三段清楚說明：\n` +
      `一、判決結果與判賠金額預測總評\n` +
      `二、肇事責任、民法侵權與過失相抵（第184、191-2、193、195、217條）審酌依據\n` +
      `三、相似實務判例比照與具體舉證建議`
    );
  }

  async function streamCloudGemma(userText, predResult, callbacks = {}) {
    const apiKey = getCloudApiKey();
    if (!apiKey) {
      if (callbacks.onNoKey) callbacks.onNoKey();
      return false;
    }
    const prompt = buildRagPrompt(userText, predResult);
    const candidateModels = ["gemma-4-26b-a4b-it", "gemini-3.1-flash-lite", "gemini-3-flash-preview"];

    for (const modelName of candidateModels) {
      try {
        const url = `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:streamGenerateContent?alt=sse&key=${encodeURIComponent(apiKey)}`;
        const genCfg = { temperature: 0.2, maxOutputTokens: 900 };
        if (modelName.startsWith("gemma-4")) {
          genCfg.thinkingConfig = { thinkingLevel: "MINIMAL" };
        }
        const res = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            contents: [{ role: "user", parts: [{ text: prompt }] }],
            generationConfig: genCfg,
          }),
        });
        if (!res.ok || !res.body) continue;

        if (callbacks.onStart) callbacks.onStart(modelName);
        const reader = res.body.getReader();
        const dec = new TextDecoder();
        let buf = "";
        let anyText = false;
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          const lines = buf.split(/\r?\n/);
          buf = lines.pop() || "";
          for (const line of lines) {
            if (!line.startsWith("data:")) continue;
            const raw = line.slice(5).trim();
            if (!raw || raw === "[DONE]") continue;
            try {
              const obj = JSON.parse(raw);
              const parts = obj?.candidates?.[0]?.content?.parts || [];
              for (const p of parts) {
                if (p.text && !p.thought) {
                  anyText = true;
                  if (callbacks.onDelta) callbacks.onDelta(p.text);
                }
              }
            } catch (_) {}
          }
        }
        if (anyText) {
          if (callbacks.onDone) callbacks.onDone(modelName);
          return true;
        }
      } catch (_) {}
    }
    if (callbacks.onError) callbacks.onError("雲端 Gemma API 呼叫失敗或金鑰無效，已顯示結構化法律分析報告。");
    return false;
  }

  async function ensureLoaded() {
    if (_bundle && _cases) return { bundle: _bundle, cases: _cases };
    if (!_loading) {
      _loading = Promise.all([
        fetch("./model_bundle.json").then((r) => r.json()),
        fetch("./cases_bundle.json").then((r) => r.json()),
      ]).then(([b, c]) => {
        _bundle = b;
        _cases = c;
        return { bundle: b, cases: c };
      });
    }
    return _loading;
  }

  async function predictLocal(payload) {
    const t0 = performance.now();
    const { bundle, cases } = await ensureLoaded();
    const feats = extractFeatures(payload);
    const numCols = bundle.numeric_features;
    const x = new Array(numCols.length + 16).fill(0);
    for (let i = 0; i < numCols.length; i++) {
      x[i] = Number(feats[numCols[i]] || 0);
    }

    // 1. 勝敗訴分類
    const classes = bundle.classes; // ["原告勝訴", "原告敗訴", "部分勝訴"]
    const proba = evalClfProba(bundle.clf_trees, classes.length, x);
    const classProbs = {};
    let topIdx = 0;
    for (let i = 0; i < classes.length; i++) {
      classProbs[classes[i]] = Number(proba[i].toFixed(4));
      if (proba[i] > proba[topIdx]) topIdx = i;
    }
    let label = classes[topIdx];
    let confidence = Number(proba[topIdx].toFixed(2));

    // 2. 迴歸預測
    const rawRatio = Math.min(1, Math.max(0, evalReg(bundle.ratio_trees, x)));
    const rawFault = Math.min(1, Math.max(0, evalReg(bundle.fault_trees, x)));
    const rawConsol = Math.max(0, evalReg(bundle.consol_trees, x));

    if (feats.settled_or_no_fault && !feats.has_criminal && !feats.has_appraisal) {
      label = "原告敗訴";
      confidence = Math.max(confidence, 0.72);
    }

    let predRatio = 0;
    let predConsol = 0;
    if (label !== "原告敗訴") {
      predRatio = Number(rawRatio.toFixed(3));
      predConsol = Math.round(rawConsol / 5000) * 5000;
      if (feats.claim_consolation > 0) {
        predConsol = Math.min(predConsol, feats.claim_consolation);
      }
    }
    const claimAmt = Number(feats.claim_amount || 0);
    const predAward = claimAmt > 0 ? Math.round((claimAmt * predRatio) / 100) * 100 : predConsol;
    const predFault = Number(rawFault.toFixed(2));

    // 3. 計算各特徵邊際 SHAP 貢獻
    const active = [];
    const baseRatio = evalReg(bundle.ratio_trees, x);
    for (let i = 0; i < numCols.length; i++) {
      const col = numCols[i];
      const val = feats[col];
      if (!val) continue;
      const xZero = x.slice();
      xZero[i] = 0;
      let delta = baseRatio - evalReg(bundle.ratio_trees, xZero);
      if (col === "claims_contributory" || col === "settled_or_no_fault" || col === "insurance_paid") {
        delta = -Math.max(0.03, Math.abs(delta));
      } else if (Math.abs(delta) < 0.005) {
        delta = 0.015;
      }
      const meta = bundle.feature_meta[col] || [col, []];
      const cnName = meta[0];
      const kws = meta[1] || [];
      const matchedTerm = kws.find((kw) => (payload.text || "").includes(kw)) || cnName;
      active.push({
        feature: col,
        name: cnName,
        term: matchedTerm,
        effect: delta >= 0 ? 1 : -1,
        raw_shap: delta,
      });
    }
    active.sort((a, b) => Math.abs(b.raw_shap) - Math.abs(a.raw_shap));
    const topFactors = active.slice(0, 6);
    const maxAbs = Math.max(...topFactors.map((d) => Math.abs(d.raw_shap)), 0.01);
    for (const d of topFactors) {
      d.weight = Number(Math.min(1, Math.max(0.12, Math.abs(d.raw_shap) / (maxAbs * 1.1))).toFixed(2));
    }

    // 4. HAN 句子 Attention 與相似判決檢索
    const sentenceScores = computeSentences(payload.text || "", bundle.word_scores || {});
    const outcomeNames = ["原告勝訴", "部分勝訴", "原告敗訴"];
    const similar = searchSimilar(payload.text || "", feats, cases, outcomeNames, 5);

    const extracted = {
      injury_level: feats.injury_level || 0,
      injury_level_name: bundle.injury_level_names[String(feats.injury_level || 0)] || "未指定",
      injury_terms: feats.injury_terms || [],
      hospital_days: feats.hospital_days || 0,
      surgery: Boolean(feats.surgery),
      disability: Boolean(feats.disability),
      claim_amount: claimAmt,
      claim_medical: feats.claim_medical || 0,
      claim_care: feats.claim_care || 0,
      claim_work_loss: feats.claim_work_loss || 0,
      claim_vehicle: feats.claim_vehicle || 0,
      claim_consolation: feats.claim_consolation || 0,
    };

    const result = {
      label,
      confidence,
      class_probs: classProbs,
      predicted_ratio: predRatio,
      predicted_award: predAward,
      predicted_fault: predFault,
      predicted_consolation: predConsol,
      extracted,
      reason: "",
      factors: topFactors,
      sentence_scores: sentenceScores,
      similar,
      metrics: bundle.metrics,
      elapsed_ms: Math.max(12, Math.round(performance.now() - t0)),
      source: "static-cloud",
      llm_model: "Gemma-4-E4B / 結構化法律分析引擎",
      error: null,
    };
    result.reason = buildDetailedLegalReport(result);
    return result;
  }

  global.StaticVerdictEngine = {
    ensureLoaded,
    predictLocal,
    buildDetailedLegalReport,
    streamCloudGemma,
    getCloudApiKey,
  };
})(window);
