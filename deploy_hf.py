r"""一鍵將本專案部署至 Hugging Face Static Spaces（100% 免費、免 PRO 訂閱、24 小時全球在線）。

說明：
    Hugging Face 目前限制免費帳號建立 Docker/Gradio Space 需升級 PRO（會回傳 402 Payment Required），
    但「Static Space (sdk=static)」對所有免費帳號完全開放！
    本腳本會自動將訓練好的 12,174 筆真實判決模型（LightGBM + HAN + 12,174 筆檢索庫）匯出為前端推論包，
    並一鍵上傳至 Hugging Face Static Space，讓組員與指導教授 24 小時都能透過公開網址直接線上預測分析。

用法：
    C:\Users\urichiao\deploy.cmd --token hf_你的Token
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402, F401
from model.export_static import export_all  # noqa: E402

STATIC_README = """---
title: 智慧司法判決預測與可解釋案例分析系統
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
---

# 台灣民事車禍判決預測與可解釋案例分析系統

- **訓練與檢索資料集**：司法院 2026 年 1–7 月真實一審民事車禍損害賠償判決共 **12,174 筆**
- **預測模型**：LightGBM 分類與多目標迴歸模型（測試集準確率 **83.53%**）＋ TreeSHAP 可解釋特徵貢獻度 ＋ 參考專案 HAN 雙層句子注意力權重
"""


def deploy(token: str, space_name: str, private: bool = False, gemini_key: str = "") -> str:
    print("[1/3] 正在匯出最新訓練模型與 12,174 筆真實判決檢索包...")
    export_all(cloud_key=gemini_key)

    api = HfApi(token=token)
    user_info = api.whoami()
    username = user_info["name"]
    repo_id = space_name if "/" in space_name else f"{username}/{space_name}"

    print(f"[2/3] 正在建立 / 確認免費 Hugging Face Static Space：{repo_id} (SDK=static) ...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="static",
        private=private,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "README.md").write_text(STATIC_README, encoding="utf-8")
        for fname in ("index.html", "engine.js", "model_bundle.json", "cases_bundle.json"):
            src = ROOT / "web" / fname
            if src.exists():
                shutil.copy2(src, tmp_path / fname)

        print(f"[3/3] 正在上傳網站與 12,174 筆判決推論引擎至 {repo_id} ...")
        api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=repo_id,
            repo_type="space",
            commit_message="Deploy Taiwan Civil Traffic Verdict Predictor (12,174 real cases static engine)",
        )

    space_url = f"https://huggingface.co/spaces/{repo_id}"
    direct_url = f"https://{username.lower()}-{space_name.split('/')[-1].lower().replace('_', '-')}.hf.space"
    print("\n============================================================")
    print("🎉 雲端網站部署成功！（100% 免費 Static Space，24 小時在線）")
    print(f"👉 獨立全螢幕網站網址：{direct_url}")
    print(f"👉 Hugging Face 頁面：{space_url}")
    print("============================================================")
    return direct_url


def main() -> int:
    ap = argparse.ArgumentParser(description="一鍵部署至 Hugging Face Static Spaces (免費免 PRO)")
    ap.add_argument("--token", default=os.getenv("HF_TOKEN", ""), help="Hugging Face Write Access Token (hf_...)")
    ap.add_argument("--gemini-key", default=os.getenv("GEMINI_API_KEY", ""), help="Google AI Studio 免費 Gemma API Key (AIza...)")
    ap.add_argument("--space-name", default="traffic-verdict-predictor", help="Space 名稱（預設：traffic-verdict-predictor）")
    ap.add_argument("--private", action="store_true", help="是否設為私人 Space（預設為公開）")
    args, unknown = ap.parse_known_args()

    token = (args.token or "").strip()
    gemini_key = (args.gemini_key or "").strip()

    # 自動辨識未帶參數名稱或多餘空白的 hf_... 與 AIza... 金鑰
    for item in unknown:
        s = item.strip()
        if s.startswith("hf_") and not token:
            token = s
        elif s.startswith("AIza") and not gemini_key:
            gemini_key = s

    if not token:
        token = input("請貼上 Hugging Face Write Token (hf_...)：").strip()

    if not token:
        print("未提供 Hugging Face Token，已取消。")
        return 1

    # 自動儲存 HF_TOKEN 至本機 .env（.env 不會上傳至雲端），方便日後直接執行 deploy.cmd 免重複貼上
    env_path = ROOT / ".env"
    if env_path.exists():
        env_text = env_path.read_text(encoding="utf-8")
        if "HF_TOKEN=" not in env_text:
            env_path.write_text(env_text.rstrip() + f"\nHF_TOKEN={token}\n", encoding="utf-8")

    deploy(token=token, space_name=args.space_name, private=args.private, gemini_key=gemini_key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
