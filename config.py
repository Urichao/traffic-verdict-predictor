"""全專案共用設定：路徑、環境變數。

所有可調參數都能用環境變數或專案根目錄的 .env 覆寫（見 .env.example）。
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

try:  # python-dotenv 為選用套件
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass

DATA = ROOT / "data"
RAW_DIR = DATA / "raw"              # 下載的每月 RAR
INTERIM_DIR = DATA / "interim"      # 篩選後的車禍民事判決 jsonl（原始全文）
PROCESSED_DIR = DATA / "processed"  # 擷取要點後的 cases.parquet
ARTIFACT_DIR = ROOT / "artifacts"   # 訓練好的模型、索引、評估結果
CASES_PATH = PROCESSED_DIR / "cases.parquet"

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, ARTIFACT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- 司法院資料開放平台 ----
OPENDATA_BASE = "https://opendata.judicial.gov.tw"
# 平台登入有 Cloudflare Turnstile 驗證，無法由程式自動登入。
# 請在瀏覽器登入後，從 Cookie「Authorization」複製值（形如 "Bearer eyJ..."）貼到 .env。
JUDICIAL_TOKEN = os.getenv("JUDICIAL_TOKEN", "")

# ---- 司法院裁判書開放 API（每日 00:00–06:00 提供服務，只能取近 7 日異動）----
JDG_API_BASE = "https://data.judicial.gov.tw/jdg/api"
JUDICIAL_USER = os.getenv("JUDICIAL_USER", "")
JUDICIAL_PASSWORD = os.getenv("JUDICIAL_PASSWORD", "")

SEVEN_ZIP = os.getenv("SEVEN_ZIP", r"C:\Program Files\7-Zip\7z.exe")

# ---- 語言模型（OpenAI 相容介面：Ollama、llama-server 或雲端 OpenAI 相容端點皆可）----
# 沿用雛形的 LLAMA_URL / LLAMA_MODEL 變數名稱
LLM_URL = os.getenv("LLAMA_URL", "http://127.0.0.1:11434/v1/chat/completions")
LLM_MODEL = os.getenv("LLAMA_MODEL", "gemma4:e4b")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "180"))
