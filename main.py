"""根目錄啟動入口：直接執行 `uvicorn main:app --reload` 即可啟動系統。"""
from api.main import app  # noqa: F401
