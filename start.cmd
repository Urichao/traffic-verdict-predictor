@echo off
chcp 65001 >nul
echo 正在啟動智慧司法判決預測系統後端 (http://127.0.0.1:8000) ...
cd /d "C:\Users\urichiao\.gemini\antigravity\scratch\traffic-verdict-predictor"
".\.venv\Scripts\uvicorn.exe" main:app --host 127.0.0.1 --port 8000 --reload
