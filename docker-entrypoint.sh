#!/usr/bin/env bash
set -e

# 若未指定外部雲端 API（預設使用容器內建 Ollama），則在背景啟動 Ollama 並自動拉取 gemma4:e4b
if [[ "${LLAMA_URL:-http://127.0.0.1:11434/v1/chat/completions}" == *"127.0.0.1:11434"* ]]; then
    echo "[entrypoint] Starting local Ollama service in background..."
    ollama serve >/tmp/ollama.log 2>&1 &
    (
        for i in {1..30}; do
            if curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
                break
            fi
            sleep 1
        done
        echo "[entrypoint] Pulling ${LLAMA_MODEL:-gemma4:e4b} in background..."
        ollama pull "${LLAMA_MODEL:-gemma4:e4b}" >/tmp/ollama_pull.log 2>&1 || true
        echo "[entrypoint] Model ${LLAMA_MODEL:-gemma4:e4b} is ready."
    ) &
fi

echo "[entrypoint] Starting FastAPI server on port ${PORT:-7860}..."
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-7860}"
