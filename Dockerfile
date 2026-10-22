FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HOME=/home/user \
    OLLAMA_MODELS=/home/user/.ollama/models

# 安裝系統相依套件（LightGBM 需要 libgomp1；Ollama 需要 curl、zstd、ca-certificates）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    ca-certificates \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Ollama（供容器內直接運行 gemma4:e4b）
RUN curl -fsSL https://ollama.com/install.sh | sh

# 建立 Hugging Face Spaces 標準非 root 使用者 (uid=1000)
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY --chown=user:user . /app
RUN sed -i 's/\r$//' /app/docker-entrypoint.sh && chmod +x /app/docker-entrypoint.sh \
    && mkdir -p /home/user/.ollama/models && chown -R user:user /home/user /app

USER user
EXPOSE 7860

CMD ["/app/docker-entrypoint.sh"]
