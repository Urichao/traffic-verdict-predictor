"""呼叫本地端 Gemma-4-E4B（支援 Ollama 與 llama.cpp llama-server）。

針對無獨顯（CPU）環境優化：
- 預設關閉 Gemma 4 的長思考鏈（`think: False`），將首字延遲與總生成時間從 >180 秒大幅縮短至 ~15 秒，
  並支援 SSE 即時串流輸出。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterator

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def _strip_thinking(text: str) -> str:
    """移除模型可能輸出的 <think>...</think> 或 <|thought|>...<|/thought|> 區塊。"""
    for open_tag, close_tag in (("<think>", "</think>"), ("<|thought|>", "<|/thought|>")):
        while open_tag in text and close_tag in text:
            i = text.find(open_tag)
            j = text.find(close_tag, i) + len(close_tag)
            text = (text[:i] + text[j:]).strip()
    return text.strip()


def _is_ollama_url(url: str) -> bool:
    return ":11434" in url or "/api/chat" in url


def _ollama_chat_url(url: str) -> str:
    if "/v1/chat/completions" in url:
        return url.replace("/v1/chat/completions", "/api/chat")
    return url


def _auth_headers() -> dict[str, str]:
    if getattr(config, "LLM_API_KEY", ""):
        return {"Authorization": f"Bearer {config.LLM_API_KEY}"}
    return {}


def chat(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    max_tokens: int = 500,
    json_mode: bool = False,
    timeout: float | None = None,
) -> str:
    """同步呼叫語言模型（自動相容 Ollama `/api/chat`、`llama-server` 與雲端 OpenAI 相容 API）。"""
    if _is_ollama_url(config.LLM_URL):
        body: dict = {
            "model": config.LLM_MODEL,
            "messages": messages,
            "think": False,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            body["format"] = "json"
        r = requests.post(
            _ollama_chat_url(config.LLM_URL),
            json=body,
            headers=_auth_headers(),
            timeout=timeout or config.LLM_TIMEOUT,
        )
        r.raise_for_status()
        content = (r.json().get("message") or {}).get("content") or ""
        return _strip_thinking(content)

    body = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    r = requests.post(
        config.LLM_URL,
        json=body,
        headers=_auth_headers(),
        timeout=timeout or config.LLM_TIMEOUT,
    )
    r.raise_for_status()
    content = r.json()["choices"][0]["message"].get("content") or ""
    return _strip_thinking(content)


def stream_chat(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    max_tokens: int = 500,
    timeout: float | None = None,
) -> Iterator[str]:
    """串流呼叫語言模型，逐段產出文字 token。"""
    if _is_ollama_url(config.LLM_URL):
        body = {
            "model": config.LLM_MODEL,
            "messages": messages,
            "think": False,
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        with requests.post(
            _ollama_chat_url(config.LLM_URL),
            json=body,
            headers=_auth_headers(),
            stream=True,
            timeout=timeout or config.LLM_TIMEOUT,
        ) as r:
            r.raise_for_status()
            for raw_line in r.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8", errors="ignore")
                try:
                    chunk = json.loads(line)
                    piece = (chunk.get("message") or {}).get("content")
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
        return

    body = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    with requests.post(
        config.LLM_URL,
        json=body,
        headers=_auth_headers(),
        stream=True,
        timeout=timeout or config.LLM_TIMEOUT,
    ) as r:
        r.raise_for_status()
        for raw_line in r.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8", errors="ignore")
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
                delta = chunk["choices"][0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    yield piece
            except (KeyError, IndexError, json.JSONDecodeError):
                continue


def parse_json_object(text: str) -> dict:
    """從模型回覆中擷取第一個 JSON 物件。"""
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j < i:
        raise ValueError(f"模型未輸出 JSON：{text[:120]}")
    return json.loads(text[i : j + 1])
