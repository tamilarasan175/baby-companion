"""Thin client for the local Ollama chat API."""
import os

import httpx

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


class LLMError(Exception):
    pass


async def chat(messages: list, tools: list = None, format=None, timeout: float = 120.0,
               temperature: float = 0.1, num_ctx: int = 8192, num_predict: int = None) -> dict:
    """Send a chat request and return the assistant message dict.

    Raises LLMError on connection problems or non-200 responses so callers can
    fall back instead of silently doing nothing.
    """
    options = {"temperature": temperature, "num_ctx": num_ctx}
    if num_predict:
        options["num_predict"] = num_predict
    payload = {"model": OLLAMA_MODEL, "messages": messages, "stream": False, "options": options}
    if tools:
        payload["tools"] = tools
    if format is not None:
        payload["format"] = format
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
    except httpx.HTTPError as e:
        raise LLMError(f"Cannot reach Ollama at {OLLAMA_URL}: {e!r}")
    if resp.status_code != 200:
        raise LLMError(f"Ollama returned {resp.status_code}: {resp.text[:200]}")
    return resp.json().get("message") or {}
