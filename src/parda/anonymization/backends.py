"""Small OpenAI-compatible clients for llama.cpp and OpenRouter."""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


THINKING = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>|</?think(?:ing)?>", re.I | re.S)


@dataclass
class Exchange:
    request: dict
    response: dict | None = None
    content: str = ""
    usage: dict | None = None
    cost: float | None = None
    error: str | None = None
    attempts: int = 1


class ChatBackend:
    def __init__(self, name: str, base_url: str, model: str, api_key: str | None = None,
                 extra_body: dict | None = None, stream: bool = False, no_think: bool = False,
                 slot: int | None = None, timing_log: Path | None = None, seed: int | None = None,
                 retries: int = 2):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.extra_body = extra_body or {}
        self.stream = stream
        self.no_think = no_think
        self.slot = slot
        self.timing_log = timing_log
        self.seed = seed
        self.retries = retries

    def chat(self, system: str, user: str, temperature: float, top_p: float,
             max_tokens: int | None, label: str) -> Exchange:
        payload = {"model": self.model, "messages": [{"role": "system", "content": system},
                   {"role": "user", "content": user}], "temperature": temperature, "top_p": top_p,
                   **self.extra_body}
        if self.stream:
            payload["stream"] = True
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.no_think:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if self.slot is not None:
            payload.update({"cache_prompt": True, "id_slot": self.slot})
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.base_url + "/chat/completions", json.dumps(payload).encode(), headers)
        started = time.time()
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=600) as response:
                    if self.stream:
                        body = self._read_stream(response, label)
                    else:
                        body = json.loads(response.read().decode())
                content = body["choices"][0]["message"].get("content") or ""
                content = THINKING.sub("", content).strip()
                if not content:
                    raise ValueError("empty completion")
                exchange = Exchange(
                    payload,
                    body,
                    content,
                    body.get("usage"),
                    (body.get("usage") or {}).get("cost"),
                    attempts=attempt + 1,
                )
                break
            except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError, KeyError) as error:
                if attempt < self.retries and self._is_transient(error):
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                exchange = Exchange(payload, error=str(error), attempts=attempt + 1)
                break
        self._write_timing(label, started, payload, exchange)
        return exchange

    # Retry connection failures and service-side errors, but not malformed model output.
    @staticmethod
    def _is_transient(error: Exception) -> bool:
        if isinstance(error, urllib.error.HTTPError):
            return error.code == 429 or error.code >= 500
        return isinstance(error, (urllib.error.URLError, OSError))

    def _read_stream(self, response, label: str) -> dict:
        # Read SSE deltas and display each token as it arrives.
        colour = "\033[96m" if label.startswith("adversary") else "\033[95m"
        print(f"\n{colour}[{label}]\033[0m ", end="", file=sys.stderr, flush=True)
        pieces = []
        for raw_line in response:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            token = delta.get("content") or delta.get("reasoning_content") or ""
            if token:
                print(token, end="", file=sys.stderr, flush=True)
                pieces.append(token)
        print("", file=sys.stderr, flush=True)
        return {"choices": [{"message": {"role": "assistant", "content": "".join(pieces)}}]}

    def _write_timing(self, label: str, started: float, payload: dict, exchange: Exchange) -> None:
        if not self.timing_log:
            return
        record = {"stage": label, "backend": self.name, "model": self.model, "slot": self.slot,
                  "started_unix": started, "elapsed_seconds": time.time() - started,
                  "request_bytes": len(json.dumps(payload).encode()),
                  "response_characters": len(exchange.content), "error": exchange.error,
                  "attempts": exchange.attempts}
        self.timing_log.parent.mkdir(parents=True, exist_ok=True)
        with self.timing_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def get_backend(kind: str, **options) -> ChatBackend:
    if kind == "llama.cpp":
        base_url = options.pop("base_url", os.getenv("LLAMA_CPP_BASE_URL", "http://127.0.0.1:8080/v1"))
        return ChatBackend("llama.cpp", base_url,
                           options.pop("model", None) or os.getenv("LLAMA_CPP_MODEL", "local"), **options)
    if kind == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        providers = options.pop("providers", ["deepseek"])
        effort = options.pop("reasoning_effort", "none")
        body = {"provider": {"order": providers, "allow_fallbacks": False},
                "reasoning": {"enabled": False} if effort == "none" else {"effort": effort},
                "usage": {"include": True}}
        return ChatBackend("openrouter", "https://openrouter.ai/api/v1",
                           options.pop("model", None) or os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash"),
                           key, body, **options)
    raise ValueError("backend must be 'llama.cpp' or 'openrouter'")
