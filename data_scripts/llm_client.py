"""Minimal, provider-agnostic LLM client with JSONL logging and response caching.

Works with any OpenAI-compatible chat-completions endpoint:
    OpenAI   LLM_BASE_URL=https://api.openai.com/v1                      LLM_MODEL=gpt-4o-mini
    Groq     LLM_BASE_URL=https://api.groq.com/openai/v1                 LLM_MODEL=llama-3.1-8b-instant
    Gemini   LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai  LLM_MODEL=gemini-2.0-flash
Set LLM_API_KEY in the environment or a .env file (never commit it).

Every call (including cache hits and dry runs) is appended to logs/api_calls.jsonl
with timestamp, model, prompt, response, and token usage, as evidence of collection.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone

import requests

from common import LOGS, ROOT

LOG_FILE = LOGS / "api_calls.jsonl"
CACHE_FILE = LOGS / "api_cache.jsonl"


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"'))


class LLMClient:
    def __init__(self, purpose: str, dry_run: bool = False, temperature: float = 0.2):
        _load_dotenv()
        self.purpose = purpose
        self.base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.api_key = os.environ.get("LLM_API_KEY")
        self.temperature = temperature
        self.dry_run = dry_run or not self.api_key
        if self.dry_run and not dry_run:
            print("[llm] LLM_API_KEY not set -> dry run (prompts are logged, no API calls)")
        self.cache: dict[str, str] = {}
        if CACHE_FILE.exists():
            for line in CACHE_FILE.open():
                rec = json.loads(line)
                self.cache[rec["key"]] = rec["response"]

    def _key(self, system: str, prompt: str) -> str:
        raw = json.dumps([self.model, self.temperature, system, prompt])
        return hashlib.sha256(raw.encode()).hexdigest()

    def _log(self, **rec) -> None:
        rec = {"timestamp": datetime.now(timezone.utc).isoformat(), "purpose": self.purpose,
               "model": self.model, "base_url": self.base_url, **rec}
        with LOG_FILE.open("a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def chat(self, prompt: str, system: str = "", json_mode: bool = False,
             meta: dict | None = None) -> str | None:
        key = self._key(system, prompt)
        if key in self.cache:
            self._log(system=system, prompt=prompt, response=self.cache[key], cached=True, meta=meta)
            return self.cache[key]
        if self.dry_run:
            self._log(system=system, prompt=prompt, response=None, dry_run=True, meta=meta)
            return None

        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": ([{"role": "system", "content": system}] if system else [])
            + [{"role": "user", "content": prompt}],
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        for attempt in range(5):
            t0 = time.time()
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
                timeout=120,
            )
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"]
            self._log(system=system, prompt=prompt, response=text, usage=data.get("usage"),
                      latency_s=round(time.time() - t0, 2), cached=False, meta=meta)
            with CACHE_FILE.open("a") as f:
                f.write(json.dumps({"key": key, "response": text}) + "\n")
            self.cache[key] = text
            return text
        self._log(system=system, prompt=prompt, response=None, error="retries exhausted", meta=meta)
        return None


def parse_json(text: str | None) -> dict | None:
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None
