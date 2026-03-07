from __future__ import annotations

import json
import os
import time
from typing import Any

import requests


class GeminiLLM:
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        host: str = "https://generativelanguage.googleapis.com",
        timeout: int = 600,
        max_retries: int = 4,
        backoff_seconds: float = 1.5,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY or pass api_key.")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.url = (
            f"{host}/v1beta/models/{model}:generateContent?key={self.api_key}"
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        timeout: int | None = None,
    ) -> str:
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
            },
        }

        effective_timeout = timeout if timeout is not None else self.timeout
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.url,
                    json=payload,
                    timeout=effective_timeout,
                )
                if response.status_code in {429, 500, 502, 503, 504}:
                    if attempt < self.max_retries:
                        sleep_s = self._retry_delay_seconds(response, attempt)
                        time.sleep(sleep_s)
                        continue
                response.raise_for_status()
                data = response.json()
                return self._extract_text(data)
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.backoff_seconds * (2 ** attempt))

        if last_error is not None:
            raise last_error
        raise RuntimeError("Gemini request failed without a specific error.")

    def answer(self, question: str, context: dict, temperature: float = 0.2, timeout: int | None = None) -> str:
        prompt = (
            "You are an architecture assistant. Use the provided structured context to answer "
            "the user question clearly and concisely.\n\n"
            f"User question:\n{question}\n\n"
            "Context JSON:\n"
            f"{json.dumps(context, indent=2)}\n\n"
            "Answer:"
        )
        return self.generate(prompt=prompt, temperature=temperature, timeout=timeout)

    def detect_intent(self, question: str) -> str:
        prompt = (
            "Classify the user question into exactly one intent from this set:\n"
            "- impact_analysis\n"
            "- symbol_explanation\n"
            "- architecture_analysis\n\n"
            "Return strict JSON only, no prose:\n"
            '{"intent":"..."}\n\n'
            f"Question: {question}"
        )
        text = self.generate(prompt=prompt, temperature=0.0)
        data = self._parse_json_response(text)
        intent = str(data.get("intent", "")).strip()
        if intent in {"impact_analysis", "symbol_explanation", "architecture_analysis"}:
            return intent
        return "symbol_explanation"

    def extract_symbol(self, question: str) -> str | None:
        prompt = (
            "Extract the primary code symbol mentioned in the question.\n"
            "Examples: GraphBuilder, CodeParser.parse, SymbolResolver.\n"
            'Return strict JSON only: {"symbol":"..."}\n'
            "If unknown, return null symbol: {\"symbol\": null}\n\n"
            f"Question: {question}"
        )
        text = self.generate(prompt=prompt, temperature=0.0)
        data = self._parse_json_response(text)
        symbol = data.get("symbol")
        if symbol is None:
            return None
        cleaned = str(symbol).strip()
        return cleaned or None

    def extract_module(self, question: str) -> str | None:
        prompt = (
            "Extract the primary module/package name from the question.\n"
            "Examples: graph, ingestion, query, context.\n"
            'Return strict JSON only: {"module":"..."}\n'
            "If unknown, return null module: {\"module\": null}\n\n"
            f"Question: {question}"
        )
        text = self.generate(prompt=prompt, temperature=0.0)
        data = self._parse_json_response(text)
        module = data.get("module")
        if module is None:
            return None
        cleaned = str(module).strip()
        return cleaned or None

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        chunks = []
        for part in parts:
            text = part.get("text")
            if text:
                chunks.append(str(text))
        return "".join(chunks).strip()

    @staticmethod
    def _parse_json_response(text: str) -> dict:
        body = text.strip()
        if not body:
            return {}
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            start = body.find("{")
            end = body.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return {}
            snippet = body[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                return {}

    def _retry_delay_seconds(self, response: requests.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return max(float(retry_after), 0.0)
            except ValueError:
                pass
        return self.backoff_seconds * (2 ** attempt)
