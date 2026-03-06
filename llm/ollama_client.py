from __future__ import annotations

import json
import requests


class OllamaLLM:
    def __init__(
        self,
        model: str = "llama3:8b",
        host: str = "http://127.0.0.1:11434",
    ) -> None:
        self.model = model
        self.url = f"{host}/api/generate"

    def generate(self, prompt: str, temperature: float = 0.2) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        response = requests.post(self.url, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["response"]

    def answer(self, question: str, context: dict, temperature: float = 0.2) -> str:
        prompt = (
            "You are an architecture assistant. Use the provided structured context to answer "
            "the user question clearly and concisely.\n\n"
            f"User question:\n{question}\n\n"
            "Context JSON:\n"
            f"{json.dumps(context, indent=2)}\n\n"
            "Answer:"
        )
        return self.generate(prompt=prompt, temperature=temperature)

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
