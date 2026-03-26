from __future__ import annotations

from typing import Any


class GemmaLLM:
    """Stub client for Gemma integration.

    This class is intentionally minimal for now. Replace method bodies with
    real transport logic when ready.
    """

    def __init__(
        self,
        model: str = "gemma",
        host: str = "http://127.0.0.1:11434",
        timeout: int = 600,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.host = host
        self.timeout = timeout
        self.api_key = api_key
        self.last_usage: dict[str, Any] | None = None

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        timeout: int | None = None,
    ) -> str:
        _ = (prompt, temperature, timeout)
        raise NotImplementedError("GemmaLLM.generate is a stub. Implement request logic in llm/gemma_client.py.")

    def answer(
        self,
        question: str,
        context: dict,
        temperature: float = 0.2,
        timeout: int | None = None,
    ) -> str:
        _ = (question, context, temperature, timeout)
        raise NotImplementedError("GemmaLLM.answer is a stub. Implement request logic in llm/gemma_client.py.")

