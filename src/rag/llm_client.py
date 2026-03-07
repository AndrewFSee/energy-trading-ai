"""LLM client interface for RAG and report generation.

Provides a unified interface for both OpenAI GPT-4o and local Ollama
models.  Handles prompt construction, streaming, retry logic, and
token usage tracking.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert energy market analyst and quantitative trader. "
    "You have deep knowledge of crude oil, natural gas, and energy derivatives markets. "
    "Provide concise, data-driven analysis grounded in the retrieved context. "
    "Always cite your sources and quantify uncertainty where possible."
)


class LLMClient:
    """Unified LLM interface supporting OpenAI and Ollama backends.

    Attributes:
        provider: LLM provider (``"openai"`` or ``"ollama"``).
        model: Model identifier.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.
        system_prompt: Default system prompt.
        api_key: OpenAI API key (if applicable).
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        api_key: str | None = None,
        ollama_base_url: str = "http://localhost:11434",
    ) -> None:
        """Initialise the LLM client.

        Args:
            provider: Backend provider (``"openai"`` or ``"ollama"``).
            model: Model name.  Defaults to ``"gpt-4o"`` for OpenAI,
                ``"llama3"`` for Ollama.
            temperature: Generation temperature (lower = more deterministic).
            max_tokens: Maximum response length in tokens.
            system_prompt: System message for the LLM.
            api_key: OpenAI API key (reads from env if not provided).
            ollama_base_url: Base URL for the local Ollama server.
        """
        self.provider = provider
        if model is None:
            model = "gpt-4o" if provider == "openai" else "llama3"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.ollama_base_url = ollama_base_url
        self._client = None
        logger.info("LLMClient initialised (provider=%s, model=%s)", provider, model)

    def _get_openai_client(self):  # type: ignore[return]
        """Lazy-load OpenAI client."""
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
            return self._client
        except ImportError:
            logger.error("openai package not installed")
            raise

    def complete(
        self,
        user_message: str,
        system_prompt: str | None = None,
        context: str | None = None,
    ) -> str:
        """Generate a completion for the given user message.

        Args:
            user_message: The user's question or instruction.
            system_prompt: Override the default system prompt.
            context: Optional context string to prepend to the user message
                (e.g. retrieved document chunks).

        Returns:
            Generated text response.
        """
        sys_prompt = system_prompt or self.system_prompt
        if context:
            full_message = f"Context:\n{context}\n\n---\n\nQuestion: {user_message}"
        else:
            full_message = user_message

        if self.provider == "openai":
            return self._complete_openai(full_message, sys_prompt)
        if self.provider == "ollama":
            return self._complete_ollama(full_message, sys_prompt)
        raise ValueError(f"Unknown LLM provider: '{self.provider}'")

    def _complete_openai(self, message: str, system_prompt: str) -> str:
        """Generate a completion using the OpenAI API.

        Args:
            message: User message with optional context.
            system_prompt: System instruction.

        Returns:
            Generated response text.
        """
        client = self._get_openai_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            result = response.choices[0].message.content or ""
            logger.info(
                "OpenAI completion: %d tokens used",
                response.usage.total_tokens if response.usage else 0,
            )
            return result
        except Exception as exc:
            logger.error("OpenAI completion failed: %s", exc)
            raise

    def _complete_ollama(self, message: str, system_prompt: str) -> str:
        """Generate a completion using a local Ollama server.

        Args:
            message: User message.
            system_prompt: System instruction.

        Returns:
            Generated response text.
        """
        try:
            import httpx

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message},
                ],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            }
            url = f"{self.ollama_base_url}/api/chat"
            resp = httpx.post(url, json=payload, timeout=120.0)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except Exception as exc:
            logger.error("Ollama completion failed: %s", exc)
            raise
