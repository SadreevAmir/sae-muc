"""OpenRouter-backed LLM backend.

Used for the LLM-as-judge stage. The API key is read from `OPENROUTER_API_KEY`
on construction; missing key raises immediately so the pipeline fails at
setup, not mid-run.
"""

from __future__ import annotations

import os
import time

from openai import APIError, OpenAI, RateLimitError

from sae_muc.models.base import Generation


class MissingAPIKeyError(RuntimeError):
    pass


class OpenRouterBackend:
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, model: str, *, max_retries: int = 3, timeout: float = 60.0) -> None:
        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            raise MissingAPIKeyError(
                "OPENROUTER_API_KEY is not set. Add it to .env on the server."
            )
        self.name = model
        self._model = model
        self._max_retries = max_retries
        self._client = OpenAI(base_url=self.BASE_URL, api_key=api_key, timeout=timeout)

    def generate(
        self,
        prompts: list[str],
        *,
        temperature: float,
        max_new_tokens: int,
        n: int = 1,
        system: str | None = None,
        seed: int | None = None,
    ) -> list[list[Generation]]:
        _ = seed  # remote API: randomness is server-side, ignore the seed
        out: list[list[Generation]] = []
        for prompt in prompts:
            messages: list[dict[str, str]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            resp = self._create_with_retry(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_new_tokens,
                n=n,
            )
            out.append(
                [
                    Generation(text=c.message.content or "", finish_reason=c.finish_reason)
                    for c in resp.choices
                ]
            )
        return out

    def _create_with_retry(self, **kwargs):
        delay = 1.0
        last_exc: Exception | None = None
        for _ in range(self._max_retries):
            try:
                return self._client.chat.completions.create(**kwargs)
            except (RateLimitError, APIError) as e:
                last_exc = e
                time.sleep(delay)
                delay *= 2
        assert last_exc is not None
        raise last_exc
