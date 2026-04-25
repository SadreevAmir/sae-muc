"""CherryIn LLM backend.

`open.cherryin.net` runs the New-API / OneAPI OpenAI-compatible proxy,
so the same openai SDK path works — only `base_url` and the env-var
name for the key change. Same retry/backoff behaviour as OpenRouter.
"""

from __future__ import annotations

import os
import time

from openai import APIError, APITimeoutError, OpenAI, RateLimitError

from sae_muc.models.base import Generation


class MissingAPIKeyError(RuntimeError):
    pass


class CherryInBackend:
    BASE_URL = "https://open.cherryin.net/v1"
    API_KEY_ENV = "CHERRYIN_API_KEY"

    def __init__(self, model: str, *, max_retries: int = 3, timeout: float = 30.0) -> None:
        api_key = os.environ.get(self.API_KEY_ENV, "").strip()
        if not api_key:
            raise MissingAPIKeyError(
                f"{self.API_KEY_ENV} is not set. Add it to .env on the machine running the pipeline."
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
            except (RateLimitError, APITimeoutError, APIError) as e:
                last_exc = e
                time.sleep(delay)
                delay *= 2
        assert last_exc is not None
        raise last_exc
