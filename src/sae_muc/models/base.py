"""Common LLM backend interface.

Stages should always consume `LLMBackend`. They do not know whether the
model is local on a GPU, reached over OpenRouter, or replaced by a fake in
tests. Hidden-state extraction and forward-hook steering belong to a
separate, HF-only interface and are wired up when those stages land.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class Generation(BaseModel):
    text: str
    finish_reason: str | None = None


class LLMBackend(Protocol):
    name: str

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
        """Produce n completions per prompt.

        Returns a list of length `len(prompts)`; each element is a list of
        length `n` of `Generation` objects (outer dimension = prompts,
        inner = samples per prompt).

        `seed` is honoured only by backends whose RNG is reachable from the
        caller (HF local). Remote APIs (OpenRouter/CherryIn) ignore it; their
        randomness lives server-side.
        """
        ...
