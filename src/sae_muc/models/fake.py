"""Deterministic fake LLM backend for unit and integration tests.

Outputs are drawn from a small canned set and seeded by
`sha256(prompt + sample_index + temperature)`, so the same inputs always
yield the same outputs across runs and machines.
"""

from __future__ import annotations

import hashlib
import random

from sae_muc.models.base import Generation


_CANNED = [
    "I don't know.",
    "I am not sure.",
    "Paris.",
    "Probably 1945.",
    "Maybe Leo Tolstoy.",
    "It is the River Thames.",
    "I cannot answer that with confidence.",
    "The first president was George Washington.",
]


class FakeBackend:
    def __init__(self, name: str = "fake") -> None:
        self.name = name

    def generate(
        self,
        prompts: list[str],
        *,
        temperature: float,
        max_new_tokens: int,
        n: int = 1,
        system: str | None = None,
    ) -> list[list[Generation]]:
        _ = max_new_tokens, system  # unused; kept for interface parity
        out: list[list[Generation]] = []
        for prompt in prompts:
            per_prompt: list[Generation] = []
            for i in range(n):
                digest = hashlib.sha256(f"{prompt}|{i}|{temperature:.4f}".encode()).hexdigest()
                rng = random.Random(int(digest[:16], 16))
                per_prompt.append(Generation(text=rng.choice(_CANNED), finish_reason="stop"))
            out.append(per_prompt)
        return out
