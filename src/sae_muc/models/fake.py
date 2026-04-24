"""Deterministic fake LLM backend for unit and integration tests.

Outputs are drawn from a small canned set and seeded by
`sha256(prompt + sample_index + temperature)`, so the same inputs always
yield the same outputs across runs and machines.
"""

from __future__ import annotations

import hashlib
import random
from typing import TYPE_CHECKING

from sae_muc.models.base import Generation

if TYPE_CHECKING:
    import torch


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
    # `d_model` and `n_layers` are fake — small enough for tests but
    # structurally what the real pipeline expects from hidden_states().
    _D_MODEL = 8
    _N_LAYERS = 4

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
                if "Decisiveness score" in prompt:
                    # Judge-shaped prompt — return a deterministic number in [0, 1] so the
                    # judge-parse step produces usable VU scores in full-pipeline tests.
                    score = (int(digest[:8], 16) % 1001) / 1000.0
                    per_prompt.append(Generation(text=f"{score:.3f}", finish_reason="stop"))
                else:
                    rng = random.Random(int(digest[:16], 16))
                    per_prompt.append(Generation(text=rng.choice(_CANNED), finish_reason="stop"))
            out.append(per_prompt)
        return out

    # --- stand-ins for HFLocalBackend so the full pipeline runs on fakes ---

    def tokenize_length(self, text: str, *, add_special_tokens: bool = True) -> int:
        _ = add_special_tokens
        # Whitespace word count — good enough for tests; not a real tokeniser.
        return max(1, len(text.split()))

    def hidden_states(self, texts: list[str]) -> list["torch.Tensor"]:
        """Deterministic synthetic hidden states: [n_layers+1, seq_len, d_model]."""
        import torch

        out: list[torch.Tensor] = []
        for text in texts:
            n_tokens = self.tokenize_length(text)
            digest = hashlib.sha256(text.encode()).digest()[:4]
            seed = int.from_bytes(digest, "big")
            gen = torch.Generator().manual_seed(seed)
            out.append(
                torch.randn(
                    self._N_LAYERS + 1,
                    n_tokens,
                    self._D_MODEL,
                    generator=gen,
                )
            )
        return out
