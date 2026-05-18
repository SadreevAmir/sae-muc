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
        seed: int | None = None,
    ) -> list[list[Generation]]:
        _ = max_new_tokens, system, seed  # unused; kept for interface parity
        out: list[list[Generation]] = []
        for prompt in prompts:
            per_prompt: list[Generation] = []
            for i in range(n):
                digest = hashlib.sha256(f"{prompt}|{i}|{temperature:.4f}".encode()).hexdigest()
                if "Decisiveness score" in prompt:
                    # VU-judge shape — return a deterministic number in [0, 1] so the
                    # judge-parse step produces usable VU scores in full-pipeline tests.
                    score = (int(digest[:8], 16) % 1001) / 1000.0
                    text = f"{score:.3f}"
                elif "Respond only with yes or no" in prompt:
                    # Accuracy-judge shape — alternate yes/no deterministically by hash.
                    text = "yes" if int(digest[:8], 16) % 2 == 0 else "no"
                else:
                    rng = random.Random(int(digest[:16], 16))
                    text = rng.choice(_CANNED)
                per_prompt.append(Generation(text=text, finish_reason="stop"))
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

    def generate_with_hook(
        self,
        prompts: list[str],
        *,
        hook_layer: int | list[int],
        hook_fn,
        temperature: float,
        max_new_tokens: int,
        n: int = 1,
        system: str | None = None,
        seed: int | None = None,
    ) -> list[list[Generation]]:
        """Probe `hook_fn` to derive a perturbation signature, then generate
        with the signature baked into the prompt so different hooks produce
        different outputs. Multi-layer hooks are summed into a single hint.
        No real activations are touched.
        """
        _ = system, seed
        alpha_hint = self._hook_perturbation_magnitude(hook_layer, hook_fn)
        tagged = [f"[hook:{alpha_hint:+.4f}] {p}" for p in prompts]
        return self.generate(
            tagged,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            n=n,
            system=system,
        )

    # --- diagnostics helpers --------------------------------------------------

    def _hook_perturbation_magnitude(self, hook_layer, hook_fn) -> float:
        """Sum-of-element shift produced by applying `hook_fn` to a constant
        probe tensor. Used by `generate_with_hook` (signature in prompt) and
        by `forward_nll_with_hook` / `forward_kl_with_hook` (drift magnitude).
        Identity hooks return 0.0 exactly; SAE round-trip hooks return a tiny
        float-precision residual even at α=0 — matches the real backend.
        """
        import torch

        if hook_layer is None or hook_fn is None:
            return 0.0
        layers = (
            [int(hook_layer)]
            if isinstance(hook_layer, int)
            else [int(l) for l in hook_layer]
        )
        probe = torch.ones(1, 1, self._D_MODEL)
        total = 0.0
        for l in layers:
            fn = hook_fn[l] if isinstance(hook_fn, dict) else hook_fn
            total += float((fn(probe) - probe).sum().item())
        return total

    def _deterministic_per_token_nll(self, text: str) -> float:
        """A stable, text-dependent per-token NLL in [1.0, 2.0). Drives both
        baseline and hooked perplexity so diagnostics tests can assert
        ratio == 1.0 when the hook is identity.
        """
        digest = hashlib.sha256(text.encode()).digest()[:4]
        return 1.0 + (int.from_bytes(digest, "big") % 1000) / 1000.0

    def forward_nll_with_hook(
        self,
        text: str,
        *,
        hook_layer=None,
        hook_fn=None,
        stride: int = 512,
    ) -> tuple[float, int]:
        _ = stride
        n_tokens = self.tokenize_length(text)
        base = self._deterministic_per_token_nll(text)
        delta = abs(self._hook_perturbation_magnitude(hook_layer, hook_fn))
        # Hooked NLL = baseline + |delta| per token. Identity hook ⇒ delta == 0
        # ⇒ sum_nll matches the baseline forward bit-exactly.
        return (base + delta) * n_tokens, n_tokens

    def forward_kl_with_hook(
        self,
        prompts: list[str],
        *,
        hook_layer,
        hook_fn,
    ) -> dict[str, float]:
        delta = abs(self._hook_perturbation_magnitude(hook_layer, hook_fn))
        n_tokens = sum(self.tokenize_length(p) for p in prompts)
        # KL ≥ 0; identity hook ⇒ delta == 0 ⇒ KL == 0.
        return {
            "mean_kl_all_positions": float(delta),
            "mean_kl_last_prompt_token": float(delta),
            "top1_disagreement_rate": 1.0 if delta > 1e-6 else 0.0,
            "top5_mass_delta": float(delta) * 0.1,
            "n_prompts": int(len(prompts)),
            "n_tokens": int(n_tokens),
        }
