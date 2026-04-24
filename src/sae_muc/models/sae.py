"""SAE backends — minimal encode/decode interface for intervention hooks.

Only `FakeSAEBackend` ships today. The real path (sae-lens pretrained SAEs
loaded by release + sae_id) is added when we wire the sae-lens dependency
and compute per-feature statistics; see `archive/old-prototype/sae_muc/`
for the reference implementation of that analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch


class SAEBackend(Protocol):
    d_in: int
    d_latent: int

    def encode(self, x: "torch.Tensor") -> "torch.Tensor":
        """[..., d_in] -> [..., d_latent]."""
        ...

    def decode(self, f: "torch.Tensor") -> "torch.Tensor":
        """[..., d_latent] -> [..., d_in]."""
        ...


class FakeSAEBackend:
    """Fixed random linear SAE used in tests.

    encode/decode are non-trivial linear maps seeded by `seed`, so
    `h - decode(encode(h))` is generally non-zero and the hook's error-
    reconstruction path is exercised. Real SAEs live behind a separate
    backend.
    """

    def __init__(self, *, d_in: int, d_latent: int = 16, seed: int = 42) -> None:
        import torch

        self.d_in = d_in
        self.d_latent = d_latent
        gen = torch.Generator().manual_seed(seed)
        self._W_enc = torch.randn(d_in, d_latent, generator=gen) * (1.0 / d_in) ** 0.5
        self._W_dec = torch.randn(d_latent, d_in, generator=gen) * (1.0 / d_latent) ** 0.5

    def encode(self, x: "torch.Tensor") -> "torch.Tensor":
        return x @ self._W_enc.to(x.device, dtype=x.dtype)

    def decode(self, f: "torch.Tensor") -> "torch.Tensor":
        return f @ self._W_dec.to(f.device, dtype=f.dtype)


def build_sae_backend(provider: str, *, d_in: int, d_latent: int = 16) -> SAEBackend:
    if provider == "fake":
        return FakeSAEBackend(d_in=d_in, d_latent=d_latent)
    if provider == "sae_lens":
        raise NotImplementedError(
            "sae-lens SAE loading is not wired up yet. See archive/old-prototype/"
            "sae_muc/ for the reference implementation and the SAE feature-analysis "
            "stage; both are planned as follow-ups."
        )
    raise ValueError(f"Unknown SAE provider: {provider!r}")
