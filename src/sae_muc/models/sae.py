"""SAE backends: minimal encode/decode interface for intervention hooks.

`FakeSAEBackend` — fixed random linear projection, deterministic, for
tests and offline smoke.

`SAELensBackend` — pretrained SAE loaded lazily via `sae-lens`. sae-lens
is an optional dependency (install with `uv sync --extra sae`); this
class imports it only inside `_ensure_loaded()` so users who don't touch
SAE branches never pay the install cost.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch

    from sae_muc.config import SAEConfig

log = logging.getLogger(__name__)


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
    `h - decode(encode(h))` is generally non-zero and hooks that preserve
    reconstruction error are exercised.
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


class SAELensBackend:
    """Pretrained SAE loaded via `sae-lens`.

    Construction is cheap — the model is only fetched on the first
    `encode`/`decode` call. `release` + `sae_id` follow sae-lens
    conventions (e.g. release="gemma-scope-2b-pt-res-canonical",
    sae_id="layer_12/width_16k/canonical").
    """

    def __init__(self, release: str, sae_id: str) -> None:
        self.release = release
        self.sae_id = sae_id
        self._sae = None
        self._d_in = 0
        self._d_latent = 0

    def _ensure_loaded(self) -> None:
        if self._sae is not None:
            return
        try:
            from sae_lens import SAE
        except ImportError as e:
            raise ImportError(
                "sae-lens is required to load pretrained SAEs. "
                "Install with `uv sync --extra sae`."
            ) from e

        import torch

        log.info("loading SAE: release=%s sae_id=%s", self.release, self.sae_id)
        sae, _cfg_dict, _sparsity = SAE.from_pretrained(
            release=self.release, sae_id=self.sae_id,
        )
        if torch.cuda.is_available():
            sae = sae.cuda()
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            sae = sae.to("mps")

        self._sae = sae
        self._d_in = int(sae.cfg.d_in)
        self._d_latent = int(sae.cfg.d_sae)

    @property
    def d_in(self) -> int:  # type: ignore[override]
        self._ensure_loaded()
        return self._d_in

    @property
    def d_latent(self) -> int:  # type: ignore[override]
        self._ensure_loaded()
        return self._d_latent

    def encode(self, x: "torch.Tensor") -> "torch.Tensor":
        self._ensure_loaded()
        device = next(self._sae.parameters()).device
        return self._sae.encode(x.to(device))

    def decode(self, f: "torch.Tensor") -> "torch.Tensor":
        self._ensure_loaded()
        return self._sae.decode(f)


def build_sae_backend(cfg: "SAEConfig") -> SAEBackend:
    """Dispatch on `cfg.provider`."""
    if cfg.provider == "fake":
        return FakeSAEBackend(d_in=cfg.d_in, d_latent=cfg.d_latent, seed=cfg.seed)
    if cfg.provider == "sae_lens":
        if not cfg.release or not cfg.sae_id:
            raise ValueError(
                "`sae.release` and `sae.sae_id` are required when provider='sae_lens'"
            )
        return SAELensBackend(release=cfg.release, sae_id=cfg.sae_id)
    raise ValueError(f"Unknown SAE provider: {cfg.provider!r}")
