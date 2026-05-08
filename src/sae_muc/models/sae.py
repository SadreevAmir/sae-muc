"""SAE backends + per-layer registry for residual-stream interventions.

Backends (`encode`/`decode` interface):

  * `FakeSAEBackend` — fixed random linear projection, deterministic, for
    tests and offline smoke.
  * `SAELensBackend` — pretrained SAE loaded lazily via `sae-lens`. sae-lens
    is an optional dependency (install with `uv sync --extra sae`); the
    import lives inside `_ensure_loaded()` so non-SAE runs don't pay it.

Multi-layer registry. Gemma-Scope and Llama-Scope SAEs are trained per
residual-stream layer — `layer_15/...` is OOD on layer 20. Paper App E.1
applies the linear-VUF / SAE intervention on a contiguous range of layers
(Llama 15-31, Mistral 15-31, Qwen 16-27), so the pipeline keeps a
`{layer: SAEBackend}` registry instead of a single `ctx.sae`. Helpers:

  * `resolve_sae_id_for_layer(cfg, layer)` — pick the sae_id for `layer`
    honouring `sae.sae_id_overrides > sae.sae_id_template > sae.sae_id`.
  * `build_sae_registry(cfg, layers)` — `{layer: SAEBackend}`. sae-lens
    backends stay lazy (weights load on first encode), so over-covering
    candidate layers in `pipeline.context.build_context` is cheap.
  * `assert_sae_layers_available(cfg, target_layers)` — validates each
    resolved sae_id against `sae_lens.loading.pretrained_saes_directory`
    before any encode is attempted, failing LOUD with a per-layer
    breakdown when a target layer isn't covered (e.g. paper_range Llama
    15-31 against a release that only ships layers 0..14).

Why a hybrid sae_id resolution: Gemma-Scope / Llama-Scope follow a
regular per-layer template (`layer_{layer}/width_16k/canonical`,
`l{layer}r_32x`); `mistral-7b-res-wg` ships only layers 8/16/24, which
overrides express directly. Legacy single-layer configs that set just
`sae.sae_id` keep working (`gemma2_2b_sae_smoke.yaml`).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
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


def resolve_sae_id_for_layer(cfg: "SAEConfig", layer: int) -> str:
    """Pick the sae_id for `layer` honouring overrides > template > legacy sae_id.

    Raises if none of the three is set — the SAEConfig is incomplete for sae-lens.
    """
    if layer in cfg.sae_id_overrides:
        return cfg.sae_id_overrides[layer]
    if cfg.sae_id_template:
        return cfg.sae_id_template.format(layer=layer)
    if cfg.sae_id is not None:
        return cfg.sae_id
    raise ValueError(
        f"SAEConfig: no sae_id resolvable for layer={layer}. Set sae.sae_id "
        f"(single-layer), sae.sae_id_template (e.g. 'layer_{{layer}}/width_16k/canonical'), "
        f"or sae.sae_id_overrides[{layer}]."
    )


def build_sae_registry(
    cfg: "SAEConfig", layers: Iterable[int]
) -> dict[int, SAEBackend]:
    """Build `{layer: SAEBackend}` for the given layers.

    sae-lens backends stay lazy — weights load on the first encode/decode call,
    so over-covering with too many candidate layers is cheap. For provider=fake,
    each layer gets its own seed (cfg.seed + layer) so per-layer mock tests can
    distinguish layers.
    """
    if cfg.provider == "fake":
        return {
            l: FakeSAEBackend(d_in=cfg.d_in, d_latent=cfg.d_latent, seed=cfg.seed + l)
            for l in layers
        }
    if cfg.provider == "sae_lens":
        if not cfg.release:
            raise ValueError("`sae.release` is required when provider='sae_lens'")
        return {
            l: SAELensBackend(release=cfg.release, sae_id=resolve_sae_id_for_layer(cfg, l))
            for l in layers
        }
    raise ValueError(f"Unknown SAE provider: {cfg.provider!r}")


def assert_sae_layers_available(cfg: "SAEConfig", target_layers: list[int]) -> None:
    """Validate that every `target_layers[i]` resolves to an sae_id present in
    `pretrained_saes.yaml` for `cfg.release`. provider=fake → no-op.

    Fails LOUD with a per-layer breakdown so paper_range / explicit-list configs
    that drift past the release's coverage point the user at exactly which
    layers are missing.
    """
    if cfg.provider == "fake":
        return
    from sae_lens.loading.pretrained_saes_directory import (
        get_pretrained_saes_directory,
    )

    directory = get_pretrained_saes_directory()
    if cfg.release not in directory:
        raise ValueError(
            f"sae.release={cfg.release!r} is not in the sae-lens registry "
            f"({len(directory)} releases known; see sae_lens/pretrained_saes.yaml)."
        )
    saes_map = directory[cfg.release].saes_map
    resolved = {l: resolve_sae_id_for_layer(cfg, l) for l in target_layers}
    missing = [(l, sid) for l, sid in resolved.items() if sid not in saes_map]
    if missing:
        breakdown = "\n".join(
            f"  layer {l} -> {sid} {'OK' if sid in saes_map else 'NOT FOUND'}"
            for l, sid in resolved.items()
        )
        raise ValueError(
            f"SAE release {cfg.release!r} does not cover layers "
            f"{[l for l, _ in missing]} (requested target_layers={target_layers}).\n"
            f"Resolved sae_ids:\n{breakdown}\n"
            f"Adjust intervene.layer, switch sae.release, or set "
            f"sae.sae_id_overrides for the missing layers."
        )


def build_sae_backend(cfg: "SAEConfig") -> SAEBackend:
    """Single-backend dispatch — kept for legacy single-layer paths.

    Prefer `build_sae_registry(cfg, layers)` for new code; this wrapper just
    materialises a single backend at layer-id 0 (for fake) or via the resolved
    sae_id (for sae_lens).
    """
    if cfg.provider == "fake":
        return FakeSAEBackend(d_in=cfg.d_in, d_latent=cfg.d_latent, seed=cfg.seed)
    if cfg.provider == "sae_lens":
        if not cfg.release:
            raise ValueError("`sae.release` is required when provider='sae_lens'")
        sae_id = cfg.sae_id or cfg.sae_id_template
        if not sae_id or "{layer}" in sae_id:
            raise ValueError(
                "`sae.sae_id` (or a layer-free sae_id_template) is required for the "
                "single-backend dispatch when provider='sae_lens'."
            )
        return SAELensBackend(release=cfg.release, sae_id=sae_id)
    raise ValueError(f"Unknown SAE provider: {cfg.provider!r}")
