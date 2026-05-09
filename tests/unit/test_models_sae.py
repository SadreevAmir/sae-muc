from __future__ import annotations

import pytest
import torch

from sae_muc.config import SAEConfig
from sae_muc.models.sae import (
    FakeSAEBackend,
    SAELensBackend,
    assert_sae_layers_available,
    build_sae_backend,
    build_sae_registry,
    resolve_sae_id_for_layer,
)


def test_fake_sae_shapes():
    sae = FakeSAEBackend(d_in=4, d_latent=6)
    x = torch.randn(3, 4)
    f = sae.encode(x)
    assert f.shape == (3, 6)
    h = sae.decode(f)
    assert h.shape == (3, 4)


def test_fake_sae_is_deterministic_across_instances():
    a = FakeSAEBackend(d_in=4, d_latent=6, seed=7)
    b = FakeSAEBackend(d_in=4, d_latent=6, seed=7)
    x = torch.randn(5, 4)
    assert torch.allclose(a.encode(x), b.encode(x))


def test_build_sae_backend_dispatch_fake():
    cfg = SAEConfig(provider="fake", d_in=4, d_latent=6)
    s = build_sae_backend(cfg)
    assert isinstance(s, FakeSAEBackend)
    assert s.d_in == 4 and s.d_latent == 6


def test_build_sae_backend_sae_lens_is_lazy():
    # __init__ must be cheap (no sae-lens import, no download) — loading happens
    # on first `.encode` call. We just assert the class exists and __init__ is ok.
    cfg = SAEConfig(
        provider="sae_lens",
        release="gemma-scope-2b-pt-res-canonical",
        sae_id="layer_12/width_16k/canonical",
    )
    backend = build_sae_backend(cfg)
    assert isinstance(backend, SAELensBackend)
    assert backend.release == "gemma-scope-2b-pt-res-canonical"
    assert backend._sae is None  # nothing loaded yet


def test_build_sae_backend_sae_lens_without_release_raises():
    cfg = SAEConfig(provider="sae_lens", release=None, sae_id=None)
    with pytest.raises(ValueError, match="release"):
        build_sae_backend(cfg)


def test_build_sae_backend_unknown_provider():
    cfg = SAEConfig.model_construct(provider="mystery", d_in=4, d_latent=6)
    with pytest.raises(ValueError, match="Unknown SAE provider"):
        build_sae_backend(cfg)


# --- per-layer registry: resolver -------------------------------------------------


def test_resolve_sae_id_template_expands_layer():
    cfg = SAEConfig(
        provider="sae_lens",
        release="gemma-scope-2b-pt-res-canonical",
        sae_id_template="layer_{layer}/width_16k/canonical",
    )
    assert resolve_sae_id_for_layer(cfg, 18) == "layer_18/width_16k/canonical"
    assert resolve_sae_id_for_layer(cfg, 22) == "layer_22/width_16k/canonical"


def test_resolve_sae_id_override_wins_over_template():
    cfg = SAEConfig(
        provider="sae_lens",
        release="gemma-scope-2b-pt-res-canonical",
        sae_id_template="layer_{layer}/width_16k/canonical",
        sae_id_overrides={20: "layer_20/width_16k/average_l0_71"},
    )
    assert resolve_sae_id_for_layer(cfg, 19) == "layer_19/width_16k/canonical"
    assert resolve_sae_id_for_layer(cfg, 20) == "layer_20/width_16k/average_l0_71"


def test_resolve_sae_id_back_compat_single_layer():
    cfg = SAEConfig(
        provider="sae_lens",
        release="gemma-scope-2b-pt-res-canonical",
        sae_id="layer_20/width_16k/canonical",
    )
    # Legacy: same sae_id returned regardless of layer (caller must request only one).
    assert resolve_sae_id_for_layer(cfg, 0) == "layer_20/width_16k/canonical"
    assert resolve_sae_id_for_layer(cfg, 99) == "layer_20/width_16k/canonical"


def test_resolve_sae_id_no_paths_raises():
    cfg = SAEConfig(provider="sae_lens", release="any")
    with pytest.raises(ValueError, match="no sae_id resolvable"):
        resolve_sae_id_for_layer(cfg, 5)


# --- per-layer registry: builder --------------------------------------------------


def test_build_sae_registry_fake_per_layer_seeded():
    cfg = SAEConfig(provider="fake", d_in=4, d_latent=6, seed=7)
    reg = build_sae_registry(cfg, [15, 20])
    assert set(reg.keys()) == {15, 20}
    # Different seeds → different encoders → different output on the same input.
    x = torch.randn(3, 4)
    assert not torch.allclose(reg[15].encode(x), reg[20].encode(x))


def test_build_sae_registry_sae_lens_is_lazy():
    cfg = SAEConfig(
        provider="sae_lens",
        release="gemma-scope-2b-pt-res-canonical",
        sae_id_template="layer_{layer}/width_16k/canonical",
    )
    reg = build_sae_registry(cfg, [18, 19, 20])
    assert set(reg.keys()) == {18, 19, 20}
    assert all(isinstance(b, SAELensBackend) for b in reg.values())
    # Each wrapper carries its own resolved sae_id, weights still unloaded.
    assert reg[19].sae_id == "layer_19/width_16k/canonical"
    assert reg[19]._sae is None


# --- per-layer registry: availability validator -----------------------------------


def _patch_directory(monkeypatch, *, release: str, sae_ids: list[str]) -> None:
    """Stub `get_pretrained_saes_directory` to a single-release fixture."""

    class _Lookup:
        def __init__(self, ids):
            self.saes_map = {sid: sid for sid in ids}

    fake_dir = {release: _Lookup(sae_ids)}
    monkeypatch.setattr(
        "sae_lens.loading.pretrained_saes_directory.get_pretrained_saes_directory",
        lambda: fake_dir,
    )


def test_assert_sae_layers_available_fake_noop():
    cfg = SAEConfig(provider="fake", d_in=4, d_latent=6)
    # Must not raise even with totally bogus layers — fake backends accept anything.
    assert_sae_layers_available(cfg, [0, 1, 99])


def test_assert_sae_layers_available_passes_when_all_present(monkeypatch):
    _patch_directory(
        monkeypatch,
        release="gemma-scope-2b-pt-res-canonical",
        sae_ids=[f"layer_{i}/width_16k/canonical" for i in range(26)],
    )
    cfg = SAEConfig(
        provider="sae_lens",
        release="gemma-scope-2b-pt-res-canonical",
        sae_id_template="layer_{layer}/width_16k/canonical",
    )
    assert_sae_layers_available(cfg, [18, 19, 20, 21, 22])


def test_assert_sae_layers_available_missing_raises_loud(monkeypatch):
    _patch_directory(
        monkeypatch,
        release="gemma-scope-2b-pt-res-canonical",
        # Coverage stops at layer 14 — 15..31 are missing.
        sae_ids=[f"layer_{i}/width_16k/canonical" for i in range(15)],
    )
    cfg = SAEConfig(
        provider="sae_lens",
        release="gemma-scope-2b-pt-res-canonical",
        sae_id_template="layer_{layer}/width_16k/canonical",
    )
    with pytest.raises(ValueError) as exc:
        assert_sae_layers_available(cfg, [13, 14, 15, 16])
    msg = str(exc.value)
    assert "does not cover layers" in msg
    assert "[15, 16]" in msg
    assert "NOT FOUND" in msg


def test_assert_sae_layers_available_unknown_release_raises(monkeypatch):
    _patch_directory(monkeypatch, release="known", sae_ids=["x"])
    cfg = SAEConfig(provider="sae_lens", release="unknown", sae_id="x")
    with pytest.raises(ValueError, match="not in the sae-lens registry"):
        assert_sae_layers_available(cfg, [0])
