from __future__ import annotations

import pytest
import torch

from sae_muc.config import SAEConfig
from sae_muc.models.sae import FakeSAEBackend, SAELensBackend, build_sae_backend


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
