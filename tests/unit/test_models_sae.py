from __future__ import annotations

import pytest
import torch

from sae_muc.models.sae import FakeSAEBackend, build_sae_backend


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


def test_build_sae_backend_dispatch():
    s = build_sae_backend("fake", d_in=4)
    assert isinstance(s, FakeSAEBackend)


def test_build_sae_backend_sae_lens_stub_raises():
    with pytest.raises(NotImplementedError, match="sae-lens"):
        build_sae_backend("sae_lens", d_in=4)


def test_build_sae_backend_unknown_provider():
    with pytest.raises(ValueError, match="Unknown SAE provider"):
        build_sae_backend("mystery", d_in=4)
