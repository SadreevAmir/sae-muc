from __future__ import annotations

import pytest

from sae_muc.models import FakeNLIBackend, HFLocalNLIBackend, build_nli_backend


def test_fake_nli_entails_case_insensitive_equality():
    nli = FakeNLIBackend()
    assert nli.entails([("Paris", "paris")]) == [True]
    assert nli.entails([("  Paris ", "Paris")]) == [True]
    assert nli.entails([("Paris", "London")]) == [False]


def test_fake_nli_batch():
    nli = FakeNLIBackend()
    pairs = [("a", "a"), ("a", "b"), ("b", "a")]
    assert nli.entails(pairs) == [True, False, False]


def test_hf_local_nli_stub_raises():
    nli = HFLocalNLIBackend("microsoft/deberta-v2-xxlarge-mnli")
    with pytest.raises(NotImplementedError):
        nli.entails([("a", "b")])


def test_build_nli_backend_dispatch():
    assert isinstance(build_nli_backend("fake", "any"), FakeNLIBackend)
    assert isinstance(build_nli_backend("hf_local", "deberta"), HFLocalNLIBackend)


def test_build_nli_backend_unknown_provider():
    with pytest.raises(ValueError, match="Unknown NLI provider"):
        build_nli_backend("wat", "x")
