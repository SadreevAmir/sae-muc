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


def test_hf_local_nli_is_lazy_until_entails_called():
    # Instantiation must be cheap — no model download at this point.
    nli = HFLocalNLIBackend("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    assert nli.name == "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    assert nli._model is None
    assert nli._tokenizer is None
    # Empty batch short-circuits — still no load.
    assert nli.entails([]) == []
    assert nli._model is None


def test_build_nli_backend_dispatch():
    assert isinstance(build_nli_backend("fake", "any"), FakeNLIBackend)
    assert isinstance(build_nli_backend("hf_local", "deberta"), HFLocalNLIBackend)


def test_build_nli_backend_unknown_provider():
    with pytest.raises(ValueError, match="Unknown NLI provider"):
        build_nli_backend("wat", "x")
