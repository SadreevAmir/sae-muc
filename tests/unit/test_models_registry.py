from __future__ import annotations

import pytest

from sae_muc.models import FakeBackend, HFLocalBackend, build_llm_backend
from sae_muc.models.openrouter import OpenRouterBackend


def test_build_fake():
    backend = build_llm_backend(provider="fake", model="fake-7b")
    assert isinstance(backend, FakeBackend)
    assert backend.name == "fake-7b"


def test_build_hf_local_is_lazy():
    # __init__ must be cheap (no model download / tokeniser load).
    backend = build_llm_backend(provider="hf_local", model="mistral-7b", dtype="bfloat16")
    assert isinstance(backend, HFLocalBackend)
    assert backend.name == "mistral-7b"


def test_build_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")

    # Avoid hitting the real OpenAI SDK init
    class _FakeClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr("sae_muc.models.openrouter.OpenAI", _FakeClient)

    backend = build_llm_backend(provider="openrouter", model="llama-3.1-70b-instruct")
    assert isinstance(backend, OpenRouterBackend)


def test_build_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown backend provider"):
        build_llm_backend(provider="nope", model="x")
