"""Tests mirror test_models_openrouter but check the CherryIn backend."""

from __future__ import annotations

import pytest

from sae_muc.models import CherryInBackend
from sae_muc.models.cherryin import MissingAPIKeyError


class _FakeChoice:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.message = type("_M", (), {"content": content})()
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, texts: list[str]) -> None:
        self.choices = [_FakeChoice(t) for t in texts]


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        n = kwargs.get("n", 1)
        return _FakeResponse([f"reply-{i}" for i in range(n)])


class _FakeChat:
    def __init__(self, completions: _FakeCompletions) -> None:
        self.completions = completions


class _FakeOpenAI:
    last_kwargs: dict = {}

    def __init__(self, **kwargs) -> None:
        type(self).last_kwargs = kwargs
        self._completions = _FakeCompletions()
        self.chat = _FakeChat(self._completions)


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("CHERRYIN_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        CherryInBackend(model="m")


def test_client_uses_cherryin_base_url(monkeypatch):
    monkeypatch.setenv("CHERRYIN_API_KEY", "k-cherry")
    monkeypatch.setattr("sae_muc.models.cherryin.OpenAI", _FakeOpenAI)

    _ = CherryInBackend(model="m")
    assert _FakeOpenAI.last_kwargs["base_url"] == CherryInBackend.BASE_URL
    assert _FakeOpenAI.last_kwargs["api_key"] == "k-cherry"


def test_generate_returns_expected_shape(monkeypatch):
    monkeypatch.setenv("CHERRYIN_API_KEY", "k")
    monkeypatch.setattr("sae_muc.models.cherryin.OpenAI", _FakeOpenAI)

    backend = CherryInBackend(model="m")
    out = backend.generate(["q1", "q2"], temperature=0.1, max_new_tokens=8, n=2)
    assert len(out) == 2
    assert len(out[0]) == 2
    assert out[0][0].text == "reply-0"


def test_registry_builds_cherryin(monkeypatch):
    from sae_muc.models import build_llm_backend

    monkeypatch.setenv("CHERRYIN_API_KEY", "k")
    monkeypatch.setattr("sae_muc.models.cherryin.OpenAI", _FakeOpenAI)

    backend = build_llm_backend(provider="cherryin", model="qwen/qwen-2.5-72b-instruct")
    assert isinstance(backend, CherryInBackend)
    assert backend.name == "qwen/qwen-2.5-72b-instruct"
