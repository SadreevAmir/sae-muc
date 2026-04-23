from __future__ import annotations

import pytest

from sae_muc.models import MissingAPIKeyError
from sae_muc.models.openrouter import OpenRouterBackend


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
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        OpenRouterBackend(model="m")


def test_empty_api_key_raises(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "   ")
    with pytest.raises(MissingAPIKeyError):
        OpenRouterBackend(model="m")


def test_client_constructed_with_base_url_and_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k-abc")
    monkeypatch.setattr("sae_muc.models.openrouter.OpenAI", _FakeOpenAI)

    _ = OpenRouterBackend(model="test-model")
    assert _FakeOpenAI.last_kwargs["base_url"] == OpenRouterBackend.BASE_URL
    assert _FakeOpenAI.last_kwargs["api_key"] == "k-abc"


def test_generate_returns_n_completions_per_prompt(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr("sae_muc.models.openrouter.OpenAI", _FakeOpenAI)

    backend = OpenRouterBackend(model="m")
    result = backend.generate(["q1", "q2"], temperature=0.2, max_new_tokens=50, n=3)

    assert len(result) == 2
    for per_prompt in result:
        assert len(per_prompt) == 3
        assert per_prompt[0].text == "reply-0"
        assert per_prompt[0].finish_reason == "stop"


def test_generate_passes_system_message(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr("sae_muc.models.openrouter.OpenAI", _FakeOpenAI)

    backend = OpenRouterBackend(model="m")
    backend.generate(["hello"], temperature=0, max_new_tokens=10, n=1, system="be brief")

    call = backend._client._completions.calls[0]
    assert call["messages"][0] == {"role": "system", "content": "be brief"}
    assert call["messages"][1] == {"role": "user", "content": "hello"}
