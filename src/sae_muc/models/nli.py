"""NLI backends — one for deterministic tests, one HF-local (stub for now).

Semantic entropy clusters answers by *bidirectional* entailment, so the
backend exposes a single `entails(pairs)` helper that takes (premise,
hypothesis) pairs and returns a boolean per pair.
"""

from __future__ import annotations

from typing import Protocol


class NLIBackend(Protocol):
    name: str

    def entails(self, pairs: list[tuple[str, str]]) -> list[bool]:
        """For each (premise, hypothesis), return True iff the premise entails the hypothesis."""
        ...


class FakeNLIBackend:
    """Deterministic stand-in for tests: entailment == normalised string equality."""

    def __init__(self, name: str = "fake-nli") -> None:
        self.name = name

    def entails(self, pairs: list[tuple[str, str]]) -> list[bool]:
        return [a.strip().lower() == b.strip().lower() for a, b in pairs]


class HFLocalNLIBackend:
    """HuggingFace NLI (e.g. DeBERTa-v2-MNLI). Stub for now.

    The real implementation — `AutoModelForSequenceClassification.from_pretrained(...)`
    with entailment argmax — lands together with the server integration step.
    The stub exists so the registry can already dispatch on `provider='hf_local'`
    and the pipeline fails with a clear message if it is invoked before then.
    """

    def __init__(self, model_name: str) -> None:
        self.name = model_name

    def entails(self, pairs: list[tuple[str, str]]) -> list[bool]:
        raise NotImplementedError(
            "HFLocalNLIBackend.entails is not implemented yet. "
            "It lands with the server integration step."
        )


def build_nli_backend(provider: str, model: str) -> NLIBackend:
    if provider == "fake":
        return FakeNLIBackend(name=model)
    if provider == "hf_local":
        return HFLocalNLIBackend(model_name=model)
    raise ValueError(f"Unknown NLI provider: {provider!r}")
