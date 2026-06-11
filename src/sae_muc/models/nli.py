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
    """HuggingFace NLI classifier (e.g. MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli).

    Loads the tokenizer and `AutoModelForSequenceClassification` lazily on
    first `entails()` call, locates the "entailment" label by name in
    `config.label2id`, and returns a bool per (premise, hypothesis) pair.

    Runs on CUDA if available, else MPS (Apple Silicon), else CPU.
    """

    def __init__(self, model_name: str) -> None:
        self.name = model_name
        self._model = None
        self._tokenizer = None
        self._entailment_idx: int | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import logging

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        log = logging.getLogger(__name__)
        log.info("Loading NLI model: %s", self.name)
        tokenizer = AutoTokenizer.from_pretrained(self.name)
        # use_safetensors: transformers >=4.57 + torch <2.6 refuses the torch.load
        # (.bin) path (CVE-2025-32434). Force safetensors so NLI models (e.g.
        # deberta-v2-xxlarge-mnli) load under our cu121 / torch-2.5.1 pin.
        model = AutoModelForSequenceClassification.from_pretrained(
            self.name, use_safetensors=True
        )
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            model = model.to("mps")

        label2id = dict(model.config.label2id or {})
        entailment_idx: int | None = None
        for label, idx in label2id.items():
            if label.lower() == "entailment":
                entailment_idx = int(idx)
                break
        if entailment_idx is None:
            raise RuntimeError(
                f"Cannot find 'entailment' label in NLI model {self.name!r}: label2id={label2id}"
            )

        self._tokenizer = tokenizer
        self._model = model
        self._entailment_idx = entailment_idx

    def entails(self, pairs: list[tuple[str, str]]) -> list[bool]:
        if not pairs:
            return []
        import torch

        self._ensure_loaded()
        premises = [p for p, _ in pairs]
        hypotheses = [h for _, h in pairs]
        device = next(self._model.parameters()).device
        inputs = self._tokenizer(
            premises,
            hypotheses,
            padding=True,
            truncation=True,
            # DeBERTa NLI context is 512; the tokenizer's model_max_length is the
            # int sentinel, so without an explicit max_length transformers can't
            # truncate (warns + skips it) and a long answer pair would overflow.
            max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            logits = self._model(**inputs).logits
        preds = logits.argmax(dim=-1).tolist()
        return [p == self._entailment_idx for p in preds]


def build_nli_backend(provider: str, model: str) -> NLIBackend:
    if provider == "fake":
        return FakeNLIBackend(name=model)
    if provider == "hf_local":
        return HFLocalNLIBackend(model_name=model)
    raise ValueError(f"Unknown NLI provider: {provider!r}")
