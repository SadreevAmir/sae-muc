"""Unit tests for HF-local decoding params (paper App C: nucleus P=0.9, top-K K=50).

These stub the tokenizer/model so we never load a real HF checkpoint; the
point is to assert that `top_p` / `top_k` are threaded into the `.generate`
kwargs on the sampling path and omitted on the greedy path.
"""

from __future__ import annotations

import pytest

from sae_muc.config import GenerateStage
from sae_muc.models.hf_local import HFLocalBackend, _add_truncation_kwargs

torch = pytest.importorskip("torch")


def test_generate_stage_pins_paper_decoding_defaults():
    # paper App C p.18: temperature 1, nucleus P=0.9, top-K K=50.
    gs = GenerateStage()
    assert gs.top_p == 0.9
    assert gs.top_k == 50


def test_add_truncation_kwargs_conditional():
    kw: dict = {}
    _add_truncation_kwargs(kw, 0.9, 50)
    assert kw == {"top_p": 0.9, "top_k": 50}

    kw2: dict = {}
    _add_truncation_kwargs(kw2, None, None)
    assert kw2 == {}

    kw3: dict = {}
    _add_truncation_kwargs(kw3, 0.95, None)
    assert kw3 == {"top_p": 0.95} and "top_k" not in kw3


class _FakeEnc(dict):
    """Mapping that also exposes `.input_ids` and a no-op `.to()`."""

    def __init__(self, input_ids):
        super().__init__(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids)
        )
        self.input_ids = input_ids

    def to(self, _device):
        return self


class _FakeTok:
    chat_template = None  # -> _apply_chat_template returns the raw prompt
    pad_token = "<pad>"
    pad_token_id = 0
    padding_side = "left"

    def __call__(self, texts, return_tensors=None, padding=None):
        ids = torch.zeros((len(texts), 3), dtype=torch.long)
        return _FakeEnc(ids)

    def batch_decode(self, seqs, skip_special_tokens=True):
        return ["out"] * seqs.shape[0]


class _CaptureModel:
    def __init__(self):
        self.captured: dict | None = None

    def generate(self, **kwargs):
        self.captured = kwargs
        b = kwargs["input_ids"].shape[0]
        n = kwargs["num_return_sequences"]
        return torch.zeros((b * n, 5), dtype=torch.long)


def _stubbed_backend() -> tuple[HFLocalBackend, _CaptureModel]:
    backend = HFLocalBackend("fake-model")
    model = _CaptureModel()
    # Assigning _model makes _ensure_loaded() a no-op (no transformers import).
    backend._model = model
    backend._tokenizer = _FakeTok()
    backend._device = "cpu"
    return backend, model


def test_generate_threads_top_p_top_k_on_sampling():
    backend, model = _stubbed_backend()
    backend.generate(
        ["q"], temperature=1.0, max_new_tokens=8, n=2, top_p=0.9, top_k=50
    )
    assert model.captured["do_sample"] is True
    assert model.captured["top_p"] == 0.9
    assert model.captured["top_k"] == 50


def test_generate_omits_top_p_top_k_on_greedy():
    backend, model = _stubbed_backend()
    # temperature 0 -> do_sample False -> no truncation kwargs at all.
    backend.generate(
        ["q"], temperature=0.0, max_new_tokens=8, n=1, top_p=0.9, top_k=50
    )
    assert model.captured["do_sample"] is False
    assert "top_p" not in model.captured
    assert "top_k" not in model.captured


def test_generate_with_hook_threads_top_p_top_k():
    backend, model = _stubbed_backend()
    backend.generate_with_hook(
        ["q"],
        hook_layer=None,
        hook_fn=None,
        temperature=1.0,
        max_new_tokens=8,
        n=3,
        top_p=0.9,
        top_k=50,
    )
    assert model.captured["top_p"] == 0.9
    assert model.captured["top_k"] == 50
