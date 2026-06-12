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


# --- batched generation: order/structure must match the unbatched path ----- #


class _IdTok:
    """Tokeniser/decoder that round-trips each prompt's integer id.

    The prompt is the string of an int. `__call__` packs that id into the
    single input column; the model replicates rows by num_return_sequences and
    `batch_decode` reconstructs `"<id>#<seq_in_batch>"`, so a result text
    uniquely identifies (which prompt, which sample) — letting us assert that
    chunking preserves order and the n-fold inner grouping.
    """

    chat_template = None
    pad_token = "<pad>"
    pad_token_id = 0
    padding_side = "left"

    def __call__(self, texts, return_tensors=None, padding=None):
        ids = torch.tensor([[int(t)] for t in texts], dtype=torch.long)
        return _FakeEnc(ids)

    def batch_decode(self, seqs, skip_special_tokens=True):
        # seqs rows are [orig_id, seq_in_batch]; rebuild a unique marker.
        return [f"{int(r[0])}#{int(r[1])}" for r in seqs]


class _IdModel:
    """Replicates each input row n times (num_return_sequences) and tags it
    with its position WITHIN the chunk, matching HF's num_return_sequences
    layout `[row0_s0, row0_s1, ..., row1_s0, ...]`."""

    def __init__(self):
        self.batch_sizes: list[int] = []

    def generate(self, **kwargs):
        in_ids = kwargs["input_ids"]
        ids = in_ids[:, 0]  # the per-row id
        in_len = in_ids.shape[1]
        n = kwargs["num_return_sequences"]
        b = ids.shape[0]
        self.batch_sizes.append(b)
        rows = []
        for i in range(b):
            for j in range(n):
                # Mirror HF: output = [prompt tokens ..., generated tokens ...].
                # The backend slices off the first `in_len` cols before decode,
                # so we put the (id, sample) marker AFTER the prompt prefix.
                rows.append([0] * in_len + [int(ids[i]), j])
        return torch.tensor(rows, dtype=torch.long)


def _id_backend() -> tuple[HFLocalBackend, _IdModel]:
    backend = HFLocalBackend("fake-model")
    model = _IdModel()
    backend._model = model
    backend._tokenizer = _IdTok()
    backend._device = "cpu"
    return backend, model


def test_batched_generation_matches_unbatched_order():
    prompts = [str(i) for i in range(7)]

    backend_full, model_full = _id_backend()
    full = backend_full.generate(
        prompts, temperature=1.0, max_new_tokens=4, n=3, seed=0, batch_size=0
    )

    backend_chunked, model_chunked = _id_backend()
    chunked = backend_chunked.generate(
        prompts, temperature=1.0, max_new_tokens=4, n=3, seed=0, batch_size=2
    )

    # Same nested structure and identical, in-order text payloads.
    assert [[g.text for g in row] for row in chunked] == [
        [g.text for g in row] for row in full
    ]
    # Each prompt id i maps to its own row; each row has n=3 samples 0..2.
    assert [[g.text for g in row] for row in chunked] == [
        [f"{i}#{j}" for j in range(3)] for i in range(7)
    ]
    # batch_size=0 ran one forward over all 7 prompts; batch_size=2 chunked
    # into 2+2+2+1, so the GPU batch is capped at 2 (not 7).
    assert model_full.batch_sizes == [7]
    assert model_chunked.batch_sizes == [2, 2, 2, 1]


def test_batched_generation_batch_size_ge_n_is_single_chunk():
    prompts = [str(i) for i in range(4)]
    backend, model = _id_backend()
    backend.generate(
        prompts, temperature=1.0, max_new_tokens=4, n=2, seed=0, batch_size=10
    )
    # batch_size >= len(prompts) collapses to one forward — byte-identical to 0.
    assert model.batch_sizes == [4]


# --- hidden_states streaming + dtype --------------------------------------- #


class _HSModel:
    """Stub causal LM returning a constant 2-layer (+embedding) hidden-state
    tuple, so we can exercise the backend's dtype cast / CPU move / streaming."""

    def __call__(self, *, output_hidden_states, return_dict, **inputs):
        class _Out:
            # tuple of (n_layers+1) tensors, each [1, seq_len, d_model]
            hidden_states = tuple(
                torch.ones(1, 3, 4) * k for k in range(3)
            )

        return _Out()


class _HSTok:
    def __call__(self, text, return_tensors=None):
        return _FakeEnc(torch.zeros((1, 3), dtype=torch.long))


def _hs_backend() -> HFLocalBackend:
    backend = HFLocalBackend("fake-model")
    backend._model = _HSModel()
    backend._tokenizer = _HSTok()
    backend._device = "cpu"
    return backend


def test_hidden_states_yields_lazily_and_casts_dtype():
    backend = _hs_backend()
    it = backend.hidden_states(["a", "b"], dtype="bfloat16")
    # It's a generator (lazy) — nothing forwarded until iterated.
    assert hasattr(it, "__next__")
    out = list(it)
    assert len(out) == 2
    for t in out:
        assert t.shape == (3, 3, 4)
        assert t.dtype == torch.bfloat16
        assert t.device.type == "cpu"


def test_hidden_states_default_dtype_is_float32():
    backend = _hs_backend()
    out = list(backend.hidden_states(["a"]))
    assert out[0].dtype == torch.float32
