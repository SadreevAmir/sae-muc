"""Pipeline-level tests for hidden_states — mock the HF backend.

The real HFLocalBackend implementation is exercised by the server smoke
test (see QUICKSTART / TODO); here we verify that the stage wires inputs
and outputs correctly given a minimal backend that returns synthetic
per-token tensors.
"""

from __future__ import annotations

import torch

from sae_muc.pipeline import generate, hidden_states, prepare


def _install_fake_hidden_state_methods(backend, d_model: int = 4, n_hidden: int = 3):
    """Patch `tokenize_length` and `hidden_states` onto a FakeBackend instance.

    `tokenize_length` is the whitespace-word count. `hidden_states` returns a
    tensor of shape `[n_hidden, n_tokens, d_model]` whose entries encode
    `(layer_idx, token_idx, dim)` so tests can make exact assertions.
    """

    def tokenize_length(text: str, add_special_tokens: bool = True) -> int:
        return len(text.split())

    def hs(texts: list[str]) -> list[torch.Tensor]:
        out = []
        for t in texts:
            n_tokens = max(1, len(t.split()))
            vals = torch.arange(n_hidden * n_tokens * d_model, dtype=torch.float32)
            out.append(vals.reshape(n_hidden, n_tokens, d_model))
        return out

    backend.tokenize_length = tokenize_length
    backend.hidden_states = hs


def test_hidden_states_stage_writes_per_layer_and_meta(fake_ctx):
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    _install_fake_hidden_state_methods(fake_ctx.llm, d_model=4, n_hidden=3)

    outputs = hidden_states.run(fake_ctx)
    # n_layers = 3 - 1 = 2 transformer layers; plus embedding + meta = 4 files.
    assert "hidden_states/embedding.safetensors" in outputs
    assert "hidden_states/layer_0.safetensors" in outputs
    assert "hidden_states/layer_1.safetensors" in outputs
    assert "hidden_states/meta.parquet" in outputs
    assert len(outputs) == 4  # no off-by-one


def test_hidden_states_meta_has_lengths_per_sample(fake_ctx):
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    _install_fake_hidden_state_methods(fake_ctx.llm)

    hidden_states.run(fake_ctx)
    meta = fake_ctx.store.load_parquet("hidden_states/meta.parquet")

    samples = fake_ctx.store.load_parquet("samples.parquet")
    assert set(meta["sample_id"]) == set(samples["sample_id"])
    assert (meta["question_len"] > 0).all()
    assert (meta["seq_len"] >= meta["question_len"]).all()
    assert (meta["answer_len"] == meta["seq_len"] - meta["question_len"]).all()
    assert (meta["n_layers"] == 2).all()


def test_hidden_states_safetensors_shapes(fake_ctx):
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    _install_fake_hidden_state_methods(fake_ctx.llm, d_model=4, n_hidden=3)

    hidden_states.run(fake_ctx)

    meta = fake_ctx.store.load_parquet("hidden_states/meta.parquet").set_index("sample_id")
    layer0 = fake_ctx.store.load_safetensors("hidden_states/layer_0.safetensors")
    layer1 = fake_ctx.store.load_safetensors("hidden_states/layer_1.safetensors")
    emb = fake_ctx.store.load_safetensors("hidden_states/embedding.safetensors")

    # Each sample has a tensor in every layer file with shape [seq_len, d_model].
    for sid, row in meta.iterrows():
        assert layer0[sid].shape == (int(row["seq_len"]), 4)
        assert layer1[sid].shape == (int(row["seq_len"]), 4)
        assert emb[sid].shape == (int(row["seq_len"]), 4)


def test_hidden_states_layer_content_matches_slice(fake_ctx):
    """Layer file i should equal `hidden_list[i+1]` (embedding is layer 0 of the tuple)."""
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    _install_fake_hidden_state_methods(fake_ctx.llm, d_model=4, n_hidden=3)

    hidden_states.run(fake_ctx)

    samples = fake_ctx.store.load_parquet("samples.parquet")
    first_sid = samples["sample_id"].iloc[0]
    emb = fake_ctx.store.load_safetensors("hidden_states/embedding.safetensors")[first_sid]
    l0 = fake_ctx.store.load_safetensors("hidden_states/layer_0.safetensors")[first_sid]
    l1 = fake_ctx.store.load_safetensors("hidden_states/layer_1.safetensors")[first_sid]
    # Our synthetic backend returns `arange`-shaped tensors, so embedding,
    # layer 0 and layer 1 are consecutive blocks of d_model * seq_len entries.
    seq_len, d = emb.shape
    assert torch.allclose(emb.flatten() + seq_len * d, l0.flatten())
    assert torch.allclose(l0.flatten() + seq_len * d, l1.flatten())
