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

    def hs(texts: list[str], *, dtype: str = "float32") -> list[torch.Tensor]:
        _ = dtype
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


def test_hidden_states_question_only_skips_answer(fake_ctx):
    """C7: storage=question_only forwards the question text only, so seq_len
    equals question_len and answer_len is zero on every row."""
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    _install_fake_hidden_state_methods(fake_ctx.llm, d_model=4, n_hidden=3)
    new_cfg = fake_ctx.cfg.model_copy(
        update={
            "stages": fake_ctx.cfg.stages.model_copy(
                update={
                    "hidden_states": fake_ctx.cfg.stages.hidden_states.model_copy(
                        update={"storage": "question_only"}
                    ),
                }
            )
        }
    )
    object.__setattr__(fake_ctx, "cfg", new_cfg)

    hidden_states.run(fake_ctx)
    meta = fake_ctx.store.load_parquet("hidden_states/meta.parquet")
    assert (meta["seq_len"] == meta["question_len"]).all()
    assert (meta["answer_len"] == 0).all()
    assert (meta["storage"] == "question_only").all()


def test_hidden_states_last_k_tokens_truncates_to_window(fake_ctx):
    """C7: storage=last_k_tokens keeps only the trailing `last_k` tokens."""
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    _install_fake_hidden_state_methods(fake_ctx.llm, d_model=4, n_hidden=3)
    new_cfg = fake_ctx.cfg.model_copy(
        update={
            "stages": fake_ctx.cfg.stages.model_copy(
                update={
                    "hidden_states": fake_ctx.cfg.stages.hidden_states.model_copy(
                        update={"storage": "last_k_tokens", "last_k": 4}
                    ),
                }
            )
        }
    )
    object.__setattr__(fake_ctx, "cfg", new_cfg)

    hidden_states.run(fake_ctx)
    meta = fake_ctx.store.load_parquet("hidden_states/meta.parquet")
    layer0 = fake_ctx.store.load_safetensors("hidden_states/layer_0.safetensors")

    # Every kept window has at most last_k tokens.
    assert (meta["seq_len"] <= 4).all()
    for sid in meta["sample_id"]:
        assert layer0[sid].shape[0] <= 4
    assert (meta["storage"] == "last_k_tokens").all()


def _set_hidden_states_dtype(ctx, dtype: str) -> None:
    new_cfg = ctx.cfg.model_copy(
        update={
            "stages": ctx.cfg.stages.model_copy(
                update={
                    "hidden_states": ctx.cfg.stages.hidden_states.model_copy(
                        update={"dtype": dtype}
                    ),
                }
            )
        }
    )
    object.__setattr__(ctx, "cfg", new_cfg)


def test_hidden_states_streams_from_generator_backend(fake_ctx):
    """The stage must consume the backend output lazily: a generator-returning
    `hidden_states` (as HFLocalBackend now is) yields the SAME artefacts as the
    list-returning fake, proving the stage never indexes the full collection."""
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    _install_fake_hidden_state_methods(fake_ctx.llm, d_model=4, n_hidden=3)

    # Snapshot the list-based artefacts.
    hidden_states.run(fake_ctx)
    meta_list = fake_ctx.store.load_parquet("hidden_states/meta.parquet").set_index("sample_id")
    l0_list = fake_ctx.store.load_safetensors("hidden_states/layer_0.safetensors")

    # Re-run with a generator-returning backend (lazy, one tensor at a time).
    list_hs = fake_ctx.llm.hidden_states

    def gen_hs(texts, *, dtype: str = "float32"):
        yield from list_hs(texts)

    fake_ctx.llm.hidden_states = gen_hs
    hidden_states.run(fake_ctx)
    meta_gen = fake_ctx.store.load_parquet("hidden_states/meta.parquet").set_index("sample_id")
    l0_gen = fake_ctx.store.load_safetensors("hidden_states/layer_0.safetensors")

    assert meta_gen["seq_len"].equals(meta_list["seq_len"])
    assert set(l0_gen) == set(l0_list)
    for sid in l0_list:
        assert torch.equal(l0_gen[sid], l0_list[sid])


def test_hidden_states_dtype_bfloat16_artefacts(fake_ctx):
    """dtype=bfloat16 is threaded to the backend AND the saved tensors are bf16
    (the real fix: the HF-local backend casts before .cpu(); the fake echoes the
    arg). Values still round-trip within bf16 tolerance."""
    prepare.run(fake_ctx)
    generate.run(fake_ctx)

    captured: dict = {}

    def hs(texts, *, dtype: str = "float32"):
        captured["dtype"] = dtype
        out = []
        for t in texts:
            n_tokens = max(1, len(t.split()))
            vals = torch.arange(3 * n_tokens * 4, dtype=torch.float32)
            tensor = vals.reshape(3, n_tokens, 4)
            out.append(tensor.bfloat16() if dtype == "bfloat16" else tensor)
        return out

    fake_ctx.llm.tokenize_length = lambda text, add_special_tokens=True: len(text.split())
    fake_ctx.llm.hidden_states = hs
    _set_hidden_states_dtype(fake_ctx, "bfloat16")

    hidden_states.run(fake_ctx)
    assert captured["dtype"] == "bfloat16"
    l0 = fake_ctx.store.load_safetensors("hidden_states/layer_0.safetensors")
    assert all(t.dtype == torch.bfloat16 for t in l0.values())


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
