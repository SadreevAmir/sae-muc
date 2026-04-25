from __future__ import annotations

import pandas as pd
import pytest
import torch

from sae_muc.pipeline import vuf
from sae_muc.pipeline._utils import _pool
from sae_muc.pipeline.vuf import _split_ids, _split_ids_by_threshold


# ------------- unit tests on pure helpers ----------------


def test_pool_last_token_q():
    hs = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])  # seq=4, d=2
    # question_len=2 → last question token is index 1 → [2, 0]
    out = _pool(hs, "last_token_q", q_len=2, seq_len=4)
    assert torch.equal(out, torch.tensor([2.0, 0.0]))


def test_pool_mean_q_vs_mean_a():
    hs = torch.arange(8, dtype=torch.float32).reshape(4, 2)  # seq=4, d=2
    mean_q = _pool(hs, "mean_q", q_len=2, seq_len=4)
    mean_a = _pool(hs, "mean_a", q_len=2, seq_len=4)
    assert torch.equal(mean_q, hs[:2].mean(dim=0))
    assert torch.equal(mean_a, hs[2:].mean(dim=0))


def test_pool_mean_a_with_no_answer_falls_back_to_last_question():
    hs = torch.tensor([[1.0, 0.0], [2.0, 0.0]])  # seq=2
    # q_len == seq_len → no answer tokens
    out = _pool(hs, "mean_a", q_len=2, seq_len=2)
    assert torch.equal(out, hs[1])


def test_pool_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown pooling"):
        _pool(torch.zeros(2, 2), "nope", 1, 2)


def test_split_ids_clamps_on_tiny_dataset():
    vu = pd.Series([0.9, 0.8, 0.2, 0.1], index=["a", "b", "c", "d"])
    top, bot = _split_ids(vu, n_top=10, n_bot=10)
    # With 4 questions and both requested >= total, split in half.
    assert len(top) == 2
    assert len(bot) == 2
    assert set(top + bot) == {"a", "b", "c", "d"}
    assert set(top).isdisjoint(bot)


def test_split_ids_order_picks_most_uncertain():
    vu = pd.Series([0.1, 0.9, 0.5, 0.8, 0.2], index=["a", "b", "c", "d", "e"])
    top, bot = _split_ids(vu, n_top=2, n_bot=2)
    assert set(top) == {"b", "d"}  # highest VU
    assert set(bot) == {"a", "e"}  # lowest VU


def test_split_ids_by_threshold_paper_app_g1_default():
    """C4: paper App G.1 splits by VU ≥ 0.9 (uncertain) / ≤ 0.05 (certain)."""
    vu = pd.Series(
        [0.95, 0.91, 0.5, 0.04, 0.0, 0.5],
        index=["u0", "u1", "m0", "c0", "c1", "m1"],
    )
    unc, cer = _split_ids_by_threshold(vu, vu_uncertain_min=0.9, vu_certain_max=0.05)
    assert set(unc) == {"u0", "u1"}
    assert set(cer) == {"c0", "c1"}


def test_split_ids_by_threshold_excludes_middle_band():
    vu = pd.Series([0.6, 0.7, 0.8], index=["a", "b", "c"])
    unc, cer = _split_ids_by_threshold(vu, vu_uncertain_min=0.9, vu_certain_max=0.05)
    assert unc == []
    assert cer == []


# ------------- stage-level tests with hand-crafted artefacts ----------------


def _fake_run_with_signal(fake_ctx, n_questions: int = 10, n_layers: int = 2, d_model: int = 4):
    """Populate artefacts so that uncertain questions carry a +1 signal in dim 0."""
    # samples.parquet
    samples = pd.DataFrame(
        {
            "sample_id": [f"q{i}" for i in range(n_questions)],
            "question": [f"question {i}" for i in range(n_questions)],
            "gold_answers": [["a"] for _ in range(n_questions)],
        }
    )
    fake_ctx.store.save_parquet("samples.parquet", samples)

    # judge_scores.parquet — first half uncertain (VU=0.9), second half certain (VU=0.1).
    judge_rows = []
    for i in range(n_questions):
        vu = 0.9 if i < n_questions // 2 else 0.1
        for j in range(3):
            judge_rows.append(
                {
                    "sample_id": f"q{i}",
                    "kind": "sample",
                    "gen_idx": j,
                    "decisiveness": 1.0 - vu,
                    "vu_score": vu,
                    "raw": str(vu),
                }
            )
    fake_ctx.store.save_parquet("judge_scores.parquet", pd.DataFrame(judge_rows))

    # hidden_states/meta.parquet
    seq_len, q_len = 5, 3
    fake_ctx.store.save_parquet(
        "hidden_states/meta.parquet",
        pd.DataFrame(
            {
                "sample_id": [f"q{i}" for i in range(n_questions)],
                "seq_len": [seq_len] * n_questions,
                "question_len": [q_len] * n_questions,
                "n_layers": [n_layers] * n_questions,
                "answer_len": [seq_len - q_len] * n_questions,
            }
        ),
    )

    # Hidden states per layer: +1 in dim 0 for uncertain, -1 for certain (at every token).
    for layer in range(n_layers):
        tensors: dict[str, torch.Tensor] = {}
        for i in range(n_questions):
            hs = torch.zeros(seq_len, d_model)
            hs[:, 0] = 1.0 if i < n_questions // 2 else -1.0
            tensors[f"q{i}"] = hs
        fake_ctx.store.save_safetensors(f"hidden_states/layer_{layer}.safetensors", tensors)


def test_vuf_writes_per_layer_direction_and_meta(fake_ctx):
    _fake_run_with_signal(fake_ctx, n_questions=10, n_layers=2, d_model=4)
    outputs = vuf.run(fake_ctx)
    # 2 layer files + meta.parquet
    assert "vuf/direction_layer_0.safetensors" in outputs
    assert "vuf/direction_layer_1.safetensors" in outputs
    assert "vuf/meta.parquet" in outputs


def test_vuf_direction_points_at_signal_and_is_unit_norm(fake_ctx):
    _fake_run_with_signal(fake_ctx, n_questions=10, n_layers=2, d_model=4)
    vuf.run(fake_ctx)

    for layer in range(2):
        d = fake_ctx.store.load_safetensors(f"vuf/direction_layer_{layer}.safetensors")["direction"]
        assert d.shape == (4,)
        assert abs(d.norm().item() - 1.0) < 1e-5
        # Signal lives in dim 0 — uncertain +1, certain -1 ⇒ direction must be +e_0.
        assert d[0].item() > 0.99


def test_vuf_respects_explicit_layer_subset(fake_ctx):
    _fake_run_with_signal(fake_ctx, n_questions=8, n_layers=4, d_model=4)

    # Override cfg to request only layer 2.
    new_cfg = fake_ctx.cfg.model_copy(
        update={
            "stages": fake_ctx.cfg.stages.model_copy(
                update={
                    "vuf": fake_ctx.cfg.stages.vuf.model_copy(update={"layers": [2]}),
                }
            )
        }
    )
    object.__setattr__(fake_ctx, "cfg", new_cfg)

    outputs = vuf.run(fake_ctx)
    assert "vuf/direction_layer_2.safetensors" in outputs
    assert "vuf/direction_layer_0.safetensors" not in outputs
    meta = fake_ctx.store.load_parquet("vuf/meta.parquet")
    assert list(meta["layer"]) == [2]


def test_vuf_handles_zero_norm_direction(fake_ctx):
    """When uncertain and certain means coincide, save direction as-is without NaN."""
    n_questions = 6
    samples = pd.DataFrame(
        {
            "sample_id": [f"q{i}" for i in range(n_questions)],
            "question": [f"q {i}" for i in range(n_questions)],
            "gold_answers": [["a"]] * n_questions,
        }
    )
    fake_ctx.store.save_parquet("samples.parquet", samples)
    judge_rows = []
    for i in range(n_questions):
        vu = 0.9 if i < n_questions // 2 else 0.1
        judge_rows.append(
            {
                "sample_id": f"q{i}",
                "kind": "sample",
                "gen_idx": 0,
                "decisiveness": 1.0 - vu,
                "vu_score": vu,
                "raw": "x",
            }
        )
    fake_ctx.store.save_parquet("judge_scores.parquet", pd.DataFrame(judge_rows))

    fake_ctx.store.save_parquet(
        "hidden_states/meta.parquet",
        pd.DataFrame(
            {
                "sample_id": [f"q{i}" for i in range(n_questions)],
                "seq_len": [3] * n_questions,
                "question_len": [2] * n_questions,
                "n_layers": [1] * n_questions,
                "answer_len": [1] * n_questions,
            }
        ),
    )
    # Same hidden states for uncertain and certain → mean difference = 0.
    tensors = {f"q{i}": torch.ones(3, 2) for i in range(n_questions)}
    fake_ctx.store.save_safetensors("hidden_states/layer_0.safetensors", tensors)

    vuf.run(fake_ctx)
    d = fake_ctx.store.load_safetensors("vuf/direction_layer_0.safetensors")["direction"]
    # Zero-norm direction kept as zero (no NaN from division).
    assert torch.allclose(d, torch.zeros_like(d))
