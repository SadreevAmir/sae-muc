from __future__ import annotations

import math

import pandas as pd

from sae_muc.models import FakeNLIBackend
from sae_muc.pipeline import generate, prepare, semantic_entropy
from sae_muc.pipeline.semantic_entropy import _cluster_by_entailment, _entropy


def test_cluster_identical_answers_go_to_one_bucket():
    nli = FakeNLIBackend()
    assert _cluster_by_entailment(["paris", "paris", "paris"], nli) == [0, 0, 0]


def test_cluster_distinct_answers_are_separate():
    nli = FakeNLIBackend()
    assert _cluster_by_entailment(["paris", "berlin", "london"], nli) == [0, 1, 2]


def test_cluster_mixed():
    nli = FakeNLIBackend()
    # "Paris" and "paris" are equal under FakeNLI normalisation.
    result = _cluster_by_entailment(["Paris", "Berlin", "paris"], nli)
    assert result == [0, 1, 0]


def test_entropy_of_uniform_3_clusters():
    # 3 clusters of size 1 → H = ln(3)
    assert _entropy([0, 1, 2]) == pytest.approx(math.log(3))


def test_entropy_of_single_cluster_is_zero():
    assert _entropy([0, 0, 0, 0]) == 0.0


def test_entropy_empty():
    assert _entropy([]) == 0.0


def test_semantic_entropy_stage_writes_parquet(fake_ctx, monkeypatch):
    # Replace generations with a hand-crafted set so we know the expected clusters.
    prepare.run(fake_ctx)
    generate.run(fake_ctx)  # produces something; we then overwrite it below

    # Build a synthetic generations frame: 3 questions, 3 samples each.
    samples = fake_ctx.store.load_parquet("samples.parquet")
    sid = samples["sample_id"].tolist()
    rows = []
    # q0: all same → 1 cluster
    for j, t in enumerate(["paris", "paris", "paris"]):
        rows.append({"sample_id": sid[0], "kind": "sample", "gen_idx": j, "text": t, "finish_reason": "stop"})
    # q1: two clusters {paris, paris}, {berlin}
    for j, t in enumerate(["paris", "paris", "berlin"]):
        rows.append({"sample_id": sid[1], "kind": "sample", "gen_idx": j, "text": t, "finish_reason": "stop"})
    # q2: three clusters (all distinct)
    for j, t in enumerate(["a", "b", "c"]):
        rows.append({"sample_id": sid[2], "kind": "sample", "gen_idx": j, "text": t, "finish_reason": "stop"})

    fake_ctx.store.save_parquet("generations.parquet", pd.DataFrame(rows))

    outputs = semantic_entropy.run(fake_ctx)
    assert outputs == ["semantic_entropy.parquet"]

    df = fake_ctx.store.load_parquet("semantic_entropy.parquet").set_index("sample_id")
    assert df.loc[sid[0], "n_clusters"] == 1
    assert df.loc[sid[0], "semantic_entropy"] == 0.0

    assert df.loc[sid[1], "n_clusters"] == 2
    # entropy of (2/3, 1/3)
    expected = -((2 / 3) * math.log(2 / 3) + (1 / 3) * math.log(1 / 3))
    assert df.loc[sid[1], "semantic_entropy"] == pytest.approx(expected)

    assert df.loc[sid[2], "n_clusters"] == 3
    assert df.loc[sid[2], "semantic_entropy"] == pytest.approx(math.log(3))


def test_semantic_entropy_stage_ignores_greedy_rows(fake_ctx):
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    se_out = semantic_entropy.run(fake_ctx)
    df = fake_ctx.store.load_parquet(se_out[0])
    gens = fake_ctx.store.load_parquet("generations.parquet")
    # n_samples recorded per question equals the number of sample-kind rows per question.
    per_q = gens[gens["kind"] == "sample"].groupby("sample_id").size()
    for sid_, n in per_q.items():
        assert df.loc[df["sample_id"] == sid_, "n_samples"].iloc[0] == n


import pytest  # noqa: E402  (imported at the bottom to avoid flake on parametrize above)
