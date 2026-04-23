from __future__ import annotations

from sae_muc.pipeline import generate, prepare


def test_generate_writes_greedy_plus_n_samples(fake_ctx):
    prepare.run(fake_ctx)
    outputs = generate.run(fake_ctx)
    assert outputs == ["generations.parquet"]

    df = fake_ctx.store.load_parquet("generations.parquet")
    # 5 questions × (1 greedy + 3 samples) = 20 rows
    assert len(df) == 5 * (1 + 3)
    assert set(df["kind"].unique()) == {"greedy", "sample"}
    assert (df[df["kind"] == "sample"]["gen_idx"].max()) == 2
    assert df["text"].map(len).min() > 0
