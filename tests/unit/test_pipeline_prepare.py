from __future__ import annotations

from sae_muc.pipeline import prepare


def test_prepare_writes_samples_parquet(fake_ctx):
    outputs = prepare.run(fake_ctx)
    assert outputs == ["samples.parquet"]
    assert fake_ctx.store.exists("samples.parquet")

    df = fake_ctx.store.load_parquet("samples.parquet")
    assert set(df.columns) == {"sample_id", "question", "gold_answers", "split"}
    assert (df["split"] == "main").all()
    assert len(df) == 5
    assert df["sample_id"].iloc[0].startswith("nq_open:validation:")
