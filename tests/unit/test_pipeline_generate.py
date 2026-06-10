from __future__ import annotations

from sae_muc.pipeline import generate, prepare


def _set_prompt_regime(ctx, regime: str) -> None:
    new_cfg = ctx.cfg.model_copy(
        update={
            "stages": ctx.cfg.stages.model_copy(
                update={
                    "generate": ctx.cfg.stages.generate.model_copy(
                        update={"prompt_regime": regime}
                    )
                }
            )
        }
    )
    object.__setattr__(ctx, "cfg", new_cfg)


def test_generate_split_writes_plain_and_eliciting(fake_ctx):
    prepare.run(fake_ctx)
    outputs = generate.run(fake_ctx)
    assert outputs == ["generations.parquet"]

    df = fake_ctx.store.load_parquet("generations.parquet")
    # split (default): plain + eliciting sets, each 1 greedy + 3 samples.
    # 5 questions × 2 regimes × (1 + 3) = 40 rows.
    assert len(df) == 5 * 2 * (1 + 3)
    assert set(df["kind"].unique()) == {"greedy", "sample"}
    assert set(df["prompt_kind"].unique()) == {"plain", "eliciting"}
    for pk in ("plain", "eliciting"):
        sub = df[df["prompt_kind"] == pk]
        assert (sub["kind"] == "greedy").sum() == 5
        assert (sub["kind"] == "sample").sum() == 5 * 3
        assert sub[sub["kind"] == "sample"]["gen_idx"].max() == 2
    assert df["text"].map(len).min() > 0


def test_generate_eliciting_only_regime(fake_ctx):
    prepare.run(fake_ctx)
    _set_prompt_regime(fake_ctx, "eliciting_only")
    generate.run(fake_ctx)

    df = fake_ctx.store.load_parquet("generations.parquet")
    # eliciting_only: single set, 5 × (1 + 3) = 20 rows, all tagged eliciting.
    assert len(df) == 5 * (1 + 3)
    assert set(df["prompt_kind"].unique()) == {"eliciting"}
