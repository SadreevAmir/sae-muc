"""combine_vuf: universal / OOD VUF pooling (paper App G.1 / Table 5)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

import torch

from sae_muc.artifacts.store import ArtifactStore
from sae_muc.pipeline import combine_vuf


def _seed_source(ctx, name, *, mean_uncertain, mean_certain, n_uncertain, n_certain, layer=0):
    """Create a source run dir with vuf/means + vuf/meta for one layer."""
    store = ArtifactStore(ctx.cfg.data_root / "runs" / name)
    store.save_safetensors(
        f"vuf/means_layer_{layer}.safetensors",
        {
            "mean_uncertain": torch.tensor(mean_uncertain, dtype=torch.float32),
            "mean_certain": torch.tensor(mean_certain, dtype=torch.float32),
        },
    )
    store.save_parquet(
        "vuf/meta.parquet",
        pd.DataFrame(
            {
                "layer": [layer],
                "path": [f"vuf/direction_layer_{layer}.safetensors"],
                "raw_norm": [1.0],
                "n_uncertain": [n_uncertain],
                "n_certain": [n_certain],
                "pooling": ["last_token_q"],
            }
        ),
    )


def _set_combine_sources(ctx, sources):
    new_cfg = ctx.cfg.model_copy(
        update={
            "stages": ctx.cfg.stages.model_copy(
                update={"vuf": ctx.cfg.stages.vuf.model_copy(update={"combine_sources": sources})}
            )
        }
    )
    object.__setattr__(ctx, "cfg", new_cfg)


def test_combine_vuf_noop_without_sources(fake_ctx):
    assert combine_vuf.run(fake_ctx) == []


def test_combine_vuf_pools_means_weighted_by_counts(fake_ctx):
    # A: uncertain=[2,0] (n=10), B: uncertain=[0,2] (n=30); certain=[0,0] both.
    _seed_source(fake_ctx, "srcA", mean_uncertain=[2.0, 0.0], mean_certain=[0.0, 0.0],
                 n_uncertain=10, n_certain=10)
    _seed_source(fake_ctx, "srcB", mean_uncertain=[0.0, 2.0], mean_certain=[0.0, 0.0],
                 n_uncertain=30, n_certain=10)
    _set_combine_sources(fake_ctx, ["srcA", "srcB"])

    outputs = combine_vuf.run(fake_ctx)
    assert "vuf/direction_layer_0.safetensors" in outputs

    direction = fake_ctx.store.load_safetensors("vuf/direction_layer_0.safetensors")["direction"]
    # pooled_uncertain = (10·[2,0] + 30·[0,2]) / 40 = [0.5, 1.5]; certain = 0.
    raw = torch.tensor([0.5, 1.5])
    expected = raw / raw.norm()
    assert torch.allclose(direction, expected, atol=1e-6)
    # Universal meta records the pooled counts.
    meta = fake_ctx.store.load_parquet("vuf/meta.parquet")
    row = meta[meta["layer"] == 0].iloc[0]
    assert int(row["n_uncertain"]) == 40 and int(row["n_certain"]) == 20


def test_combine_vuf_cross_dataset_cosine_diagnostic(fake_ctx):
    _seed_source(fake_ctx, "srcA", mean_uncertain=[2.0, 0.0], mean_certain=[0.0, 0.0],
                 n_uncertain=10, n_certain=10)
    _seed_source(fake_ctx, "srcB", mean_uncertain=[0.0, 2.0], mean_certain=[0.0, 0.0],
                 n_uncertain=10, n_certain=10)
    _set_combine_sources(fake_ctx, ["srcA", "srcB"])

    combine_vuf.run(fake_ctx)
    cos = fake_ctx.store.load_parquet("vuf/cross_dataset_cosine.parquet")
    # Orthogonal source VUFs ([1,0] vs [0,1]) → cosine 0.
    assert cos.loc[cos["layer"] == 0, "mean_cross_dataset_cosine"].iloc[0] == pytest.approx(0.0, abs=1e-6)
    assert int(cos.loc[cos["layer"] == 0, "n_sources"].iloc[0]) == 2


def test_combine_vuf_single_source_is_ood_reuse(fake_ctx):
    # A single source reproduces that source's VUF (Table 5 OOD reuse).
    _seed_source(fake_ctx, "srcA", mean_uncertain=[2.0, 0.0], mean_certain=[0.0, 0.0],
                 n_uncertain=10, n_certain=10)
    _set_combine_sources(fake_ctx, ["srcA"])

    combine_vuf.run(fake_ctx)
    direction = fake_ctx.store.load_safetensors("vuf/direction_layer_0.safetensors")["direction"]
    assert torch.allclose(direction, torch.tensor([1.0, 0.0]), atol=1e-6)
    # No cosine diagnostic with a single source.
    assert not fake_ctx.store.exists("vuf/cross_dataset_cosine.parquet")


def test_combine_vuf_unknown_source_raises(fake_ctx):
    _set_combine_sources(fake_ctx, ["does-not-exist"])
    with pytest.raises(ValueError, match="not found"):
        combine_vuf.run(fake_ctx)
