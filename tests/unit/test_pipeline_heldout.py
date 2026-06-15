"""End-to-end coverage for the disjoint held-out evaluation feature.

`dataset.heldout_n > 0` carries extra questions through the whole pipeline but
excludes them from every fit (VUF / SAE / detector) and reports their metrics
separately, giving a contamination-free read on the intervention.
"""

from __future__ import annotations

import pandas as pd

from sae_muc.config import ExperimentConfig
from sae_muc.pipeline import build_context, run_all


def _cfg(tmp_path, heldout_n: int) -> ExperimentConfig:
    """All-fakes config on the synthetic `fake` dataset (no HF needed)."""
    return ExperimentConfig.model_validate(
        {
            "model": {"provider": "fake", "name": "fake-7b"},
            "dataset": {
                "name": "fake",
                "split": "validation",
                "n_samples": 4,
                "heldout_n": heldout_n,
                "seed": 7,
            },
            "judge": {"provider": "fake", "model": "fake-judge"},
            "nli": {"provider": "fake", "model": "fake-nli"},
            "sae": {"provider": "fake", "d_in": 8, "d_latent": 16, "seed": 7},
            "data_root": str(tmp_path / "data"),
            "stages": {
                "generate": {"n_samples": 3, "max_new_tokens": 16},
                # adaptive MUC is the setting the held-out feature targets;
                # pin it so the variant dir is the deterministic "adaptive".
                "intervene": {"mode": "adaptive", "method": "linear_vuf"},
            },
        }
    )


def test_heldout_carried_through_but_excluded_from_fit(tmp_path):
    _rid, ctx = build_context(_cfg(tmp_path, heldout_n=2))
    run_all(ctx)

    # samples.parquet: 4 main + 2 disjoint held-out.
    samples = ctx.store.load_parquet("samples.parquet")
    assert (samples["split"] == "main").sum() == 4
    assert (samples["split"] == "heldout").sum() == 2
    main_ids = set(samples.loc[samples["split"] == "main", "sample_id"])
    held_ids = set(samples.loc[samples["split"] == "heldout", "sample_id"])
    assert main_ids.isdisjoint(held_ids)

    # Held-out questions ARE generated/judged (carried through the pipeline).
    gens = ctx.store.load_parquet("generations.parquet")
    assert held_ids.issubset(set(gens["sample_id"]))

    # ...but NEVER enter the VUF fit (split selection is main-only).
    splits = ctx.store.load_parquet("vuf/splits.parquet")
    assert set(splits["sample_id"]).issubset(main_ids)
    assert held_ids.isdisjoint(set(splits["sample_id"]))

    # Per-split metrics: baseline + intervention, both main and held-out.
    assert ctx.store.exists("metrics.json")
    assert ctx.store.exists("metrics.heldout.json")
    assert ctx.store.exists("intervention/adaptive/metrics.heldout.json")

    comp = ctx.store.load_parquet("metrics_comparison.parquet")
    assert set(comp["split"]) == {"main", "heldout"}
    # before + adaptive variant, per split.
    assert set(comp.loc[comp["split"] == "heldout", "variant"]) >= {"before", "adaptive"}


def test_no_heldout_is_backward_compatible(tmp_path):
    _rid, ctx = build_context(_cfg(tmp_path, heldout_n=0))
    run_all(ctx)

    samples = ctx.store.load_parquet("samples.parquet")
    assert (samples["split"] == "main").all()
    assert len(samples) == 4

    # No held-out artefacts when the feature is off.
    assert ctx.store.exists("metrics.json")
    assert not ctx.store.exists("metrics.heldout.json")

    comp = ctx.store.load_parquet("metrics_comparison.parquet")
    assert set(comp["split"]) == {"main"}
