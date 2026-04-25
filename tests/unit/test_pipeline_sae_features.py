from __future__ import annotations

import pandas as pd
import pytest
import torch

from sae_muc.pipeline import sae_features
from sae_muc.pipeline.sae_features import _cohens_d


def test_cohens_d_positive_when_uncertain_larger():
    f_u = torch.tensor([[1.0, 2.0], [1.2, 2.2], [0.8, 1.8]])
    f_c = torch.tensor([[0.0, 0.0], [0.1, 0.0], [-0.1, 0.1]])
    d = _cohens_d(f_u, f_c)
    assert d[0].item() > 0
    assert d[1].item() > 0


def test_cohens_d_negative_when_certain_larger():
    f_u = torch.tensor([[0.0, 0.0], [0.1, 0.0], [-0.1, 0.1]])
    f_c = torch.tensor([[1.0, 2.0], [1.2, 2.2], [0.8, 1.8]])
    d = _cohens_d(f_u, f_c)
    assert d[0].item() < 0
    assert d[1].item() < 0


def test_cohens_d_handles_zero_variance_gracefully():
    """When both groups are constant the pooled-std clamp avoids NaN/Inf."""
    f_u = torch.ones(3, 2)
    f_c = torch.zeros(3, 2)
    d = _cohens_d(f_u, f_c)
    # Finite and positive (uncertain > certain).
    assert torch.isfinite(d).all()
    assert (d > 0).all()


def _seed_sae_features_inputs(fake_ctx):
    # vuf splits — 2 uncertain, 2 certain, 1 middle.
    fake_ctx.store.save_parquet(
        "vuf/splits.parquet",
        pd.DataFrame(
            [
                {"sample_id": "s0", "mean_vu": 0.95, "split": "uncertain"},
                {"sample_id": "s1", "mean_vu": 0.90, "split": "uncertain"},
                {"sample_id": "s2", "mean_vu": 0.50, "split": "middle"},
                {"sample_id": "s3", "mean_vu": 0.05, "split": "certain"},
                {"sample_id": "s4", "mean_vu": 0.10, "split": "certain"},
            ]
        ),
    )
    # vuf meta — 3 layers available.
    fake_ctx.store.save_parquet(
        "vuf/meta.parquet",
        pd.DataFrame(
            {
                "layer": [0, 1, 2],
                "path": [f"vuf/direction_layer_{l}.safetensors" for l in (0, 1, 2)],
                "raw_norm": [1.0, 1.0, 1.0],
                "n_uncertain": [2, 2, 2],
                "n_certain": [2, 2, 2],
                "pooling": ["last_token_q"] * 3,
            }
        ),
    )
    # hidden_states meta + layer 1 tensors with a signal along dim 0 for uncertain.
    fake_ctx.store.save_parquet(
        "hidden_states/meta.parquet",
        pd.DataFrame(
            {
                "sample_id": ["s0", "s1", "s2", "s3", "s4"],
                "seq_len": [5] * 5,
                "question_len": [3] * 5,
                "n_layers": [3] * 5,
                "answer_len": [2] * 5,
            }
        ),
    )
    tensors: dict[str, torch.Tensor] = {}
    for sid in ("s0", "s1", "s2", "s3", "s4"):
        hs = torch.zeros(5, 8)  # FakeBackend _D_MODEL = 8 → FakeSAE d_in = 8
        if sid in {"s0", "s1"}:
            hs[:, 0] = 2.0
        elif sid in {"s3", "s4"}:
            hs[:, 0] = -2.0
        tensors[sid] = hs
    fake_ctx.store.save_safetensors("hidden_states/layer_1.safetensors", tensors)


def test_sae_features_selects_k_top_from_each_tail(fake_ctx):
    # Must request an SAE method so the stage actually runs.
    new_cfg = fake_ctx.cfg.model_copy(
        update={
            "stages": fake_ctx.cfg.stages.model_copy(
                update={
                    "sae_features": fake_ctx.cfg.stages.sae_features.model_copy(
                        update={"k_top": 3}
                    ),
                    "intervene": fake_ctx.cfg.stages.intervene.model_copy(
                        update={"layer": 1, "method": "sae_emd"}
                    ),
                }
            )
        }
    )
    object.__setattr__(fake_ctx, "cfg", new_cfg)

    _seed_sae_features_inputs(fake_ctx)
    outputs = sae_features.run(fake_ctx)
    assert "sae_features/stats.parquet" in outputs

    df = fake_ctx.store.load_parquet("sae_features/stats.parquet")
    # FakeSAE d_latent = 16 (from conftest fake_cfg)
    assert len(df) == 16
    assert set(df.columns) >= {
        "feature_id", "layer", "cohen_d", "mean_uncertain", "mean_certain", "selected_as",
    }
    assert (df["selected_as"] == "uncertainty").sum() == 3
    assert (df["selected_as"] == "certainty").sum() == 3
    assert (df["layer"] == 1).all()
    # All selected uncertainty features have positive d; all certainty negative.
    unc_d = df.loc[df["selected_as"] == "uncertainty", "cohen_d"]
    cer_d = df.loc[df["selected_as"] == "certainty", "cohen_d"]
    assert (unc_d > 0).all()
    assert (cer_d < 0).all()


def test_sae_features_warns_on_tiny_splits(fake_ctx, caplog):
    import logging

    # One uncertain + one certain → sae_features logs a warning.
    fake_ctx.store.save_parquet(
        "vuf/splits.parquet",
        pd.DataFrame(
            [
                {"sample_id": "s0", "mean_vu": 0.9, "split": "uncertain"},
                {"sample_id": "s1", "mean_vu": 0.1, "split": "certain"},
            ]
        ),
    )
    fake_ctx.store.save_parquet(
        "vuf/meta.parquet",
        pd.DataFrame(
            {
                "layer": [1],
                "path": ["vuf/direction_layer_1.safetensors"],
                "raw_norm": [1.0], "n_uncertain": [1], "n_certain": [1],
                "pooling": ["last_token_q"],
            }
        ),
    )
    fake_ctx.store.save_parquet(
        "hidden_states/meta.parquet",
        pd.DataFrame(
            {"sample_id": ["s0", "s1"], "seq_len": [4, 4], "question_len": [2, 2],
             "n_layers": [1, 1], "answer_len": [2, 2]}
        ),
    )
    fake_ctx.store.save_safetensors(
        "hidden_states/layer_1.safetensors",
        {"s0": torch.ones(4, 8), "s1": torch.zeros(4, 8)},
    )

    new_cfg = fake_ctx.cfg.model_copy(
        update={
            "stages": fake_ctx.cfg.stages.model_copy(
                update={
                    "intervene": fake_ctx.cfg.stages.intervene.model_copy(
                        update={"layer": 1, "method": "sae_emd"}
                    ),
                }
            )
        }
    )
    object.__setattr__(fake_ctx, "cfg", new_cfg)

    caplog.set_level(logging.WARNING, logger="sae_muc.pipeline.sae_features")
    sae_features.run(fake_ctx)
    assert any("tiny splits" in r.getMessage() for r in caplog.records)


def test_sae_features_raises_on_d_in_mismatch(fake_ctx):
    """Regression for I3: a wrong SAEConfig.d_in must fail loudly with a clear
    message instead of crashing inside torch matmul during sae.encode."""
    new_cfg = fake_ctx.cfg.model_copy(
        update={
            # Force a mismatch: FakeBackend hidden states are [_, 8] but the SAE
            # encoder is now built to expect 4-dim input.
            "sae": fake_ctx.cfg.sae.model_copy(update={"d_in": 4}),
            "stages": fake_ctx.cfg.stages.model_copy(
                update={
                    "intervene": fake_ctx.cfg.stages.intervene.model_copy(
                        update={"layer": 1, "method": "sae_emd"}
                    ),
                }
            ),
        }
    )
    object.__setattr__(fake_ctx, "cfg", new_cfg)
    # Rebuild the SAE backend so it reflects the updated d_in.
    from sae_muc.models.sae import build_sae_backend

    object.__setattr__(fake_ctx, "sae", build_sae_backend(new_cfg.sae))

    _seed_sae_features_inputs(fake_ctx)
    with pytest.raises(ValueError, match="SAE.d_in=4 != model hidden size d_model=8"):
        sae_features.run(fake_ctx)


def test_sae_features_skipped_for_non_sae_methods(fake_ctx, caplog):
    """When intervene.method is linear_vuf / sae_projected, the stage must no-op."""
    import logging

    caplog.set_level(logging.INFO, logger="sae_muc.pipeline.sae_features")
    # fake_cfg defaults to method=linear_vuf → the stage must early-return.
    outputs = sae_features.run(fake_ctx)
    assert outputs == []
    assert any("skipped" in r.getMessage() for r in caplog.records)
