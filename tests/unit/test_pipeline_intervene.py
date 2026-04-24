from __future__ import annotations

import pandas as pd
import pytest
import torch

from sae_muc.pipeline import generate, hidden_states, intervene, prepare
from sae_muc.pipeline.intervene import _alpha_dir, _build_hook, _resolve_layer


# ------------- helpers ----------------


def test_build_hook_adds_alpha_times_direction():
    direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    residual = torch.zeros(2, 3, 4)
    out = _build_hook(direction, alpha=0.5)(residual)
    # Every token, every position → 0.5 * e_0
    expected = torch.zeros(2, 3, 4)
    expected[..., 0] = 0.5
    assert torch.allclose(out, expected)


def test_resolve_layer_auto_picks_middle():
    assert _resolve_layer("auto", [0, 1, 2, 3]) == 2
    assert _resolve_layer("auto", [5, 7, 9]) == 7


def test_resolve_layer_explicit_ok():
    assert _resolve_layer(3, [0, 1, 2, 3]) == 3


def test_resolve_layer_explicit_missing_raises():
    with pytest.raises(ValueError, match="has no VUF direction"):
        _resolve_layer(10, [0, 1, 2])


def test_alpha_dir_format():
    assert _alpha_dir(1.0) == "intervention/alpha_+1.00"
    assert _alpha_dir(-0.5) == "intervention/alpha_-0.50"
    assert _alpha_dir(0.0) == "intervention/alpha_+0.00"


# ------------- stage level ----------------


def _seed_vuf_artefacts(fake_ctx, *, layers: tuple[int, ...] = (0, 1, 2), d_model: int = 8):
    meta = pd.DataFrame(
        {
            "layer": list(layers),
            "path": [f"vuf/direction_layer_{l}.safetensors" for l in layers],
            "raw_norm": [1.0] * len(layers),
            "n_uncertain": [5] * len(layers),
            "n_certain": [5] * len(layers),
            "pooling": ["last_token_q"] * len(layers),
        }
    )
    fake_ctx.store.save_parquet("vuf/meta.parquet", meta)
    for layer in layers:
        # Non-trivial direction so `alpha * direction` is detectable.
        d = torch.zeros(d_model)
        d[layer % d_model] = 1.0
        fake_ctx.store.save_safetensors(
            f"vuf/direction_layer_{layer}.safetensors", {"direction": d.contiguous()}
        )


def test_intervene_writes_per_alpha_files_and_summary(fake_ctx):
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    hidden_states.run(fake_ctx)
    _seed_vuf_artefacts(fake_ctx)

    # Narrow the alpha grid so the test stays fast.
    new_cfg = fake_ctx.cfg.model_copy(
        update={
            "stages": fake_ctx.cfg.stages.model_copy(
                update={
                    "intervene": fake_ctx.cfg.stages.intervene.model_copy(
                        update={"alpha_grid": [-1.0, 0.0, 1.0], "layer": 1},
                    ),
                }
            )
        }
    )
    object.__setattr__(fake_ctx, "cfg", new_cfg)

    outputs = intervene.run(fake_ctx)
    # 3 per-alpha files + intervention/meta.parquet = 4
    for alpha in (-1.0, 0.0, 1.0):
        assert f"intervention/alpha_{alpha:+.2f}/generations.parquet" in outputs
    assert "intervention/meta.parquet" in outputs

    meta = fake_ctx.store.load_parquet("intervention/meta.parquet")
    assert list(meta["alpha"]) == [-1.0, 0.0, 1.0]
    assert (meta["layer"] == 1).all()
    assert (meta["method"] == "linear_vuf").all()


def test_intervene_different_alphas_produce_different_outputs(fake_ctx):
    """FakeBackend folds the hook's perturbation into the prompt, so distinct
    α values must yield distinct generations."""
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    hidden_states.run(fake_ctx)
    _seed_vuf_artefacts(fake_ctx)
    new_cfg = fake_ctx.cfg.model_copy(
        update={
            "stages": fake_ctx.cfg.stages.model_copy(
                update={
                    "intervene": fake_ctx.cfg.stages.intervene.model_copy(
                        update={"alpha_grid": [-1.0, 1.0], "layer": 0},
                    ),
                }
            )
        }
    )
    object.__setattr__(fake_ctx, "cfg", new_cfg)

    intervene.run(fake_ctx)
    neg = fake_ctx.store.load_parquet("intervention/alpha_-1.00/generations.parquet")
    pos = fake_ctx.store.load_parquet("intervention/alpha_+1.00/generations.parquet")
    # Same sample_ids, same shape, but texts should differ for at least some rows.
    assert len(neg) == len(pos)
    assert not (neg["text"].tolist() == pos["text"].tolist())


def test_intervene_auto_layer_picks_middle(fake_ctx):
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    hidden_states.run(fake_ctx)
    _seed_vuf_artefacts(fake_ctx, layers=(0, 1, 2, 3))

    new_cfg = fake_ctx.cfg.model_copy(
        update={
            "stages": fake_ctx.cfg.stages.model_copy(
                update={
                    "intervene": fake_ctx.cfg.stages.intervene.model_copy(
                        update={"alpha_grid": [0.0], "layer": "auto"},
                    ),
                }
            )
        }
    )
    object.__setattr__(fake_ctx, "cfg", new_cfg)
    intervene.run(fake_ctx)
    meta = fake_ctx.store.load_parquet("intervention/meta.parquet")
    assert (meta["layer"] == 2).all()  # 4 layers → middle index 2


def test_intervene_rejects_unimplemented_sae_methods(fake_ctx):
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    hidden_states.run(fake_ctx)
    _seed_vuf_artefacts(fake_ctx)
    for method in ("sae_emd", "sae_clamp"):
        new_cfg = fake_ctx.cfg.model_copy(
            update={
                "stages": fake_ctx.cfg.stages.model_copy(
                    update={
                        "intervene": fake_ctx.cfg.stages.intervene.model_copy(
                            update={"method": method},
                        ),
                    }
                )
            }
        )
        object.__setattr__(fake_ctx, "cfg", new_cfg)
        with pytest.raises(NotImplementedError, match=method):
            intervene.run(fake_ctx)


def test_intervene_sae_projected_runs_end_to_end(fake_ctx):
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    hidden_states.run(fake_ctx)
    _seed_vuf_artefacts(fake_ctx, d_model=8)

    new_cfg = fake_ctx.cfg.model_copy(
        update={
            "stages": fake_ctx.cfg.stages.model_copy(
                update={
                    "intervene": fake_ctx.cfg.stages.intervene.model_copy(
                        update={
                            "method": "sae_projected",
                            "alpha_grid": [-1.0, 1.0],
                            "layer": 1,
                        }
                    ),
                }
            )
        }
    )
    object.__setattr__(fake_ctx, "cfg", new_cfg)

    outputs = intervene.run(fake_ctx)
    for alpha in (-1.0, 1.0):
        assert f"intervention/alpha_{alpha:+.2f}/generations.parquet" in outputs
    meta = fake_ctx.store.load_parquet("intervention/meta.parquet")
    assert (meta["method"] == "sae_projected").all()

    # Different alphas should still yield distinct outputs via the hook probe.
    neg = fake_ctx.store.load_parquet("intervention/alpha_-1.00/generations.parquet")
    pos = fake_ctx.store.load_parquet("intervention/alpha_+1.00/generations.parquet")
    assert not (neg["text"].tolist() == pos["text"].tolist())


def test_sae_projected_hook_preserves_shape_and_changes_output():
    from sae_muc.models.sae import FakeSAEBackend
    from sae_muc.pipeline.intervene import _build_sae_projected_hook

    direction = torch.randn(8)
    sae = FakeSAEBackend(d_in=8)
    hook_zero = _build_sae_projected_hook(direction, sae, alpha=0.0)
    hook_one = _build_sae_projected_hook(direction, sae, alpha=1.0)

    residual = torch.randn(2, 3, 8)
    out_zero = hook_zero(residual)
    out_one = hook_one(residual)

    # Shape and dtype preserved.
    assert out_zero.shape == residual.shape
    assert out_one.shape == residual.shape
    # α=0 → identity up to the SAE's own error round-trip (which the hook
    # undoes by adding `err`), so out_zero ≈ residual.
    assert torch.allclose(out_zero, residual, atol=1e-5)
    # α=1 → output actually differs from residual.
    assert not torch.allclose(out_one, residual, atol=1e-3)
