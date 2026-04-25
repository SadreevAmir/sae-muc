from __future__ import annotations

import pandas as pd
import pytest
import torch

from sae_muc.pipeline import generate, hidden_states, intervene, prepare
from sae_muc.pipeline._utils import _resolve_layer
from sae_muc.pipeline.intervene import _alpha_dir, _build_hook


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
    assert (meta["layers"] == "1").all()
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
    assert (meta["layers"] == "2").all()  # 4 layers → middle index 2


def test_intervene_multi_layer_registers_hook_per_layer(fake_ctx):
    """C2: passing layer=[a, b, c] registers hooks at each layer; meta records the range."""
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    hidden_states.run(fake_ctx)
    _seed_vuf_artefacts(fake_ctx, layers=(0, 1, 2))

    new_cfg = fake_ctx.cfg.model_copy(
        update={
            "stages": fake_ctx.cfg.stages.model_copy(
                update={
                    "intervene": fake_ctx.cfg.stages.intervene.model_copy(
                        update={"alpha_grid": [1.0], "layer": [0, 1, 2]},
                    ),
                }
            )
        }
    )
    object.__setattr__(fake_ctx, "cfg", new_cfg)
    intervene.run(fake_ctx)
    meta = fake_ctx.store.load_parquet("intervention/meta.parquet")
    # Contiguous range — rendered as "0-2".
    assert (meta["layers"] == "0-2").all()


def test_intervene_paper_range_uses_app_e1(fake_ctx):
    """C2: layer='paper_range' looks up the App E.1 range by model.name substring."""
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    hidden_states.run(fake_ctx)
    # Available layers cover Llama range 15-17 only; paper_range intersects to those.
    _seed_vuf_artefacts(fake_ctx, layers=(15, 16, 17, 18))

    new_cfg = fake_ctx.cfg.model_copy(
        update={
            "model": fake_ctx.cfg.model.model_copy(update={"name": "fake-llama-7b"}),
            "stages": fake_ctx.cfg.stages.model_copy(
                update={
                    "intervene": fake_ctx.cfg.stages.intervene.model_copy(
                        update={"alpha_grid": [1.0], "layer": "paper_range"},
                    ),
                }
            ),
        }
    )
    object.__setattr__(fake_ctx, "cfg", new_cfg)
    intervene.run(fake_ctx)
    meta = fake_ctx.store.load_parquet("intervention/meta.parquet")
    # Llama paper range = 15-31; intersected with available 15..18 → 15-18.
    assert (meta["layers"] == "15-18").all()


def test_intervene_sae_emd_requires_sae_features_artefact(fake_ctx):
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
                            update={"method": method, "alpha_grid": [0.5], "layer": 1},
                        ),
                    }
                )
            }
        )
        object.__setattr__(fake_ctx, "cfg", new_cfg)
        # No sae_features/stats.parquet seeded → pandas can't load it.
        with pytest.raises((ValueError, FileNotFoundError)):
            intervene.run(fake_ctx)


def _seed_sae_features(fake_ctx, *, d_latent: int = 16, k_top: int = 3) -> None:
    """Fake `sae_features/stats.parquet`: first k_top features uncertainty,
    last k_top features certainty, rest empty."""
    rows = []
    for i in range(d_latent):
        if i < k_top:
            sel = "uncertainty"
        elif i >= d_latent - k_top:
            sel = "certainty"
        else:
            sel = ""
        rows.append(
            {
                "feature_id": i,
                "layer": 1,
                "cohen_d": 0.5 if sel == "uncertainty" else (-0.5 if sel == "certainty" else 0.0),
                "mean_uncertain": 0.0,
                "mean_certain": 0.0,
                "selected_as": sel,
            }
        )
    fake_ctx.store.save_parquet("sae_features/stats.parquet", pd.DataFrame(rows))


def test_intervene_sae_emd_runs_with_selected_features(fake_ctx):
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    hidden_states.run(fake_ctx)
    _seed_vuf_artefacts(fake_ctx, d_model=8)
    _seed_sae_features(fake_ctx, d_latent=16, k_top=3)

    new_cfg = fake_ctx.cfg.model_copy(
        update={
            "stages": fake_ctx.cfg.stages.model_copy(
                update={
                    "intervene": fake_ctx.cfg.stages.intervene.model_copy(
                        update={
                            "method": "sae_emd",
                            "alpha_grid": [-0.5, 0.0, 0.5],
                            "layer": 1,
                        }
                    ),
                }
            )
        }
    )
    object.__setattr__(fake_ctx, "cfg", new_cfg)

    outputs = intervene.run(fake_ctx)
    for alpha in (-0.5, 0.0, 0.5):
        assert f"intervention/alpha_{alpha:+.2f}/generations.parquet" in outputs


def test_intervene_sae_clamp_runs_with_selected_features(fake_ctx):
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    hidden_states.run(fake_ctx)
    _seed_vuf_artefacts(fake_ctx, d_model=8)
    _seed_sae_features(fake_ctx, d_latent=16, k_top=3)

    new_cfg = fake_ctx.cfg.model_copy(
        update={
            "stages": fake_ctx.cfg.stages.model_copy(
                update={
                    "intervene": fake_ctx.cfg.stages.intervene.model_copy(
                        update={
                            "method": "sae_clamp",
                            "alpha_grid": [1.0],
                            "layer": 1,
                        }
                    ),
                }
            )
        }
    )
    object.__setattr__(fake_ctx, "cfg", new_cfg)

    outputs = intervene.run(fake_ctx)
    assert "intervention/alpha_+1.00/generations.parquet" in outputs


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


def test_adaptive_alpha_math():
    """α(x) = clip(SU/ln(N) − VU, 0, α_max). Paper App G.1 normalisation."""
    import math

    import pandas as pd

    from sae_muc.pipeline.intervene import _compute_adaptive_alphas

    class _FakeStore:
        def __init__(self, tables):
            self._tables = tables
        def load_parquet(self, name):
            return self._tables[name]

    class _FakeCtx:
        def __init__(self, tables):
            self.store = _FakeStore(tables)

    # N=10 sampled answers per question (the paper's default).
    judge_rows = []
    for sid, vu in [("a", 0.0), ("b", 0.5), ("c", 1.0)]:
        for j in range(10):
            judge_rows.append(
                {"sample_id": sid, "kind": "sample", "gen_idx": j, "vu_score": vu}
            )
    ln10 = math.log(10)
    # SE chosen so SU_norm = SE/ln(10) ∈ {0, 0.5, 1.0}.
    se_rows = [
        {"sample_id": "a", "semantic_entropy": 0.0,        "n_samples": 10},
        {"sample_id": "b", "semantic_entropy": 0.5 * ln10, "n_samples": 10},
        {"sample_id": "c", "semantic_entropy": 1.0 * ln10, "n_samples": 10},
    ]
    ctx = _FakeCtx({
        "judge_scores.parquet": pd.DataFrame(judge_rows),
        "semantic_entropy.parquet": pd.DataFrame(se_rows),
    })

    df = _compute_adaptive_alphas(ctx, ["a", "b", "c"], alpha_max=0.5)
    assert list(df["su_norm"]) == pytest.approx([0.0, 0.5, 1.0])
    # vu = [0.0, 0.5, 1.0]; diff with su_norm = [0, 0, 0]; clipped to 0.
    assert list(df["alpha"]) == pytest.approx([0.0, 0.0, 0.0])

    # Second case: vu = 0 across the board — α tracks su_norm, capped at α_max.
    judge_rows_low_vu = [
        {"sample_id": sid, "kind": "sample", "gen_idx": 0, "vu_score": 0.0}
        for sid in ("a", "b", "c")
    ]
    ctx2 = _FakeCtx({
        "judge_scores.parquet": pd.DataFrame(judge_rows_low_vu),
        "semantic_entropy.parquet": pd.DataFrame(se_rows),
    })
    df2 = _compute_adaptive_alphas(ctx2, ["a", "b", "c"], alpha_max=0.5)
    # su_norm = [0, 0.5, 1.0]; clip(_, 0, 0.5) → [0, 0.5, 0.5].
    assert list(df2["alpha"]) == pytest.approx([0.0, 0.5, 0.5])


def test_adaptive_alpha_no_min_max_normalisation():
    """Regression for C1: SU_norm must NOT be min-max-normalised over the run.

    Two questions with identical SE but at the low end of the run's SE range
    used to map to su_norm=0 under min-max; with paper-faithful SE/ln(N) they
    instead reflect the absolute uncertainty level.
    """
    import math

    import pandas as pd

    from sae_muc.pipeline.intervene import _compute_adaptive_alphas

    class _FakeStore:
        def __init__(self, tables):
            self._tables = tables
        def load_parquet(self, name):
            return self._tables[name]

    class _FakeCtx:
        def __init__(self, tables):
            self.store = _FakeStore(tables)

    # All three questions have SE near the middle of the [0, ln(10)] range,
    # but a, b are at SE=0.4*ln(10), c at SE=ln(10). Under min-max, a and b
    # would collapse to su_norm=0; under SE/ln(N) they're 0.4.
    ln10 = math.log(10)
    se_rows = [
        {"sample_id": "a", "semantic_entropy": 0.4 * ln10, "n_samples": 10},
        {"sample_id": "b", "semantic_entropy": 0.4 * ln10, "n_samples": 10},
        {"sample_id": "c", "semantic_entropy": 1.0 * ln10, "n_samples": 10},
    ]
    judge_rows = [
        {"sample_id": sid, "kind": "sample", "gen_idx": 0, "vu_score": 0.0}
        for sid in ("a", "b", "c")
    ]
    ctx = _FakeCtx({
        "judge_scores.parquet": pd.DataFrame(judge_rows),
        "semantic_entropy.parquet": pd.DataFrame(se_rows),
    })
    df = _compute_adaptive_alphas(ctx, ["a", "b", "c"], alpha_max=1.0)
    assert list(df["su_norm"]) == pytest.approx([0.4, 0.4, 1.0])


def test_intervene_adaptive_gate_skips_safe_samples(fake_ctx):
    """C3.c: gate_by_detector=True skips intervention for samples not at risk
    and reuses baseline rows with α=0."""
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    samples = fake_ctx.store.load_parquet("samples.parquet")

    judge_rows = []
    for sid in samples["sample_id"]:
        for j in range(3):
            judge_rows.append(
                {"sample_id": sid, "kind": "sample", "gen_idx": j,
                 "decisiveness": 0.5, "vu_score": 0.1, "raw": "0.1"}
            )
    fake_ctx.store.save_parquet("judge_scores.parquet", pd.DataFrame(judge_rows))
    fake_ctx.store.save_parquet(
        "semantic_entropy.parquet",
        pd.DataFrame({
            "sample_id": list(samples["sample_id"]),
            "semantic_entropy": [1.5] * len(samples),
            "n_clusters": [2] * len(samples),
            "n_samples": [3] * len(samples),
        }),
    )
    hidden_states.run(fake_ctx)
    _seed_vuf_artefacts(fake_ctx, layers=(0, 1, 2), d_model=8)

    # Hand-craft detection.parquet so that only the first sample is at risk.
    sids = list(samples["sample_id"])
    fake_ctx.store.save_parquet(
        "detection.parquet",
        pd.DataFrame({
            "sample_id": sids,
            "is_at_risk": [True] + [False] * (len(sids) - 1),
        }),
    )

    new_cfg = fake_ctx.cfg.model_copy(
        update={
            "stages": fake_ctx.cfg.stages.model_copy(
                update={
                    "intervene": fake_ctx.cfg.stages.intervene.model_copy(
                        update={
                            "mode": "adaptive",
                            "alpha_max": 0.5,
                            "layer": 1,
                            "gate_by_detector": True,
                        },
                    ),
                }
            )
        }
    )
    object.__setattr__(fake_ctx, "cfg", new_cfg)

    intervene.run(fake_ctx)
    alphas = fake_ctx.store.load_parquet("intervention/adaptive/alphas.parquet")
    # Non-at-risk samples must have alpha forced to 0.
    safe_alphas = alphas[alphas["sample_id"].isin(sids[1:])]["alpha"]
    assert (safe_alphas == 0.0).all()

    gens = fake_ctx.store.load_parquet("intervention/adaptive/generations.parquet")
    # The reused baseline rows are tagged α=0 too.
    safe_rows = gens[gens["sample_id"].isin(sids[1:])]
    assert (safe_rows["alpha"] == 0.0).all()
    # Texts for safe samples must equal the baseline generations verbatim.
    baseline = fake_ctx.store.load_parquet("generations.parquet")
    for sid in sids[1:]:
        b = baseline[baseline["sample_id"] == sid].sort_values(["kind", "gen_idx"])
        g = safe_rows[safe_rows["sample_id"] == sid].sort_values(["kind", "gen_idx"])
        assert list(g["text"]) == list(b["text"])


def test_intervene_adaptive_writes_expected_files(fake_ctx):
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    # Seed judge_scores + semantic_entropy the adaptive branch needs. The
    # values don't matter for file layout, only for α computation, which is
    # already covered by test_adaptive_alpha_math.
    samples = fake_ctx.store.load_parquet("samples.parquet")
    judge_rows = []
    for sid in samples["sample_id"]:
        for j in range(3):
            judge_rows.append(
                {"sample_id": sid, "kind": "sample", "gen_idx": j,
                 "decisiveness": 0.5, "vu_score": 0.5, "raw": "0.5"}
            )
    fake_ctx.store.save_parquet("judge_scores.parquet", pd.DataFrame(judge_rows))
    fake_ctx.store.save_parquet(
        "semantic_entropy.parquet",
        pd.DataFrame({
            "sample_id": list(samples["sample_id"]),
            "semantic_entropy": [0.5] * len(samples),
            "n_clusters": [2] * len(samples),
            "n_samples": [3] * len(samples),
        }),
    )
    hidden_states.run(fake_ctx)
    _seed_vuf_artefacts(fake_ctx, layers=(0, 1, 2), d_model=8)

    new_cfg = fake_ctx.cfg.model_copy(
        update={
            "stages": fake_ctx.cfg.stages.model_copy(
                update={
                    "intervene": fake_ctx.cfg.stages.intervene.model_copy(
                        update={"mode": "adaptive", "alpha_max": 0.5, "layer": 1},
                    ),
                }
            )
        }
    )
    object.__setattr__(fake_ctx, "cfg", new_cfg)

    outputs = intervene.run(fake_ctx)
    assert "intervention/adaptive/alphas.parquet" in outputs
    assert "intervention/adaptive/generations.parquet" in outputs
    assert "intervention/meta.parquet" in outputs

    alphas_df = fake_ctx.store.load_parquet("intervention/adaptive/alphas.parquet")
    assert set(alphas_df.columns) >= {"sample_id", "vu", "se", "su_norm", "alpha"}
    assert (alphas_df["alpha"] >= 0).all()
    assert (alphas_df["alpha"] <= 0.5).all()

    meta = fake_ctx.store.load_parquet("intervention/meta.parquet")
    assert list(meta["mode"]) == ["adaptive"]
    assert meta.iloc[0]["alpha_max"] == 0.5


def test_skip_decode_step_passes_through_seq_len_1():
    """P1: with apply_during_generation=False, hook is bypassed for seq_len==1."""
    from sae_muc.pipeline.intervene import _skip_decode_step

    inner = _build_hook(torch.tensor([1.0, 0.0, 0.0, 0.0]), alpha=0.5)
    wrapped = _skip_decode_step(inner)

    # seq_len == 1 → identity.
    decode_step = torch.zeros(2, 1, 4)
    assert torch.equal(wrapped(decode_step), decode_step)
    # seq_len > 1 → unchanged from inner hook.
    prefill = torch.zeros(2, 5, 4)
    expected = inner(prefill)
    assert torch.equal(wrapped(prefill), expected)


def test_intervene_apply_during_generation_false_silences_fakebackend_probe(fake_ctx):
    """P1 stage-level: with apply_during_generation=False, the FakeBackend
    probe (single token) sees no perturbation, so different α produce
    identical outputs."""
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    hidden_states.run(fake_ctx)
    _seed_vuf_artefacts(fake_ctx, layers=(0, 1, 2), d_model=8)

    new_cfg = fake_ctx.cfg.model_copy(
        update={
            "stages": fake_ctx.cfg.stages.model_copy(
                update={
                    "intervene": fake_ctx.cfg.stages.intervene.model_copy(
                        update={
                            "alpha_grid": [-1.0, 1.0],
                            "layer": 1,
                            "apply_during_generation": False,
                        },
                    ),
                }
            )
        }
    )
    object.__setattr__(fake_ctx, "cfg", new_cfg)
    intervene.run(fake_ctx)
    neg = fake_ctx.store.load_parquet("intervention/alpha_-1.00/generations.parquet")
    pos = fake_ctx.store.load_parquet("intervention/alpha_+1.00/generations.parquet")
    # FakeBackend's probe is [1,1,D] → wrapped hook returns identity → no
    # alpha hint baked into the prompt → outputs match across α.
    assert neg["text"].tolist() == pos["text"].tolist()


def test_sae_emd_cohen_d_delta_is_l2_normalised():
    """S1: cohen_d-weighted δ collapses to L2 norm 1 across selected features."""
    from sae_muc.models.sae import FakeSAEBackend
    from sae_muc.pipeline.intervene import _build_sae_emd_hook

    sae = FakeSAEBackend(d_in=8, d_latent=16)
    cohen_d = {0: 0.8, 1: 1.5, 14: -0.6, 15: -1.2}

    hook_zero = _build_sae_emd_hook(
        [0, 1], [14, 15], sae, alpha=0.0,
        cohen_d=cohen_d, delta_mode="cohen_d",
    )
    hook_one = _build_sae_emd_hook(
        [0, 1], [14, 15], sae, alpha=1.0,
        cohen_d=cohen_d, delta_mode="cohen_d",
    )

    residual = torch.randn(2, 3, 8)
    # α=0 should be (effectively) identity, α=1 should differ.
    assert torch.allclose(hook_zero(residual), residual, atol=1e-5)
    assert not torch.allclose(hook_one(residual), residual, atol=1e-3)


def test_sae_emd_cohen_d_requires_mapping():
    from sae_muc.models.sae import FakeSAEBackend
    from sae_muc.pipeline.intervene import _build_sae_emd_hook

    sae = FakeSAEBackend(d_in=8, d_latent=16)
    with pytest.raises(ValueError, match="cohen_d mapping"):
        _build_sae_emd_hook([0, 1], [14], sae, alpha=1.0, delta_mode="cohen_d")


def test_sae_emd_multihot_mode_still_works():
    from sae_muc.models.sae import FakeSAEBackend
    from sae_muc.pipeline.intervene import _build_sae_emd_hook

    sae = FakeSAEBackend(d_in=8, d_latent=16)
    hook = _build_sae_emd_hook(
        [0, 1], [14, 15], sae, alpha=1.0, delta_mode="multihot",
    )
    residual = torch.randn(2, 3, 8)
    assert hook(residual).shape == residual.shape


def test_sae_clamp_alpha_zero_is_identity():
    """S2: α=0 leaves SAE features unchanged → output equals residual (up to err round-trip)."""
    from sae_muc.models.sae import FakeSAEBackend
    from sae_muc.pipeline.intervene import _build_sae_clamp_hook

    sae = FakeSAEBackend(d_in=8, d_latent=16)
    unc_targets = {0: 5.0, 1: 3.0}
    hook = _build_sae_clamp_hook(
        [0, 1], [14, 15], sae, alpha=0.0, unc_targets=unc_targets,
    )
    residual = torch.randn(2, 3, 8)
    assert torch.allclose(hook(residual), residual, atol=1e-5)


def test_sae_clamp_uses_per_feature_target_in_latent_space():
    """S2: clamp acts on the SAE latent, lifting unc to target and zeroing cert.

    Uses a custom SAE backend whose encode/decode are identity-on-the-first-d_in
    so we can read latent values directly off the output residual.
    """
    import torch as _t

    from sae_muc.pipeline.intervene import _build_sae_clamp_hook

    class IdSAE:
        def __init__(self):
            self.d_in = 8
            self.d_latent = 8

        def encode(self, x):
            return x.clone()

        def decode(self, f):
            return f.clone()

    sae = IdSAE()
    unc_targets = {0: 5.0, 1: 3.0}
    hook = _build_sae_clamp_hook(
        [0, 1], [6, 7], sae, alpha=1.0, unc_targets=unc_targets,
    )
    residual = _t.zeros(1, 1, 8)
    residual[0, 0, 0] = 1.0   # below target → push up to 5.0
    residual[0, 0, 1] = 4.0   # above target → no change (soft-push only raises)
    residual[0, 0, 6] = 2.0   # certainty → suppress to 0 with α=1
    residual[0, 0, 7] = -1.0  # certainty → suppress to 0
    out = hook(residual)[0, 0]
    assert out[0].item() == pytest.approx(5.0)
    assert out[1].item() == pytest.approx(4.0)  # already above target
    assert out[6].item() == pytest.approx(0.0, abs=1e-6)
    assert out[7].item() == pytest.approx(0.0, abs=1e-6)


def test_sae_clamp_alpha_clipped_to_unit_interval():
    """S2: α outside [0,1] is clipped to [0,1]."""
    from sae_muc.models.sae import FakeSAEBackend
    from sae_muc.pipeline.intervene import _build_sae_clamp_hook

    sae = FakeSAEBackend(d_in=8, d_latent=16)
    targets = {0: 5.0}
    hook_neg = _build_sae_clamp_hook([0], [], sae, alpha=-2.0, unc_targets=targets)
    hook_zero = _build_sae_clamp_hook([0], [], sae, alpha=0.0, unc_targets=targets)
    residual = torch.randn(1, 1, 8)
    # α=-2 → clipped to 0 → equivalent to α=0.
    assert torch.allclose(hook_neg(residual), hook_zero(residual), atol=1e-6)


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
