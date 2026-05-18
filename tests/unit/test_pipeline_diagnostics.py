from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from sae_muc.artifacts import StageManifest
from sae_muc.pipeline import (
    diagnostics,
    diagnostics_datasets as dd,
    generate,
    hidden_states,
    intervene,
    prepare,
    sae_features,
    vuf,
)


# ---------- helpers ------------------------------------------------------------


def _seed_vuf_artefacts(fake_ctx, *, layers=(0, 1, 2), d_model: int = 8):
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
        d = torch.zeros(d_model)
        d[layer % d_model] = 1.0
        fake_ctx.store.save_safetensors(
            f"vuf/direction_layer_{layer}.safetensors", {"direction": d.contiguous()}
        )


def _seed_sae_features(fake_ctx, *, d_latent: int = 16, k_top: int = 3, layer: int = 1):
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
                "layer": layer,
                "cohen_d": 0.5 if sel == "uncertainty" else (-0.5 if sel == "certainty" else 0.0),
                "mean_uncertain": 0.0,
                "mean_certain": 0.0,
                "selected_as": sel,
            }
        )
    fake_ctx.store.save_parquet("sae_features/stats.parquet", pd.DataFrame(rows))


def _seed_intervene(fake_ctx, *, alpha_grid, layer=1, method="linear_vuf"):
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    hidden_states.run(fake_ctx)
    _seed_vuf_artefacts(fake_ctx)
    if method in ("sae_emd", "sae_clamp"):
        _seed_sae_features(fake_ctx, layer=layer)

    new_cfg = fake_ctx.cfg.model_copy(
        update={
            "stages": fake_ctx.cfg.stages.model_copy(
                update={
                    "intervene": fake_ctx.cfg.stages.intervene.model_copy(
                        update={
                            "method": method,
                            "alpha_grid": list(alpha_grid),
                            "layer": layer,
                        },
                    ),
                }
            )
        }
    )
    object.__setattr__(fake_ctx, "cfg", new_cfg)
    intervene.run(fake_ctx)


def _force_diagnostics(fake_ctx, **overrides):
    """Apply arbitrary overrides on top of the diagnostics sub-config."""
    new_diag = fake_ctx.cfg.stages.diagnostics.model_copy(update=overrides)
    new_cfg = fake_ctx.cfg.model_copy(
        update={
            "stages": fake_ctx.cfg.stages.model_copy(update={"diagnostics": new_diag}),
        }
    )
    object.__setattr__(fake_ctx, "cfg", new_cfg)


@pytest.fixture
def mock_benchmarks(monkeypatch):
    """Patch dataset loaders so tests don't hit HF Hub.

    Returns a dict of fixture rows so tests can introspect what was scored.
    """
    mmlu_items = [
        {"question": "2 + 2 = ?", "choices": ["3", "4", "5", "6"], "answer": 1},
        {"question": "Capital of France?", "choices": ["Paris", "London", "Berlin", "Rome"], "answer": 0},
    ]
    hella_items = [
        {"ctx": "A woman is outside with a broom.", "endings": ["She sweeps the porch.", "She rides it.", "She eats it.", "She cuts hair."], "answer": 0},
        {"ctx": "A man begins to cook on a stove.", "endings": ["He puts on a hat.", "He cracks eggs.", "He sleeps.", "He drives."], "answer": 1},
    ]
    gsm_items = [
        {"question": "Tom has 5 apples and gets 3 more. How many?", "gold_text": "Tom has 5+3=8 apples.\n#### 8", "gold_number": 8.0},
        {"question": "What is 10 minus 4?", "gold_text": "10-4=6.\n#### 6", "gold_number": 6.0},
    ]
    monkeypatch.setattr(dd, "load_mmlu", lambda n: mmlu_items[:n])
    monkeypatch.setattr(dd, "load_hellaswag", lambda n: hella_items[:n])
    monkeypatch.setattr(dd, "load_gsm8k", lambda n: gsm_items[:n])
    return {"mmlu": mmlu_items, "hellaswag": hella_items, "gsm8k": gsm_items}


# ---------- pure helper tests --------------------------------------------------


def test_parse_layers_field_single():
    assert diagnostics._parse_layers_field("15") == [15]


def test_parse_layers_field_range():
    assert diagnostics._parse_layers_field("15-17") == [15, 16, 17]


def test_parse_layers_field_csv():
    assert diagnostics._parse_layers_field("0,2,5") == [0, 2, 5]


def test_variant_name_extraction():
    assert (
        diagnostics._variant_name("intervention/alpha_+0.50/generations.parquet")
        == "alpha_+0.50"
    )
    assert (
        diagnostics._variant_name("intervention/adaptive/generations.parquet")
        == "adaptive"
    )


def test_extract_first_number_handles_common_shapes():
    assert dd._extract_first_number("the answer is 42") == 42.0
    assert dd._extract_first_number("#### 8") == 8.0
    assert dd._extract_first_number("3.14 pi") == pytest.approx(3.14)
    assert dd._extract_first_number("no number here") is None
    assert dd._extract_first_number("") is None
    assert dd._extract_first_number(None) is None


# ---------- dataset scorers ----------------------------------------------------


def test_score_mmlu_returns_finite_metrics(fake_ctx):
    items = [
        {"question": "2 + 2 = ?", "choices": ["3", "4", "5", "6"], "answer": 1},
        {"question": "Capital of France?", "choices": ["Paris", "London", "Berlin", "Rome"], "answer": 0},
    ]
    out = dd.score_mmlu(fake_ctx.llm, items)
    assert 0.0 <= out["accuracy"] <= 1.0
    assert math.isfinite(out["mean_nll"])
    assert out["n"] == 2


def test_score_hellaswag_returns_finite_metrics(fake_ctx):
    items = [
        {"ctx": "A woman is outside with a broom.", "endings": ["She sweeps.", "She rides it.", "She eats it.", "She cuts hair."], "answer": 0},
    ]
    out = dd.score_hellaswag(fake_ctx.llm, items)
    assert 0.0 <= out["accuracy"] <= 1.0
    assert math.isfinite(out["mean_nll"])


def test_score_gsm8k_brief_generation_then_parse(fake_ctx):
    items = [
        {"question": "What is 2 + 2?", "gold_text": "It's 4.\n#### 4", "gold_number": 4.0},
        {"question": "What is 3 + 3?", "gold_text": "It's 6.\n#### 6", "gold_number": 6.0},
    ]
    out = dd.score_gsm8k(fake_ctx.llm, items, max_new_tokens=8)
    assert 0.0 <= out["accuracy"] <= 1.0
    assert out["n"] == 2


def test_score_mmlu_empty_items_returns_nan():
    out = dd.score_mmlu(None, [])
    assert math.isnan(out["accuracy"])
    assert out["n"] == 0


# ---------- stage-level: per-variant cross-run mode ----------------------------


def test_diagnostics_writes_ppl_kl_bench_artefacts(fake_ctx, mock_benchmarks):
    _seed_intervene(fake_ctx, alpha_grid=[0.0])
    _force_diagnostics(fake_ctx, corpus_n_chars=0, n_mmlu=2, n_hellaswag=1, n_gsm8k=2)
    outputs = diagnostics.run(fake_ctx)
    assert "diagnostics/perplexity.parquet" in outputs
    assert "diagnostics/benchmarks.parquet" in outputs
    assert "diagnostics/kl.parquet" in outputs
    assert "diagnostics/summary.json" in outputs


def test_diagnostics_baseline_row_has_ratio_one(fake_ctx, mock_benchmarks):
    _seed_intervene(fake_ctx, alpha_grid=[0.0])
    _force_diagnostics(fake_ctx, corpus_n_chars=0, n_mmlu=2, n_hellaswag=1, n_gsm8k=2)
    diagnostics.run(fake_ctx)
    ppl = fake_ctx.store.load_parquet("diagnostics/perplexity.parquet")
    base = ppl[ppl["variant"] == "baseline"].iloc[0]
    assert base["ppl_ratio_vs_baseline"] == pytest.approx(1.0, abs=0)


def test_diagnostics_linear_vuf_alpha_zero_is_identity(fake_ctx, mock_benchmarks):
    _seed_intervene(fake_ctx, alpha_grid=[0.0], method="linear_vuf")
    _force_diagnostics(fake_ctx, corpus_n_chars=0, corpora=["wikitext"])  # cheaper, no MC
    diagnostics.run(fake_ctx)
    ppl = fake_ctx.store.load_parquet("diagnostics/perplexity.parquet")
    variant = ppl[ppl["variant"] == "alpha_+0.00"].iloc[0]
    assert variant["ppl_ratio_vs_baseline"] == pytest.approx(1.0, abs=0)
    kl = fake_ctx.store.load_parquet("diagnostics/kl.parquet")
    variant_kl = kl[kl["variant"] == "alpha_+0.00"].iloc[0]
    assert variant_kl["mean_kl_all_positions"] == pytest.approx(0.0, abs=0)


def test_diagnostics_sae_emd_alpha_zero_ratio_close_to_one(fake_ctx, mock_benchmarks):
    _seed_intervene(fake_ctx, alpha_grid=[0.0], method="sae_emd")
    _force_diagnostics(fake_ctx, corpus_n_chars=0, corpora=["wikitext"])
    diagnostics.run(fake_ctx)
    ppl = fake_ctx.store.load_parquet("diagnostics/perplexity.parquet")
    variant = ppl[ppl["variant"] == "alpha_+0.00"].iloc[0]
    assert variant["ppl_ratio_vs_baseline"] == pytest.approx(1.0, abs=1e-3)


def test_diagnostics_strong_alpha_moves_ppl_and_kl(fake_ctx, mock_benchmarks):
    _seed_intervene(fake_ctx, alpha_grid=[2.0], method="linear_vuf")
    _force_diagnostics(fake_ctx, corpus_n_chars=0, corpora=["wikitext"])
    diagnostics.run(fake_ctx)
    ppl = fake_ctx.store.load_parquet("diagnostics/perplexity.parquet")
    variant = ppl[ppl["variant"] == "alpha_+2.00"].iloc[0]
    assert variant["ppl_ratio_vs_baseline"] > 1.0
    kl = fake_ctx.store.load_parquet("diagnostics/kl.parquet")
    variant_kl = kl[kl["variant"] == "alpha_+2.00"].iloc[0]
    assert variant_kl["mean_kl_all_positions"] > 0.0


def test_diagnostics_kl_is_non_negative_everywhere(fake_ctx, mock_benchmarks):
    _seed_intervene(fake_ctx, alpha_grid=[-1.0, 0.0, 1.0])
    _force_diagnostics(fake_ctx, corpus_n_chars=0, corpora=["wikitext"])
    diagnostics.run(fake_ctx)
    kl = fake_ctx.store.load_parquet("diagnostics/kl.parquet")
    assert (kl["mean_kl_all_positions"] >= 0).all()
    assert (kl["top1_disagreement_rate"] >= 0).all()
    assert (kl["top1_disagreement_rate"] <= 1).all()


def test_diagnostics_summary_lists_every_variant(fake_ctx, mock_benchmarks):
    _seed_intervene(fake_ctx, alpha_grid=[-0.5, 0.5])
    _force_diagnostics(fake_ctx, corpus_n_chars=0, corpora=["wikitext"])
    diagnostics.run(fake_ctx)
    summary = fake_ctx.store.load_json("diagnostics/summary.json")
    variant_names = {v["variant"] for v in summary["variants"]}
    assert variant_names == {"alpha_-0.50", "alpha_+0.50"}
    assert summary["skipped"] is False


def test_diagnostics_disabled_short_circuits_to_stub(fake_ctx):
    _seed_intervene(fake_ctx, alpha_grid=[0.0])
    _force_diagnostics(fake_ctx, enabled=False)
    outputs = diagnostics.run(fake_ctx)
    assert outputs == ["diagnostics/summary.json"]
    summary = fake_ctx.store.load_json("diagnostics/summary.json")
    assert summary["skipped"] is True
    assert "disabled" in summary["reason"]
    assert not fake_ctx.store.exists("diagnostics/perplexity.parquet")


def test_diagnostics_missing_intervene_skips(fake_ctx):
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    outputs = diagnostics.run(fake_ctx)
    assert outputs == ["diagnostics/summary.json"]
    summary = fake_ctx.store.load_json("diagnostics/summary.json")
    assert summary["skipped"] is True


def test_diagnostics_backend_without_logits_api_skips(fake_ctx):
    _seed_intervene(fake_ctx, alpha_grid=[0.0])

    class _RemoteLikeBackend:
        name = "fake-remote"

    object.__setattr__(fake_ctx, "llm", _RemoteLikeBackend())
    outputs = diagnostics.run(fake_ctx)
    assert outputs == ["diagnostics/summary.json"]
    summary = fake_ctx.store.load_json("diagnostics/summary.json")
    assert summary["skipped"] is True


def test_diagnostics_manifest_skip_on_resume(fake_ctx, mock_benchmarks):
    from sae_muc.pipeline.runner import run_stage

    _seed_intervene(fake_ctx, alpha_grid=[0.0])
    _force_diagnostics(fake_ctx, corpus_n_chars=0, corpora=["wikitext"])
    assert run_stage(fake_ctx, "diagnostics") is True
    manifest = StageManifest(fake_ctx.store.run_dir, "diagnostics")
    assert manifest.should_skip() is True
    assert run_stage(fake_ctx, "diagnostics") is False


def test_diagnostics_handles_adaptive_variant(fake_ctx, mock_benchmarks):
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    samples = fake_ctx.store.load_parquet("samples.parquet")
    judge_rows = [
        {"sample_id": sid, "kind": "sample", "gen_idx": j,
         "decisiveness": 0.5, "vu_score": 0.3, "raw": "0.3"}
        for sid in samples["sample_id"] for j in range(3)
    ]
    fake_ctx.store.save_parquet("judge_scores.parquet", pd.DataFrame(judge_rows))
    fake_ctx.store.save_parquet(
        "semantic_entropy.parquet",
        pd.DataFrame({
            "sample_id": list(samples["sample_id"]),
            "semantic_entropy": [0.6] * len(samples),
            "n_clusters": [2] * len(samples),
            "n_samples": [3] * len(samples),
        }),
    )
    hidden_states.run(fake_ctx)
    _seed_vuf_artefacts(fake_ctx)
    new_cfg = fake_ctx.cfg.model_copy(
        update={
            "stages": fake_ctx.cfg.stages.model_copy(
                update={
                    "intervene": fake_ctx.cfg.stages.intervene.model_copy(
                        update={"mode": "adaptive", "alpha_max": 0.5, "layer": 1},
                    ),
                    "diagnostics": fake_ctx.cfg.stages.diagnostics.model_copy(
                        update={"corpus_n_chars": 0, "corpora": ["wikitext"]},
                    ),
                }
            )
        }
    )
    object.__setattr__(fake_ctx, "cfg", new_cfg)
    intervene.run(fake_ctx)
    diagnostics.run(fake_ctx)
    ppl = fake_ctx.store.load_parquet("diagnostics/perplexity.parquet")
    assert "adaptive" in set(ppl["variant"])
    adaptive_row = ppl[ppl["variant"] == "adaptive"].iloc[0]
    assert np.isfinite(adaptive_row["alpha"])
    assert adaptive_row["alpha"] >= 0.0


# ---------- stage-level: multi-method × alpha sweep ----------------------------


def test_diagnostics_multi_method_sweep_writes_matrix(fake_ctx, mock_benchmarks):
    _seed_intervene(fake_ctx, alpha_grid=[0.0])
    _seed_sae_features(fake_ctx)  # needed because compare_methods includes sae_emd
    _force_diagnostics(
        fake_ctx,
        corpus_n_chars=0,
        corpora=["wikitext", "mmlu"],
        n_mmlu=2,
        compare_methods=["linear_vuf", "sae_emd"],
        alpha_sweep=[-1.0, 0.0, 1.0],
    )
    outputs = diagnostics.run(fake_ctx)
    assert "diagnostics/method_alpha_sweep.parquet" in outputs

    sweep = fake_ctx.store.load_parquet("diagnostics/method_alpha_sweep.parquet")
    # Methods × alphas × datasets: 2 × 3 × 2 = 12 + 2 baseline rows (1 per ds)
    actual_methods = set(sweep["method"])
    assert {"linear_vuf", "sae_emd", "baseline"} == actual_methods
    actual_datasets = set(sweep["dataset"])
    assert actual_datasets == {"wikitext", "mmlu"}


def test_diagnostics_multi_method_skipped_when_compare_methods_empty(fake_ctx, mock_benchmarks):
    _seed_intervene(fake_ctx, alpha_grid=[0.0])
    _force_diagnostics(fake_ctx, corpus_n_chars=0, corpora=["wikitext"], compare_methods=[])
    outputs = diagnostics.run(fake_ctx)
    assert "diagnostics/method_alpha_sweep.parquet" not in outputs
    assert not fake_ctx.store.exists("diagnostics/method_alpha_sweep.parquet")


def test_diagnostics_multi_method_alpha_zero_matches_baseline_for_linear_vuf(
    fake_ctx, mock_benchmarks,
):
    """linear_vuf at α=0 is a bit-exact no-op → its wiki_nll in the sweep
    matches the baseline row exactly."""
    _seed_intervene(fake_ctx, alpha_grid=[0.0])
    _force_diagnostics(
        fake_ctx,
        corpus_n_chars=0,
        corpora=["wikitext"],
        compare_methods=["linear_vuf"],
        alpha_sweep=[0.0],
    )
    diagnostics.run(fake_ctx)
    sweep = fake_ctx.store.load_parquet("diagnostics/method_alpha_sweep.parquet")
    base_wiki = sweep[(sweep["method"] == "baseline") & (sweep["dataset"] == "wikitext")].iloc[0]
    lv0 = sweep[
        (sweep["method"] == "linear_vuf")
        & (sweep["alpha"] == 0.0)
        & (sweep["dataset"] == "wikitext")
    ].iloc[0]
    assert lv0["mean_nll"] == pytest.approx(base_wiki["mean_nll"], abs=0)


def test_sae_features_runs_when_compare_methods_demands_sae(fake_ctx, fake_hf_rows):
    """Regression for the sae_features gating: even if the main run is
    linear_vuf, sae_features must execute when diagnostics will sweep
    sae_emd / sae_clamp."""
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    # Seed minimal judge + se so vuf.run can build the split.
    samples = fake_ctx.store.load_parquet("samples.parquet")
    judge_rows = []
    for i, sid in enumerate(samples["sample_id"]):
        vu_per_sample = 0.95 if i < 2 else 0.0  # 2 uncertain + rest certain
        for j in range(3):
            judge_rows.append({
                "sample_id": sid, "kind": "sample", "gen_idx": j,
                "decisiveness": 1 - vu_per_sample, "vu_score": vu_per_sample, "raw": str(vu_per_sample),
            })
    fake_ctx.store.save_parquet("judge_scores.parquet", pd.DataFrame(judge_rows))
    hidden_states.run(fake_ctx)
    vuf.run(fake_ctx)
    new_cfg = fake_ctx.cfg.model_copy(
        update={
            "stages": fake_ctx.cfg.stages.model_copy(
                update={
                    "intervene": fake_ctx.cfg.stages.intervene.model_copy(
                        update={"method": "linear_vuf", "layer": 1},
                    ),
                    "diagnostics": fake_ctx.cfg.stages.diagnostics.model_copy(
                        update={"compare_methods": ["sae_emd"], "alpha_sweep": [0.0]},
                    ),
                }
            )
        }
    )
    object.__setattr__(fake_ctx, "cfg", new_cfg)
    outputs = sae_features.run(fake_ctx)
    # Without the un-gating fix this would early-return with [] because
    # the primary method is linear_vuf.
    assert outputs == ["sae_features/stats.parquet"]
    assert fake_ctx.store.exists("sae_features/stats.parquet")
