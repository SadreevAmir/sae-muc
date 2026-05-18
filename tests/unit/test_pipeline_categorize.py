from __future__ import annotations

import pandas as pd
import torch

from sae_muc.models.base import Generation
from sae_muc.pipeline import (
    categorize_uncertainty as cu,
    diagnostics,
    generate,
    hidden_states,
    prepare,
    vuf,
)


# ---------- pure helper tests ---------------------------------------------------


def test_parse_category_handles_clean_output():
    assert cu.parse_category("ABSTAIN") == "ABSTAIN"
    assert cu.parse_category("HEDGE") == "HEDGE"
    assert cu.parse_category("CONFIDENT") == "CONFIDENT"


def test_parse_category_handles_prefixes_and_case():
    assert cu.parse_category("The category is ABSTAIN.") == "ABSTAIN"
    assert cu.parse_category(" hedge\n") == "HEDGE"
    assert cu.parse_category("category: confident") == "CONFIDENT"
    assert cu.parse_category("Answer: Probably HEDGE based on tone.") == "HEDGE"


def test_parse_category_returns_none_for_unparseable():
    assert cu.parse_category("hmm") is None
    assert cu.parse_category("") is None
    assert cu.parse_category("I don't know how to categorize this.") is None


def test_aggregate_majority_clear_winner():
    labels = ["ABSTAIN"] * 7 + ["HEDGE"] * 3
    agg, counts = cu.aggregate_votes(labels)
    assert agg == "ABSTAIN"
    assert counts == {"ABSTAIN": 7, "HEDGE": 3, "CONFIDENT": 0, "unparsed": 0}


def test_aggregate_majority_tied_returns_mixed():
    labels = ["ABSTAIN"] * 5 + ["HEDGE"] * 5
    agg, _ = cu.aggregate_votes(labels)
    assert agg == "MIXED"


def test_aggregate_majority_all_confident_returns_mixed():
    """CONFIDENT generations don't count toward ABSTAIN-vs-HEDGE majority;
    a question with all CONFIDENT labels has zero valid votes."""
    labels = ["CONFIDENT"] * 10
    agg, counts = cu.aggregate_votes(labels)
    assert agg == "MIXED"
    assert counts["CONFIDENT"] == 10


def test_aggregate_majority_below_threshold_returns_mixed():
    # 5 ABSTAIN + 4 HEDGE (over 9 valid) = 55% < 60% threshold → MIXED.
    labels = ["ABSTAIN"] * 5 + ["HEDGE"] * 4
    agg, _ = cu.aggregate_votes(labels)
    assert agg == "MIXED"


def test_aggregate_empty_and_none_only():
    assert cu.aggregate_votes([])[0] == "MIXED"
    assert cu.aggregate_votes([None, None, None])[0] == "MIXED"


def test_aggregate_unparsed_counted_separately():
    labels = ["ABSTAIN"] * 6 + ["HEDGE"] * 1 + [None] * 3
    agg, counts = cu.aggregate_votes(labels)
    assert agg == "ABSTAIN"
    assert counts["unparsed"] == 3


# ---------- stage-level tests ---------------------------------------------------


def _seed_judge_scores(fake_ctx, *, uncertain_sids: list[str], certain_sids: list[str]):
    """Create a judge_scores.parquet with controlled VU values per sample_id."""
    rows = []
    for sid in uncertain_sids:
        # Greedy row gets VU=0.95 too so refusal classification could trigger
        # (irrelevant for categorize but matches real schema).
        rows.append({"sample_id": sid, "kind": "greedy", "gen_idx": 0,
                     "decisiveness": 0.05, "vu_score": 0.95, "raw": "0.05"})
        for j in range(3):
            rows.append({"sample_id": sid, "kind": "sample", "gen_idx": j,
                         "decisiveness": 0.05, "vu_score": 0.95, "raw": "0.05"})
    for sid in certain_sids:
        rows.append({"sample_id": sid, "kind": "greedy", "gen_idx": 0,
                     "decisiveness": 0.95, "vu_score": 0.05, "raw": "0.95"})
        for j in range(3):
            rows.append({"sample_id": sid, "kind": "sample", "gen_idx": j,
                         "decisiveness": 0.95, "vu_score": 0.05, "raw": "0.95"})
    fake_ctx.store.save_parquet("judge_scores.parquet", pd.DataFrame(rows))


def _enable_categorize(fake_ctx, *, enabled: bool = True, per_category: bool = False):
    new_cfg = fake_ctx.cfg.model_copy(
        update={
            "stages": fake_ctx.cfg.stages.model_copy(
                update={
                    "categorize": fake_ctx.cfg.stages.categorize.model_copy(
                        update={"enabled": enabled},
                    ),
                    "vuf": fake_ctx.cfg.stages.vuf.model_copy(
                        update={"per_category": per_category},
                    ),
                }
            ),
        }
    )
    object.__setattr__(fake_ctx, "cfg", new_cfg)


def test_categorize_disabled_writes_empty_stub(fake_ctx):
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    _seed_judge_scores(fake_ctx, uncertain_sids=[], certain_sids=[])
    _enable_categorize(fake_ctx, enabled=False)
    outputs = cu.run(fake_ctx)
    assert outputs == ["categories.parquet"]
    df = fake_ctx.store.load_parquet("categories.parquet")
    assert len(df) == 0
    assert set(df.columns) >= {
        "level", "sample_id", "gen_idx", "category", "raw",
        "n_abstain", "n_hedge", "n_confident", "n_unparsed",
    }


def test_categorize_skips_certain_bucket(fake_ctx, monkeypatch):
    """Only uncertain-bucket sample generations get judge calls."""
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    samples = fake_ctx.store.load_parquet("samples.parquet")
    sids = list(samples["sample_id"])
    assert len(sids) >= 4, "fixture should produce ≥4 samples for this test"
    uncertain = sids[:2]
    certain = sids[2:]
    _seed_judge_scores(fake_ctx, uncertain_sids=uncertain, certain_sids=certain)
    _enable_categorize(fake_ctx, enabled=True)

    call_count = {"n": 0}

    def fake_generate(prompts, **_kwargs):
        call_count["n"] += len(prompts)
        return [[Generation(text="ABSTAIN", finish_reason="stop")] for _ in prompts]

    monkeypatch.setattr(fake_ctx.judge, "generate", fake_generate)
    cu.run(fake_ctx)
    # 2 uncertain × 3 samples (high-T gens) = 6 calls. Certain bucket is skipped.
    assert call_count["n"] == 6


def test_categorize_writes_per_gen_and_per_question_rows(fake_ctx, monkeypatch):
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    samples = fake_ctx.store.load_parquet("samples.parquet")
    sids = list(samples["sample_id"])
    uncertain = sids[:2]
    _seed_judge_scores(fake_ctx, uncertain_sids=uncertain, certain_sids=sids[2:])
    _enable_categorize(fake_ctx, enabled=True)

    def fake_generate(prompts, **_kwargs):
        return [[Generation(text="ABSTAIN", finish_reason="stop")] for _ in prompts]

    monkeypatch.setattr(fake_ctx.judge, "generate", fake_generate)
    cu.run(fake_ctx)
    df = fake_ctx.store.load_parquet("categories.parquet")
    gen_rows = df[df["level"] == "gen"]
    q_rows = df[df["level"] == "question"]
    assert len(gen_rows) == 6     # 2 uncertain × 3 samples
    assert len(q_rows) == 2       # one aggregate row per uncertain question
    assert (q_rows["category"] == "ABSTAIN").all()
    assert (q_rows["n_abstain"] == 3).all()


def test_categorize_majority_handles_split_votes(fake_ctx, monkeypatch):
    """7×ABSTAIN + 3×HEDGE across 10 sample gens per question (we use 3 here
    for fixture, but force varied responses): one question gets ABSTAIN
    majority, another gets MIXED."""
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    samples = fake_ctx.store.load_parquet("samples.parquet")
    sids = list(samples["sample_id"])
    uncertain = sids[:2]
    _seed_judge_scores(fake_ctx, uncertain_sids=uncertain, certain_sids=sids[2:])
    _enable_categorize(fake_ctx, enabled=True)

    # Responses keyed by which uncertain question + which gen_idx the prompt
    # corresponds to. We can't easily inspect — instead alternate per call.
    # Pattern: first uncertain Q → all ABSTAIN; second uncertain Q → tied.
    responses = iter([
        "ABSTAIN", "ABSTAIN", "ABSTAIN",           # uncertain[0]
        "ABSTAIN", "HEDGE", "CONFIDENT",           # uncertain[1] — 1 valid each side → MIXED
    ])

    def fake_generate(prompts, **_kwargs):
        return [[Generation(text=next(responses), finish_reason="stop")] for _ in prompts]

    monkeypatch.setattr(fake_ctx.judge, "generate", fake_generate)
    cu.run(fake_ctx)
    df = fake_ctx.store.load_parquet("categories.parquet")
    q_rows = df[df["level"] == "question"].set_index("sample_id")
    assert q_rows.loc[uncertain[0], "category"] == "ABSTAIN"
    assert q_rows.loc[uncertain[1], "category"] == "MIXED"


# ---------- vuf.per_category integration ---------------------------------------


def _seed_full_vuf_inputs(
    fake_ctx, *, abstain_sids: list[str], hedge_sids: list[str], certain_sids: list[str],
):
    """Stage prepare/generate/hidden_states, then write VUF prerequisites
    plus a hand-crafted categories.parquet with the given partition."""
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    hidden_states.run(fake_ctx)
    uncertain = abstain_sids + hedge_sids
    _seed_judge_scores(fake_ctx, uncertain_sids=uncertain, certain_sids=certain_sids)

    # Hand-crafted categories.parquet — q-rows only, all 3 valid votes.
    rows = []
    for sid in abstain_sids:
        rows.append({
            "level": "question", "sample_id": sid, "gen_idx": -1,
            "category": "ABSTAIN", "raw": "",
            "n_abstain": 3, "n_hedge": 0, "n_confident": 0, "n_unparsed": 0,
        })
    for sid in hedge_sids:
        rows.append({
            "level": "question", "sample_id": sid, "gen_idx": -1,
            "category": "HEDGE", "raw": "",
            "n_abstain": 0, "n_hedge": 3, "n_confident": 0, "n_unparsed": 0,
        })
    fake_ctx.store.save_parquet("categories.parquet", pd.DataFrame(rows))


def test_vuf_per_category_off_byte_identical_outputs(fake_ctx):
    """Sanity for back-compat: with per_category=False vuf writes the same
    files as before (just one direction per layer, plus the empty
    category_meta stub)."""
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    hidden_states.run(fake_ctx)
    samples = fake_ctx.store.load_parquet("samples.parquet")
    sids = list(samples["sample_id"])
    _seed_judge_scores(fake_ctx, uncertain_sids=sids[:2], certain_sids=sids[2:])

    outputs = vuf.run(fake_ctx)
    # No per-category files written.
    assert not any("direction_abstain_" in p for p in outputs)
    assert not any("direction_hedge_" in p for p in outputs)
    # category_meta still exists (zero rows).
    assert fake_ctx.store.exists("vuf/category_meta.parquet")
    cat_meta = fake_ctx.store.load_parquet("vuf/category_meta.parquet")
    assert len(cat_meta) == 0
    # Main meta keeps legacy schema — no `variant` column.
    main_meta = fake_ctx.store.load_parquet("vuf/meta.parquet")
    assert "variant" not in main_meta.columns
    # Legacy direction filename pattern preserved.
    n_layers = 4  # FakeBackend._N_LAYERS
    for l in range(n_layers):
        assert fake_ctx.store.exists(f"vuf/direction_layer_{l}.safetensors")


def test_vuf_per_category_writes_three_direction_families(fake_ctx):
    samples = fake_ctx.store.load_parquet  # touch to force fixture lazy-load?
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    hidden_states.run(fake_ctx)
    samples_df = fake_ctx.store.load_parquet("samples.parquet")
    sids = list(samples_df["sample_id"])
    assert len(sids) >= 5

    abstain = [sids[0], sids[1]]
    hedge = [sids[2], sids[3]]
    certain = [sids[4]] if len(sids) == 5 else sids[4:]
    _seed_full_vuf_inputs(
        fake_ctx, abstain_sids=abstain, hedge_sids=hedge, certain_sids=certain,
    )
    _enable_categorize(fake_ctx, enabled=True, per_category=True)

    outputs = vuf.run(fake_ctx)
    n_layers = 4
    for l in range(n_layers):
        assert f"vuf/direction_layer_{l}.safetensors" in outputs
        assert f"vuf/direction_abstain_layer_{l}.safetensors" in outputs
        assert f"vuf/direction_hedge_layer_{l}.safetensors" in outputs

    cat_meta = fake_ctx.store.load_parquet("vuf/category_meta.parquet")
    assert set(cat_meta["variant"]) == {"abstain", "hedge"}
    assert len(cat_meta) == n_layers * 2

    # Each direction is L2-normalised (norm ≈ 1.0).
    for path in (
        f"vuf/direction_abstain_layer_0.safetensors",
        f"vuf/direction_hedge_layer_0.safetensors",
    ):
        d = fake_ctx.store.load_safetensors(path)["direction"]
        assert torch.allclose(d.norm(), torch.tensor(1.0), atol=1e-5)


def test_vuf_per_category_skips_empty_bucket(fake_ctx):
    """If only ABSTAIN has ≥2 members, the hedge direction file is not written."""
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    hidden_states.run(fake_ctx)
    samples_df = fake_ctx.store.load_parquet("samples.parquet")
    sids = list(samples_df["sample_id"])

    # All uncertain are ABSTAIN; no HEDGE category present.
    _seed_full_vuf_inputs(
        fake_ctx, abstain_sids=sids[:2], hedge_sids=[], certain_sids=sids[2:],
    )
    _enable_categorize(fake_ctx, enabled=True, per_category=True)

    outputs = vuf.run(fake_ctx)
    assert any("direction_abstain_" in p for p in outputs)
    assert not any("direction_hedge_" in p for p in outputs)
    cat_meta = fake_ctx.store.load_parquet("vuf/category_meta.parquet")
    assert set(cat_meta["variant"]) == {"abstain"}


def test_vuf_splits_gains_category_column_only_when_per_category_on(fake_ctx):
    prepare.run(fake_ctx)
    generate.run(fake_ctx)
    hidden_states.run(fake_ctx)
    samples_df = fake_ctx.store.load_parquet("samples.parquet")
    sids = list(samples_df["sample_id"])
    _seed_full_vuf_inputs(
        fake_ctx, abstain_sids=sids[:2], hedge_sids=sids[2:4], certain_sids=sids[4:],
    )
    _enable_categorize(fake_ctx, enabled=True, per_category=True)
    vuf.run(fake_ctx)
    splits = fake_ctx.store.load_parquet("vuf/splits.parquet")
    assert "category" in splits.columns
    # Each uncertain sid in the partition has a category recorded.
    by_sid = splits.set_index("sample_id")
    assert by_sid.loc[sids[0], "category"] == "ABSTAIN"
    assert by_sid.loc[sids[2], "category"] == "HEDGE"


# ---------- diagnostics cosine ---------------------------------------


def _seed_category_direction_files(fake_ctx, *, layers=(0, 1, 2), d_model=8):
    """Hand-roll vuf/meta + vuf/category_meta + safetensors so we don't need
    the upstream vuf.run for this isolated test."""
    main_meta = pd.DataFrame({
        "layer": list(layers),
        "path": [f"vuf/direction_layer_{l}.safetensors" for l in layers],
        "raw_norm": [1.0] * len(layers),
        "n_uncertain": [3] * len(layers),
        "n_certain": [3] * len(layers),
        "pooling": ["last_token_q"] * len(layers),
    })
    cat_meta = pd.DataFrame({
        "layer": [l for l in layers for _ in range(2)],
        "variant": ["abstain", "hedge"] * len(layers),
        "path": [
            f"vuf/direction_{v}_layer_{l}.safetensors"
            for l in layers for v in ("abstain", "hedge")
        ],
        "raw_norm": [1.0] * (2 * len(layers)),
        "n_uncertain": [3] * (2 * len(layers)),
        "n_certain": [3] * (2 * len(layers)),
        "pooling": ["last_token_q"] * (2 * len(layers)),
    })
    fake_ctx.store.save_parquet("vuf/meta.parquet", main_meta)
    fake_ctx.store.save_parquet("vuf/category_meta.parquet", cat_meta)

    rng = torch.Generator().manual_seed(0)
    for l in layers:
        # Three random unit vectors; cosines will be non-trivial.
        for variant in ("", "abstain_", "hedge_"):
            v = torch.randn(d_model, generator=rng)
            v = v / v.norm()
            name = f"vuf/direction_{variant}layer_{l}.safetensors" if variant else (
                f"vuf/direction_layer_{l}.safetensors"
            )
            fake_ctx.store.save_safetensors(name, {"direction": v.contiguous()})


def test_diagnostics_category_cosines_when_per_category_artefacts_present(fake_ctx):
    _seed_category_direction_files(fake_ctx, layers=(0, 1))
    rows = diagnostics._compute_category_cosines(fake_ctx)
    assert rows is not None
    assert len(rows) == 2
    for r in rows:
        for col in ("cosine_abstain_hedge", "cosine_abstain_main", "cosine_hedge_main"):
            assert -1.0 - 1e-6 <= r[col] <= 1.0 + 1e-6


def test_diagnostics_category_cosines_returns_none_when_no_artefacts(fake_ctx):
    # No vuf/category_meta.parquet — _compute returns None.
    assert diagnostics._compute_category_cosines(fake_ctx) is None


def test_diagnostics_category_cosines_returns_none_when_meta_empty(fake_ctx):
    # Write an empty category_meta — also returns None.
    fake_ctx.store.save_parquet(
        "vuf/meta.parquet",
        pd.DataFrame({
            "layer": [0], "path": ["x"], "raw_norm": [1.0],
            "n_uncertain": [1], "n_certain": [1], "pooling": ["last_token_q"],
        }),
    )
    fake_ctx.store.save_parquet(
        "vuf/category_meta.parquet",
        pd.DataFrame({
            "layer": pd.Series([], dtype="int64"),
            "variant": pd.Series([], dtype="object"),
            "path": pd.Series([], dtype="object"),
            "raw_norm": pd.Series([], dtype="float64"),
            "n_uncertain": pd.Series([], dtype="int64"),
            "n_certain": pd.Series([], dtype="int64"),
            "pooling": pd.Series([], dtype="object"),
        }),
    )
    assert diagnostics._compute_category_cosines(fake_ctx) is None
