"""diagnostics: measure intervention side-effects on general LM capability.

Sidecar stage at the end of the pipeline. Two complementary modes:

1. **Per-variant** (always on, default cross-run workflow).
   For every row in `intervention/meta.parquet` we install the *same* hook
   the `intervene` stage used and score it on:
     * **WikiText-2-raw-v1** validation perplexity — vanilla LM probe
       (CAA / ITI / RepE / ReFT standard).
     * **MMLU** (4-way MC, accuracy + NLL on the gold letter).
     * **HellaSwag** (4-way completion, length-normalised NLL argmin).
     * **GSM8K** No-CoT (brief greedy gen + numeric parse; NLL on gold).
     * **KL divergence** of next-token distributions on cached QA prompts.
   Output: one row per variant in each of `perplexity.parquet`,
   `benchmarks.parquet`, `kl.parquet`.

2. **Multi-method × α matrix** (in-run sweep, opt-in).
   When `cfg.stages.diagnostics.compare_methods` and `alpha_sweep` are
   both populated, the stage *also* rebuilds hooks for every
   `(method, alpha)` pair from the same VUF directions / SAE feature
   stats, runs the same dataset scorers, and writes a long-format
   `method_alpha_sweep.parquet`. Independent of `intervention/meta.parquet`
   — meant to give a clean "method × α × dataset" matrix in a single run.

The stage no-ops gracefully when:
  * `cfg.stages.diagnostics.enabled == False`,
  * the LLM backend lacks `forward_nll_with_hook` (e.g. OpenRouter — no
    logits),
  * `intervention/meta.parquet` is missing AND no multi-method sweep is
    requested (no work to do).

Hook fidelity: every hook is rebuilt from explicit args via
`intervene.build_per_layer_hooks` — never reads `ctx.cfg.stages.intervene`,
which could differ on a re-run.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from sae_muc.pipeline import diagnostics_datasets as dd
from sae_muc.pipeline._utils import _resolve_layers
from sae_muc.pipeline.context import PipelineContext
from sae_muc.pipeline.intervene import (
    _load_sae_feature_stats,
    build_per_layer_hooks,
)

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)

OUTPUT_PPL = "diagnostics/perplexity.parquet"
OUTPUT_BENCH = "diagnostics/benchmarks.parquet"
OUTPUT_KL = "diagnostics/kl.parquet"
OUTPUT_SWEEP = "diagnostics/method_alpha_sweep.parquet"
OUTPUT_CATEGORY = "diagnostics/category_directions.parquet"
OUTPUT_CATEGORY_SAE = "diagnostics/category_sae_jaccards.parquet"
OUTPUT_SUMMARY = "diagnostics/summary.json"

_SAE_FEATURE_METHODS = ("sae_emd", "sae_clamp")

_SYNTHETIC_CORPUS = (
    "The quick brown fox jumps over the lazy dog. "
    "She sells seashells by the seashore. "
    "Sphinx of black quartz judge my vow. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
    "Bright vixens jump dozy fowl quack. "
    "Waltz nymph for quick jigs vex Bud. "
) * 4


# ---------- helpers ------------------------------------------------------------


def _parse_layers_field(layers_str: str) -> list[int]:
    """Inverse of `_layers_str` in intervene.py: '15', '15-17', '0,2,5' → list[int]."""
    s = str(layers_str).strip()
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    if "," in s:
        return [int(x) for x in s.split(",")]
    return [int(s)]


def _variant_name(path: str) -> str:
    from pathlib import Path as _P

    return _P(path).parent.name


def _load_corpus(ctx: PipelineContext) -> str:
    n_chars = int(ctx.cfg.stages.diagnostics.corpus_n_chars)
    if n_chars <= 0:
        return _SYNTHETIC_CORPUS
    try:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        text = "\n".join(row["text"] for row in ds).strip()
        return text[:n_chars] if n_chars else text
    except Exception as e:  # noqa: BLE001
        log.warning(
            "diagnostics: wikitext-2-raw-v1 load failed (%s); using synthetic corpus",
            e,
        )
        return _SYNTHETIC_CORPUS


def _select_kl_prompts(ctx: PipelineContext) -> list[str]:
    from sae_muc.data.prompts import format_answer_prompt

    n_max = int(ctx.cfg.stages.diagnostics.kl_max_prompts)
    samples = ctx.store.load_parquet("samples.parquet")
    questions = list(samples["question"])[:n_max] if n_max > 0 else list(samples["question"])
    return [format_answer_prompt(q, eliciting=True) for q in questions]


def _load_directions(
    ctx: PipelineContext, layers: list[int]
) -> dict[int, "torch.Tensor"]:
    return {
        layer: ctx.store.load_safetensors(
            f"vuf/direction_layer_{layer}.safetensors"
        )["direction"]
        for layer in layers
    }


def _variant_alpha(row: pd.Series) -> float:
    alpha = row.get("alpha")
    if alpha is None or pd.isna(alpha):
        return float(row.get("mean_alpha", 0.0))
    return float(alpha)


def _build_hooks_for(
    ctx: PipelineContext,
    *,
    method: str,
    layers: list[int],
    alpha: float,
) -> dict[int, "callable"]:
    """Build per-layer hooks for `(method, alpha)` from current artefacts.

    Reads VUF directions and (for SAE methods) sae_features/stats.parquet.
    Always reads `sae_emd_delta` from the *current* cfg.stages.intervene
    — this knob isn't variant-specific in `intervention/meta.parquet` and
    is constant for the whole project run.
    """
    directions = _load_directions(ctx, layers)
    layer_stats = (
        _load_sae_feature_stats(ctx, layers)
        if method in _SAE_FEATURE_METHODS
        else None
    )
    return build_per_layer_hooks(
        method=method,
        sae_emd_delta=ctx.cfg.stages.intervene.sae_emd_delta,
        apply_during_generation=True,  # diagnostics has no decode steps to skip.
        layers=layers,
        directions=directions,
        alpha=alpha,
        saes=ctx.saes,
        layer_stats=layer_stats,
    )


# ---------- scoring loop ------------------------------------------------------


def _score_one_hook(
    ctx: PipelineContext,
    *,
    hook_layer,
    hook_fn,
    corpus: str | None,
    kl_prompts: list[str] | None,
    mmlu_items: list[dict],
    hella_items: list[dict],
    gsm_items: list[dict],
) -> dict[str, float | int]:
    """Run every enabled scorer once. None entries skip that scorer."""
    cfg = ctx.cfg.stages.diagnostics
    out: dict[str, float | int] = {}

    if "wikitext" in cfg.corpora and corpus is not None:
        sum_nll, n_tokens = ctx.llm.forward_nll_with_hook(
            corpus, hook_layer=hook_layer, hook_fn=hook_fn,
        )
        out["wiki_sum_nll"] = float(sum_nll)
        out["wiki_n_tokens"] = int(n_tokens)
        out["wiki_mean_nll"] = float(sum_nll / n_tokens) if n_tokens > 0 else float("nan")
        out["wiki_ppl"] = float(np.exp(out["wiki_mean_nll"])) if n_tokens > 0 else float("nan")
    if kl_prompts is not None and hook_layer is not None and hook_fn is not None:
        kl = ctx.llm.forward_kl_with_hook(
            kl_prompts, hook_layer=hook_layer, hook_fn=hook_fn,
        )
        out.update({f"kl_{k}": v for k, v in kl.items()})

    if "mmlu" in cfg.corpora and mmlu_items:
        mmlu = dd.score_mmlu(ctx.llm, mmlu_items, hook_layer=hook_layer, hook_fn=hook_fn)
        out["mmlu_accuracy"] = mmlu["accuracy"]
        out["mmlu_mean_nll"] = mmlu["mean_nll"]
        out["mmlu_n"] = mmlu["n"]
    if "hellaswag" in cfg.corpora and hella_items:
        hs = dd.score_hellaswag(ctx.llm, hella_items, hook_layer=hook_layer, hook_fn=hook_fn)
        out["hellaswag_accuracy"] = hs["accuracy"]
        out["hellaswag_mean_nll"] = hs["mean_nll"]
        out["hellaswag_n"] = hs["n"]
    if "gsm8k" in cfg.corpora and gsm_items:
        gsm = dd.score_gsm8k(
            ctx.llm, gsm_items, hook_layer=hook_layer, hook_fn=hook_fn,
            max_new_tokens=cfg.gsm8k_max_new_tokens, seed=int(ctx.cfg.seed),
        )
        out["gsm8k_accuracy"] = gsm["accuracy"]
        out["gsm8k_mean_nll"] = gsm["mean_nll"]
        out["gsm8k_n"] = gsm["n"]

    return out


# ---------- run --------------------------------------------------------------


def _write_disabled_stub(ctx: PipelineContext, reason: str) -> list[str]:
    ctx.store.save_json(
        OUTPUT_SUMMARY,
        {"skipped": True, "reason": reason, "variants": [], "sweep_rows": 0},
    )
    return [OUTPUT_SUMMARY]


def _populated_meta(ctx: PipelineContext) -> pd.DataFrame | None:
    if not ctx.store.exists("intervention/meta.parquet"):
        return None
    return ctx.store.load_parquet("intervention/meta.parquet")


def _load_benchmark_items(cfg) -> tuple[list[dict], list[dict], list[dict]]:
    mmlu = dd.load_mmlu(cfg.n_mmlu) if "mmlu" in cfg.corpora else []
    hella = dd.load_hellaswag(cfg.n_hellaswag) if "hellaswag" in cfg.corpora else []
    gsm = dd.load_gsm8k(cfg.n_gsm8k) if "gsm8k" in cfg.corpora else []
    return mmlu, hella, gsm


def _expand_ppl_row(variant: str, method: str, mode: str, alpha: float, layers: str, scores: dict) -> dict:
    """Slice WikiText-related fields out of a `_score_one_hook` dict."""
    return {
        "variant": variant,
        "method": method,
        "mode": mode,
        "alpha": float(alpha),
        "layers": str(layers),
        "mean_nll": scores.get("wiki_mean_nll", float("nan")),
        "mean_ppl": scores.get("wiki_ppl", float("nan")),
        "n_tokens": int(scores.get("wiki_n_tokens", 0)),
    }


def _expand_bench_row(variant: str, method: str, mode: str, alpha: float, layers: str, scores: dict) -> dict:
    return {
        "variant": variant,
        "method": method,
        "mode": mode,
        "alpha": float(alpha),
        "layers": str(layers),
        "mmlu_accuracy": scores.get("mmlu_accuracy", float("nan")),
        "mmlu_mean_nll": scores.get("mmlu_mean_nll", float("nan")),
        "mmlu_n": int(scores.get("mmlu_n", 0)),
        "hellaswag_accuracy": scores.get("hellaswag_accuracy", float("nan")),
        "hellaswag_mean_nll": scores.get("hellaswag_mean_nll", float("nan")),
        "hellaswag_n": int(scores.get("hellaswag_n", 0)),
        "gsm8k_accuracy": scores.get("gsm8k_accuracy", float("nan")),
        "gsm8k_mean_nll": scores.get("gsm8k_mean_nll", float("nan")),
        "gsm8k_n": int(scores.get("gsm8k_n", 0)),
    }


def _expand_kl_row(variant: str, method: str, mode: str, alpha: float, layers: str, scores: dict) -> dict:
    return {
        "variant": variant,
        "method": method,
        "mode": mode,
        "alpha": float(alpha),
        "layers": str(layers),
        "mean_kl_all_positions": float(scores.get("kl_mean_kl_all_positions", 0.0)),
        "mean_kl_last_prompt_token": float(scores.get("kl_mean_kl_last_prompt_token", 0.0)),
        "top1_disagreement_rate": float(scores.get("kl_top1_disagreement_rate", 0.0)),
        "top5_mass_delta": float(scores.get("kl_top5_mass_delta", 0.0)),
        "n_prompts": int(scores.get("kl_n_prompts", 0)),
        "n_tokens": int(scores.get("kl_n_tokens", 0)),
    }


def _compute_category_cosines(ctx: PipelineContext) -> list[dict] | None:
    """Per-layer cosines between main / abstain / hedge VUF directions.

    Reads `vuf/category_meta.parquet` (produced by `vuf.run` when
    `vuf.per_category=True`) and pairs each entry with the corresponding
    main direction. Returns one row per layer with three cosines, or None
    when category_meta is empty / missing (per-category was off).
    """
    if not ctx.store.exists("vuf/category_meta.parquet"):
        return None
    cat_meta = ctx.store.load_parquet("vuf/category_meta.parquet")
    if cat_meta.empty:
        return None
    main_meta = ctx.store.load_parquet("vuf/meta.parquet")

    def _load(path: str):
        import torch  # local — keep module-level light

        return ctx.store.load_safetensors(path)["direction"]

    def _cos(a, b) -> float:
        denom = float((a.norm() * b.norm()).item())
        if denom <= 1e-12:
            return float("nan")
        return float((a @ b).item() / denom)

    rows: list[dict] = []
    layers = sorted(set(int(x) for x in cat_meta["layer"].tolist()))
    for layer in layers:
        main_path_rows = main_meta[main_meta["layer"] == layer]
        if main_path_rows.empty:
            continue
        main_path = str(main_path_rows.iloc[0]["path"])
        cat_rows = cat_meta[cat_meta["layer"] == layer]
        by_variant = {
            str(r["variant"]): str(r["path"]) for _, r in cat_rows.iterrows()
        }
        # Skip layers where neither category was successfully built.
        if "abstain" not in by_variant and "hedge" not in by_variant:
            continue

        r_main = _load(main_path)
        r_abstain = _load(by_variant["abstain"]) if "abstain" in by_variant else None
        r_hedge = _load(by_variant["hedge"]) if "hedge" in by_variant else None
        r_cross = (
            _load(by_variant["abstain_vs_hedge"])
            if "abstain_vs_hedge" in by_variant
            else None
        )

        n_abstain = (
            int(cat_rows[cat_rows["variant"] == "abstain"].iloc[0]["n_uncertain"])
            if "abstain" in by_variant
            else 0
        )
        n_hedge = (
            int(cat_rows[cat_rows["variant"] == "hedge"].iloc[0]["n_uncertain"])
            if "hedge" in by_variant
            else 0
        )

        rows.append(
            {
                "layer": int(layer),
                # disentanglement: are abstain and hedge two axes (cos≈0)
                # or one (cos≈1)?
                "cosine_abstain_hedge": (
                    _cos(r_abstain, r_hedge)
                    if r_abstain is not None and r_hedge is not None
                    else float("nan")
                ),
                # alignment: how much of paper's `main` direction is
                # actually each per-category direction?
                "cosine_abstain_main": (
                    _cos(r_abstain, r_main) if r_abstain is not None else float("nan")
                ),
                "cosine_hedge_main": (
                    _cos(r_hedge, r_main) if r_hedge is not None else float("nan")
                ),
                # cross axis: r_abstain − r_hedge — the steering direction
                # that flips uncertain *type* (abstain ↔ hedge) without
                # changing overall uncertainty level. cosine_cross_main
                # measures whether paper's main VUF accidentally encodes
                # this type-flip axis (high) or is purely the
                # uncertain-vs-certain axis (low).
                "cosine_cross_main": (
                    _cos(r_cross, r_main) if r_cross is not None else float("nan")
                ),
                "n_abstain": n_abstain,
                "n_hedge": n_hedge,
            }
        )
    return rows


def _compute_category_sae_jaccards(ctx: PipelineContext) -> list[dict] | None:
    """Per-layer Jaccard between SAE-feature top-K sets across main /
    abstain / hedge variants.

    `Jaccard(A, B) = |A ∩ B| / |A ∪ B|` ∈ [0, 1].
        ≈ 0 — disjoint top-K sets → SAE encodes the categories with
              different features (positive disentanglement signal).
        ≈ 1 — same top-K → SAE sees them as the same concept.

    Reads `sae_features/stats.parquet` (main) and
    `sae_features/category_stats.parquet` (per-variant). Returns None
    when no per-category SAE work was done.
    """
    if not ctx.store.exists("sae_features/category_stats.parquet"):
        return None
    cat_stats = ctx.store.load_parquet("sae_features/category_stats.parquet")
    if cat_stats.empty:
        return None
    if not ctx.store.exists("sae_features/stats.parquet"):
        return None
    main_stats = ctx.store.load_parquet("sae_features/stats.parquet")

    def _topk_set(df: pd.DataFrame) -> set[int]:
        return {int(x) for x in df[df["selected_as"] == "uncertainty"]["feature_id"].tolist()}

    def _jaccard(a: set[int], b: set[int]) -> float:
        if not a and not b:
            return float("nan")
        union = a | b
        if not union:
            return float("nan")
        return len(a & b) / len(union)

    rows: list[dict] = []
    layers = sorted(set(int(x) for x in cat_stats["layer"].tolist()))
    for layer in layers:
        main_layer = main_stats[main_stats["layer"] == layer]
        cat_layer = cat_stats[cat_stats["layer"] == layer]
        if main_layer.empty:
            continue
        main_top = _topk_set(main_layer)
        abstain_top = _topk_set(cat_layer[cat_layer["variant"] == "abstain"])
        hedge_top = _topk_set(cat_layer[cat_layer["variant"] == "hedge"])
        # Skip layers where neither category got any features.
        if not abstain_top and not hedge_top:
            continue
        rows.append(
            {
                "layer": int(layer),
                # Headline number — analogous to cosine_abstain_hedge for
                # linear directions but for the discrete SAE feature view.
                "jaccard_abstain_hedge": _jaccard(abstain_top, hedge_top),
                "jaccard_abstain_main": _jaccard(abstain_top, main_top),
                "jaccard_hedge_main": _jaccard(hedge_top, main_top),
                "n_main_topk": int(len(main_top)),
                "n_abstain_topk": int(len(abstain_top)),
                "n_hedge_topk": int(len(hedge_top)),
            }
        )
    return rows


def _sweep_layers(ctx: PipelineContext) -> list[int]:
    """Layers used by the in-run multi-method sweep.

    Mirrors the resolution from `intervene.run` so the same layer policy
    applies: explicit list, single int, 'auto', or 'paper_range'.
    """
    vuf_meta = ctx.store.load_parquet("vuf/meta.parquet")
    available = sorted(int(x) for x in vuf_meta["layer"].tolist())
    return _resolve_layers(
        ctx.cfg.stages.intervene.layer, available,
        model_name=ctx.cfg.model.name,
    )


def run(ctx: PipelineContext) -> list[str]:
    cfg = ctx.cfg.stages.diagnostics
    if not cfg.enabled:
        log.info("diagnostics: disabled by config; writing stub")
        return _write_disabled_stub(ctx, reason="disabled by stages.diagnostics.enabled=false")

    if getattr(ctx.llm, "forward_nll_with_hook", None) is None:
        log.warning(
            "diagnostics: backend %s lacks forward_nll_with_hook; stage skipped",
            type(ctx.llm).__name__,
        )
        return _write_disabled_stub(
            ctx, reason=f"backend {type(ctx.llm).__name__} has no logits API"
        )

    meta = _populated_meta(ctx)
    do_sweep = bool(cfg.compare_methods) and bool(cfg.alpha_sweep)
    if meta is None and not do_sweep:
        log.warning(
            "diagnostics: intervention/meta.parquet missing and no sweep requested; "
            "stage skipped"
        )
        return _write_disabled_stub(ctx, reason="intervention/meta.parquet not found")

    corpus = _load_corpus(ctx) if "wikitext" in cfg.corpora else None
    kl_prompts = _select_kl_prompts(ctx) if ctx.store.exists("samples.parquet") else []
    mmlu_items, hella_items, gsm_items = _load_benchmark_items(cfg)
    log.info(
        "diagnostics: corpora=%s (wiki=%s, mmlu=%d, hellaswag=%d, gsm8k=%d), kl_prompts=%d",
        cfg.corpora,
        len(corpus) if corpus else 0,
        len(mmlu_items), len(hella_items), len(gsm_items), len(kl_prompts),
    )

    outputs: list[str] = []

    # -------- baseline (no hook): scored once, reused across modes ------------
    baseline_scores = _score_one_hook(
        ctx,
        hook_layer=None, hook_fn=None,
        corpus=corpus, kl_prompts=None,  # KL against itself is zero by definition
        mmlu_items=mmlu_items, hella_items=hella_items, gsm_items=gsm_items,
    )
    base_ppl = float(baseline_scores.get("wiki_ppl", float("nan")))

    ppl_rows: list[dict] = [
        _expand_ppl_row("baseline", "none", "none", 0.0, "", baseline_scores)
        | {"ppl_ratio_vs_baseline": 1.0}
    ]
    bench_rows: list[dict] = [
        _expand_bench_row("baseline", "none", "none", 0.0, "", baseline_scores)
    ]
    kl_rows: list[dict] = [
        {
            "variant": "baseline", "method": "none", "mode": "none",
            "alpha": 0.0, "layers": "",
            "mean_kl_all_positions": 0.0, "mean_kl_last_prompt_token": 0.0,
            "top1_disagreement_rate": 0.0, "top5_mass_delta": 0.0,
            "n_prompts": int(len(kl_prompts)), "n_tokens": 0,
        }
    ]
    sweep_rows: list[dict] = []

    # -------- mode 1: per-variant from intervention/meta.parquet --------------
    if meta is not None:
        for _, row in meta.iterrows():
            variant = _variant_name(str(row["path"]))
            method = str(row["method"])
            mode = str(row.get("mode", "fixed"))
            layers = _parse_layers_field(row["layers"])
            alpha = _variant_alpha(row)
            log.info(
                "diagnostics variant=%s method=%s mode=%s layers=%s alpha=%.3f",
                variant, method, mode, row["layers"], alpha,
            )
            hooks = _build_hooks_for(ctx, method=method, layers=layers, alpha=alpha)
            scores = _score_one_hook(
                ctx,
                hook_layer=layers, hook_fn=hooks,
                corpus=corpus, kl_prompts=kl_prompts,
                mmlu_items=mmlu_items, hella_items=hella_items, gsm_items=gsm_items,
            )
            ppl_row = _expand_ppl_row(variant, method, mode, alpha, row["layers"], scores)
            ppl_row["ppl_ratio_vs_baseline"] = (
                ppl_row["mean_ppl"] / base_ppl if base_ppl and not np.isnan(base_ppl) else float("nan")
            )
            ppl_rows.append(ppl_row)
            bench_rows.append(_expand_bench_row(variant, method, mode, alpha, row["layers"], scores))
            kl_rows.append(_expand_kl_row(variant, method, mode, alpha, row["layers"], scores))

    # -------- mode 2: multi-method × alpha sweep ------------------------------
    if do_sweep:
        sweep_layers = _sweep_layers(ctx)
        log.info(
            "diagnostics sweep: methods=%s × alphas=%s on layers=%s",
            list(cfg.compare_methods), list(cfg.alpha_sweep), sweep_layers,
        )
        # Baseline row for the sweep frame (one row per dataset for plotability).
        for ds_name, acc_key, nll_key in (
            ("wikitext",  None,                "wiki_mean_nll"),
            ("mmlu",      "mmlu_accuracy",     "mmlu_mean_nll"),
            ("hellaswag", "hellaswag_accuracy","hellaswag_mean_nll"),
            ("gsm8k",     "gsm8k_accuracy",    "gsm8k_mean_nll"),
        ):
            if ds_name not in cfg.corpora:
                continue
            sweep_rows.append({
                "method": "baseline",
                "alpha": 0.0,
                "dataset": ds_name,
                "accuracy": baseline_scores.get(acc_key) if acc_key else float("nan"),
                "mean_nll": baseline_scores.get(nll_key, float("nan")),
                "layers": ",".join(str(l) for l in sweep_layers),
            })

        for method in cfg.compare_methods:
            for alpha in cfg.alpha_sweep:
                a = float(alpha)
                log.info("sweep: method=%s alpha=%+.3f", method, a)
                hooks = _build_hooks_for(ctx, method=method, layers=sweep_layers, alpha=a)
                scores = _score_one_hook(
                    ctx,
                    hook_layer=sweep_layers, hook_fn=hooks,
                    corpus=corpus, kl_prompts=None,  # KL is per-variant only
                    mmlu_items=mmlu_items, hella_items=hella_items, gsm_items=gsm_items,
                )
                for ds_name, acc_key, nll_key in (
                    ("wikitext",  None,                 "wiki_mean_nll"),
                    ("mmlu",      "mmlu_accuracy",      "mmlu_mean_nll"),
                    ("hellaswag", "hellaswag_accuracy", "hellaswag_mean_nll"),
                    ("gsm8k",     "gsm8k_accuracy",     "gsm8k_mean_nll"),
                ):
                    if ds_name not in cfg.corpora:
                        continue
                    sweep_rows.append({
                        "method": method,
                        "alpha": a,
                        "dataset": ds_name,
                        "accuracy": scores.get(acc_key) if acc_key else float("nan"),
                        "mean_nll": scores.get(nll_key, float("nan")),
                        "layers": ",".join(str(l) for l in sweep_layers),
                    })

    # -------- write artefacts -------------------------------------------------
    ctx.store.save_parquet(OUTPUT_PPL, pd.DataFrame(ppl_rows))
    outputs.append(OUTPUT_PPL)
    if any("mmlu" in cfg.corpora for _ in [0]) or any(
        k in cfg.corpora for k in ("mmlu", "hellaswag", "gsm8k")
    ):
        ctx.store.save_parquet(OUTPUT_BENCH, pd.DataFrame(bench_rows))
        outputs.append(OUTPUT_BENCH)
    ctx.store.save_parquet(OUTPUT_KL, pd.DataFrame(kl_rows))
    outputs.append(OUTPUT_KL)
    if sweep_rows:
        ctx.store.save_parquet(OUTPUT_SWEEP, pd.DataFrame(sweep_rows))
        outputs.append(OUTPUT_SWEEP)

    # Per-category VUF cosine analysis — disentanglement of ABSTAIN vs HEDGE.
    # Empty when `vuf.per_category` was off (no category_meta.parquet rows).
    cat_rows = _compute_category_cosines(ctx)
    if cat_rows:
        ctx.store.save_parquet(OUTPUT_CATEGORY, pd.DataFrame(cat_rows))
        outputs.append(OUTPUT_CATEGORY)

    # SAE-feature Jaccard analysis — second, independent disentanglement
    # readout. Linear cosines might say "same axis" while SAE jaccards
    # say "different features" — that's the interesting case (SAE picks
    # up structure the linear probe misses).
    sae_rows = _compute_category_sae_jaccards(ctx)
    if sae_rows:
        ctx.store.save_parquet(OUTPUT_CATEGORY_SAE, pd.DataFrame(sae_rows))
        outputs.append(OUTPUT_CATEGORY_SAE)

    summary = {
        "skipped": False,
        "corpora": list(cfg.corpora),
        "n_variants": int(len(meta)) if meta is not None else 0,
        "sweep_rows": int(len(sweep_rows)),
        "baseline_ppl": base_ppl,
        "baseline_mmlu_accuracy": float(baseline_scores.get("mmlu_accuracy", float("nan"))),
        "baseline_hellaswag_accuracy": float(baseline_scores.get("hellaswag_accuracy", float("nan"))),
        "baseline_gsm8k_accuracy": float(baseline_scores.get("gsm8k_accuracy", float("nan"))),
        "variants": [
            {
                "variant": p["variant"],
                "method": p["method"],
                "mode": p["mode"],
                "alpha": p["alpha"],
                "layers": p["layers"],
                "mean_ppl": p["mean_ppl"],
                "ppl_ratio_vs_baseline": p["ppl_ratio_vs_baseline"],
                "mmlu_accuracy": b["mmlu_accuracy"],
                "hellaswag_accuracy": b["hellaswag_accuracy"],
                "gsm8k_accuracy": b["gsm8k_accuracy"],
            }
            for p, b in zip(ppl_rows[1:], bench_rows[1:], strict=True)
        ],
    }
    ctx.store.save_json(OUTPUT_SUMMARY, summary)
    outputs.append(OUTPUT_SUMMARY)
    return outputs
