"""vuf: difference-in-means Verbal Uncertainty Feature, per layer.

Per paper §3.1: for each layer l,
    r_VU^(l) = mean(h(x)) over top-N uncertain questions
             - mean(h(x)) over bottom-N certain questions
then L2-normalised.

Per-question VU is the mean of judge scores over the N high-T sampled
generations (paper §2.2). The hidden state h(x) is pooled from the
stored full-sequence residual-stream tensor using the pooling mode from
config (default `last_token_q` — the paper's protocol).

We iterate over transformer layers only (layer_0 … layer_{n-1}); the
embedding output is available in `hidden_states/embedding.safetensors`
but is skipped by default for VUF.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from sae_muc.pipeline._utils import _pool
from sae_muc.pipeline.context import PipelineContext

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)

OUTPUT_META = "vuf/meta.parquet"
OUTPUT_SPLITS = "vuf/splits.parquet"
# Per-category direction metadata lives in a separate parquet so the legacy
# meta (one row per layer, consumed by intervene / sae_features / diagnostics)
# stays backward-compatible. Empty parquet if per_category is off.
OUTPUT_CATEGORY_META = "vuf/category_meta.parquet"

# Minimum members per category before we attempt to build a direction. Below
# this, diff-in-means is noise — skip the file and warn.
_MIN_CATEGORY_SIZE = 2


def _layer_in_path(layer: int) -> str:
    return f"hidden_states/layer_{layer}.safetensors"


def _means_out_path(layer: int) -> str:
    return f"vuf/means_layer_{layer}.safetensors"


def _direction_out_path(layer: int, variant: str = "main") -> str:
    """`main` keeps the legacy flat path; per-category gets a variant suffix.

    Keeping the legacy filename for variant='main' is what makes the
    extension byte-identical with disabled flags — checksums on
    direction_layer_*.safetensors match the pre-extension build exactly.
    """
    if variant == "main":
        return f"vuf/direction_layer_{layer}.safetensors"
    return f"vuf/direction_{variant}_layer_{layer}.safetensors"


def _per_question_mean_vu(judge_df: pd.DataFrame) -> pd.Series:
    sampled = judge_df[judge_df["kind"] == "sample"]
    return sampled.groupby("sample_id")["vu_score"].mean().dropna()


def _resolve_layers(layers_cfg: list[int] | str, n_layers: int) -> list[int]:
    if layers_cfg == "auto":
        return list(range(n_layers))
    return [int(x) for x in layers_cfg]


def _split_ids(vu_per_q: pd.Series, n_top: int, n_bot: int) -> tuple[list[str], list[str]]:
    """Top-N uncertain + bottom-N certain, clamped so the two sets don't overlap."""
    total = len(vu_per_q)
    n_top = min(max(n_top, 1), total)
    n_bot = min(max(n_bot, 1), total)
    if n_top + n_bot > total:
        n_top = total // 2
        n_bot = total - n_top
    sorted_ids = vu_per_q.sort_values(ascending=False)
    return (
        sorted_ids.iloc[:n_top].index.tolist(),
        sorted_ids.iloc[total - n_bot :].index.tolist(),
    )


def _split_ids_by_threshold(
    vu_per_q: pd.Series, *, vu_uncertain_min: float, vu_certain_max: float,
) -> tuple[list[str], list[str]]:
    """VU ≥ vu_uncertain_min → uncertain; VU ≤ vu_certain_max → certain
    (paper App G.1 protocol for the mitigation pipeline).

    Float tolerance: groupby.mean() of identical values can shift the result
    by a ULP (e.g. mean([0.05, 0.05, 0.05]) == 0.05000000000000001), so a
    user threshold of exactly 0.05 would silently exclude that row. Loosen
    both inequalities by a small epsilon — semantics don't change at any
    realistic decisiveness-judge precision (judges return 1-3 decimal
    places), but boundary rows are no longer at the mercy of FP.
    """
    eps = 1e-9
    uncertain = vu_per_q.index[vu_per_q >= vu_uncertain_min - eps].tolist()
    certain = vu_per_q.index[vu_per_q <= vu_certain_max + eps].tolist()
    return uncertain, certain


def _load_question_categories(ctx) -> dict[str, str]:
    """Read per-question categories from `categories.parquet` (if any).

    Returns `{sample_id: category}` for question-level rows with category
    in {"ABSTAIN", "HEDGE"} (MIXED / CONFIDENT / None are filtered out —
    they don't contribute to per-category directions).

    Empty dict if `categories.parquet` is missing (categorize stage off or
    skipped), if it's the zero-row stub, or if no question-level row has
    a usable category. The caller treats empty-dict as "skip per-category".
    """
    if not ctx.store.exists("categories.parquet"):
        return {}
    df = ctx.store.load_parquet("categories.parquet")
    if df.empty or "level" not in df.columns:
        return {}
    q = df[df["level"] == "question"]
    if q.empty:
        return {}
    out: dict[str, str] = {}
    for _, row in q.iterrows():
        cat = row.get("category")
        if cat in ("ABSTAIN", "HEDGE"):
            out[str(row["sample_id"])] = str(cat)
    return out


def run(ctx: PipelineContext) -> list[str]:
    import torch

    stage_cfg = ctx.cfg.stages.vuf

    judge = ctx.store.load_parquet("judge_scores.parquet")
    vu_per_q = _per_question_mean_vu(judge)
    if stage_cfg.selection == "vu_threshold":
        uncertain_ids, certain_ids = _split_ids_by_threshold(
            vu_per_q,
            vu_uncertain_min=stage_cfg.vu_uncertain_min,
            vu_certain_max=stage_cfg.vu_certain_max,
        )
        if not uncertain_ids or not certain_ids:
            log.warning(
                "vuf: vu_threshold selection produced empty split "
                "(uncertain=%d @>=%.2f, certain=%d @<=%.2f); "
                "falling back to top_n with n_top=%d / n_bot=%d",
                len(uncertain_ids), stage_cfg.vu_uncertain_min,
                len(certain_ids), stage_cfg.vu_certain_max,
                stage_cfg.n_top, stage_cfg.n_bot,
            )
            uncertain_ids, certain_ids = _split_ids(vu_per_q, stage_cfg.n_top, stage_cfg.n_bot)
    else:
        uncertain_ids, certain_ids = _split_ids(vu_per_q, stage_cfg.n_top, stage_cfg.n_bot)

    # Per-category labels — empty dict if categorize stage was off or
    # produced no labelled questions. `per_category_active` gates the
    # extra-direction work without touching the main path.
    question_categories = (
        _load_question_categories(ctx) if stage_cfg.per_category else {}
    )
    per_category_active = bool(question_categories)

    # Persist split for downstream stages (sae_features) to reuse without
    # recomputing the sort. One row per question with the VU that drove it.
    # `category` column is always written (defaults to empty string) so the
    # schema is stable; existing consumers (sae_features) filter `split`
    # only and don't touch it.
    splits_rows: list[dict] = []
    for sid, vu in vu_per_q.items():
        if sid in uncertain_ids:
            split = "uncertain"
        elif sid in certain_ids:
            split = "certain"
        else:
            split = "middle"
        splits_rows.append(
            {
                "sample_id": sid,
                "mean_vu": float(vu),
                "split": split,
                "category": question_categories.get(sid, ""),
            }
        )
    ctx.store.save_parquet(OUTPUT_SPLITS, pd.DataFrame(splits_rows))

    meta = ctx.store.load_parquet("hidden_states/meta.parquet").set_index("sample_id")
    n_layers = int(meta.iloc[0]["n_layers"])
    layers = _resolve_layers(stage_cfg.layers, n_layers)

    abstain_ids = [sid for sid in uncertain_ids if question_categories.get(sid) == "ABSTAIN"]
    hedge_ids = [sid for sid in uncertain_ids if question_categories.get(sid) == "HEDGE"]
    log.info(
        "diff-in-means VUF: %d uncertain vs %d certain (pooling=%s, layers=%d)",
        len(uncertain_ids), len(certain_ids), stage_cfg.pooling, len(layers),
    )
    if per_category_active:
        log.info(
            "per-category active: %d ABSTAIN / %d HEDGE in uncertain bucket",
            len(abstain_ids), len(hedge_ids),
        )

    outputs: list[str] = []
    dir_meta: list[dict] = []
    category_meta: list[dict] = []

    for layer in layers:
        tensors = ctx.store.load_safetensors(_layer_in_path(layer))

        def pooled(sid: str, _tensors=tensors) -> "torch.Tensor":
            # Upcast to float32 before the diff-in-means so a bf16 hidden-state
            # artefact (large-N runs) doesn't lose precision when averaging
            # hundreds of vectors. No-op when the artefact is already float32.
            return _pool(
                _tensors[sid].float(),
                stage_cfg.pooling,
                int(meta.loc[sid, "question_len"]),
                int(meta.loc[sid, "seq_len"]),
            )

        certain_stack = torch.stack([pooled(sid) for sid in certain_ids])
        certain_mean = certain_stack.mean(dim=0)

        def _direction_from(ids: list[str]) -> tuple["torch.Tensor", float]:
            stacked = torch.stack([pooled(sid) for sid in ids])
            raw = stacked.mean(dim=0) - certain_mean
            n = raw.norm().item()
            return (raw / n if n > 0 else raw), float(n)

        # Main direction — legacy schema, byte-identical with the pre-extension
        # build when per_category is off. Lands in `vuf/meta.parquet`.
        main_dir, main_norm = _direction_from(uncertain_ids)
        main_path = _direction_out_path(layer, "main")
        ctx.store.save_safetensors(main_path, {"direction": main_dir.contiguous()})
        outputs.append(main_path)

        # Sidecar: the two per-set MEAN activations (pre-diff, pre-norm). The
        # universal-VUF combine stage pools these across datasets weighted by
        # counts to reconstruct a single diff-in-means over the union of
        # contrast sets (paper App G.1), without storing every activation.
        uncertain_mean = torch.stack([pooled(sid) for sid in uncertain_ids]).mean(dim=0)
        means_path = _means_out_path(layer)
        ctx.store.save_safetensors(
            means_path,
            {
                "mean_uncertain": uncertain_mean.contiguous(),
                "mean_certain": certain_mean.contiguous(),
            },
        )
        outputs.append(means_path)
        dir_meta.append(
            {
                "layer": layer,
                "path": main_path,
                "raw_norm": main_norm,
                "n_uncertain": len(uncertain_ids),
                "n_certain": len(certain_ids),
                "pooling": stage_cfg.pooling,
            }
        )

        # Per-category directions — land in `vuf/category_meta.parquet`,
        # separate parquet so existing consumers of `meta.parquet` don't
        # see duplicate layer rows.
        category_directions: dict[str, "torch.Tensor"] = {}
        if per_category_active:
            for variant, ids in (("abstain", abstain_ids), ("hedge", hedge_ids)):
                if len(ids) < _MIN_CATEGORY_SIZE:
                    log.warning(
                        "vuf.per_category: layer %d skipping %s direction "
                        "(%d members < %d minimum)",
                        layer, variant, len(ids), _MIN_CATEGORY_SIZE,
                    )
                    continue
                d, n = _direction_from(ids)
                category_directions[variant] = d
                path = _direction_out_path(layer, variant)
                ctx.store.save_safetensors(path, {"direction": d.contiguous()})
                outputs.append(path)
                category_meta.append(
                    {
                        "layer": layer,
                        "variant": variant,
                        "path": path,
                        "raw_norm": n,
                        "n_uncertain": len(ids),
                        "n_certain": len(certain_ids),
                        "pooling": stage_cfg.pooling,
                    }
                )

            # Cross-category direction: normalize(unit_abstain − unit_hedge),
            # where unit_X is the L2-normalised category direction (i.e. the
            # version downstream steering hooks would actually use). The
            # `certain` reference cancels because both inputs share it before
            # normalisation, but the magnitudes don't — using unit vectors
            # rather than the raw diff-in-means gives a steering axis whose
            # scale is consistent across categories with different
            # uncertain-vs-certain separations. Skip when either category
            # was unavailable.
            if "abstain" in category_directions and "hedge" in category_directions:
                cross_raw = category_directions["abstain"] - category_directions["hedge"]
                cross_norm = cross_raw.norm().item()
                cross_dir = cross_raw / cross_norm if cross_norm > 0 else cross_raw
                cross_path = _direction_out_path(layer, "abstain_vs_hedge")
                ctx.store.save_safetensors(
                    cross_path, {"direction": cross_dir.contiguous()}
                )
                outputs.append(cross_path)
                category_meta.append(
                    {
                        "layer": layer,
                        "variant": "abstain_vs_hedge",
                        "path": cross_path,
                        "raw_norm": float(cross_norm),
                        "n_uncertain": len(abstain_ids) + len(hedge_ids),
                        "n_certain": 0,  # cross direction uses no certain pool
                        "pooling": stage_cfg.pooling,
                    }
                )

    ctx.store.save_parquet(OUTPUT_META, pd.DataFrame(dir_meta))
    outputs.append(OUTPUT_META)
    outputs.append(OUTPUT_SPLITS)

    # Always write category_meta — empty parquet when per_category is off,
    # one row per (layer, variant) otherwise. Stable schema, downstream
    # diagnostics gates on len(df) > 0.
    cat_df = pd.DataFrame(category_meta) if category_meta else pd.DataFrame(
        {
            "layer": pd.Series([], dtype="int64"),
            "variant": pd.Series([], dtype="object"),
            "path": pd.Series([], dtype="object"),
            "raw_norm": pd.Series([], dtype="float64"),
            "n_uncertain": pd.Series([], dtype="int64"),
            "n_certain": pd.Series([], dtype="int64"),
            "pooling": pd.Series([], dtype="object"),
        }
    )
    ctx.store.save_parquet(OUTPUT_CATEGORY_META, cat_df)
    outputs.append(OUTPUT_CATEGORY_META)
    return outputs
