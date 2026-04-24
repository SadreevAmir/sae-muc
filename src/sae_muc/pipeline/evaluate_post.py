"""evaluate_post: Tab.3 metrics per intervention variant + before/after comparison.

Reads `metrics.json` for the baseline, then for each variant in
`intervention/meta.parquet` builds a feature frame from that variant's
re-scored artefacts (judge_scores/accuracy/semantic_entropy) and computes
the same metric set. Writes one `metrics.json` per variant directory plus
a flat `metrics_comparison.parquet` at run root — one row per variant,
suitable for Tab.3 side-by-side display.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from sae_muc.pipeline import evaluate
from sae_muc.pipeline.context import PipelineContext

log = logging.getLogger(__name__)

OUTPUT_COMPARISON = "metrics_comparison.parquet"


def _variant_name(src_path: str) -> str:
    """`intervention/adaptive/generations.parquet` -> `adaptive`;
    `intervention/alpha_+0.50/generations.parquet` -> `alpha_+0.50`.
    """
    return Path(src_path).parent.name


def _flatten_metrics(metrics: dict) -> dict:
    """Drop nested `thresholds`/`split` keys to keep comparison rows flat."""
    flat = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            # Keep nested thresholds as-is in separate columns.
            for kk, vv in v.items():
                flat[f"{k}.{kk}"] = vv
        else:
            flat[k] = v
    return flat


def run(ctx: PipelineContext) -> list[str]:
    meta = ctx.store.load_parquet("intervention/meta.parquet")
    log.info("computing after-metrics for %d intervention variant(s)", len(meta))

    before = ctx.store.load_json("metrics.json")
    comparison_rows: list[dict] = [{"variant": "before", **_flatten_metrics(before)}]

    outputs: list[str] = []
    for _, row in meta.iterrows():
        src = str(row["path"])
        variant_dir = Path(src).parent
        name = _variant_name(src)
        df = evaluate.build_frame_from_paths(
            ctx,
            generations_path=str(variant_dir / "generations.parquet"),
            accuracy_path=str(variant_dir / "accuracy.parquet"),
            judge_path=str(variant_dir / "judge_scores.parquet"),
            se_path=str(variant_dir / "semantic_entropy.parquet"),
        )
        m = evaluate._compute_metrics(df, vu_threshold=0.5, su_threshold=None)
        metrics_path = str(variant_dir / "metrics.json")
        ctx.store.save_json(metrics_path, m)
        outputs.append(metrics_path)
        comparison_rows.append({"variant": name, **_flatten_metrics(m)})

    ctx.store.save_parquet(OUTPUT_COMPARISON, pd.DataFrame(comparison_rows))
    outputs.append(OUTPUT_COMPARISON)
    return outputs
