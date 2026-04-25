"""evaluate: Table-3 metrics for the current run (pre-intervention state).

Computes per-question / aggregate metrics on the baseline generations:

  * hallucination_rate          — fraction with (not correct) AND (not refusal)
  * confident_hallucination_rate — fraction that are hallucinated AND VU < threshold
  * correct_rate                — fraction correct
  * refusal_rate                — fraction that refused
  * vu_su_disagreement_rate     — fraction where (VU > τ_vu) != (SU > τ_su)
  * correlation                 — Pearson correlation between VU and SU
  * vu_correct_mean             — mean VU among correct answers
  * vu_incorrect_mean           — mean VU among incorrect answers

Thresholds come from `cfg.stages.evaluate.{vu_threshold,su_threshold}` with
reasonable defaults (VU=0.5, SU=median). Post-intervention metrics (per α)
require re-running judge/semantic_entropy/accuracy_judge on the intervened
generations; that chain is deferred — see TODO.md.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from sae_muc.pipeline.context import PipelineContext

log = logging.getLogger(__name__)

OUTPUT = "metrics.json"


def _safe_mean(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    return float(values.mean())


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def build_frame_from_paths(
    ctx: PipelineContext,
    *,
    generations_path: str,
    accuracy_path: str,
    judge_path: str,
    se_path: str,
) -> pd.DataFrame:
    """Join the per-variant artefacts into the per-question frame metrics consume."""
    gens = ctx.store.load_parquet(generations_path)
    greedy = gens[gens["kind"] == "greedy"].set_index("sample_id")
    accuracy = ctx.store.load_parquet(accuracy_path).set_index("sample_id")
    judge = ctx.store.load_parquet(judge_path)
    se = ctx.store.load_parquet(se_path).set_index("sample_id")
    vu_per_q = judge[judge["kind"] == "sample"].groupby("sample_id")["vu_score"].mean()
    vu_greedy_per_q = (
        judge[judge["kind"] == "greedy"].set_index("sample_id")["vu_score"]
    )
    refusal_threshold = float(ctx.cfg.stages.detect.refusal_vu_threshold)

    rows: list[dict] = []
    for sid in greedy.index:
        if sid not in se.index or sid not in vu_per_q.index:
            continue
        correct_raw = accuracy.loc[sid, "is_correct"] if sid in accuracy.index else None
        is_correct = bool(correct_raw) if pd.notna(correct_raw) else None
        vu_g = vu_greedy_per_q.get(sid)
        refusal = (vu_g is not None) and pd.notna(vu_g) and (float(vu_g) >= refusal_threshold)
        rows.append(
            {
                "sample_id": sid,
                "vu": float(vu_per_q.loc[sid]),
                "se": float(se.loc[sid, "semantic_entropy"]),
                "is_correct": is_correct,
                "is_refusal": bool(refusal),
            }
        )
    return pd.DataFrame(rows)


def _compute_metrics(df: pd.DataFrame, vu_threshold: float, su_threshold: float | None) -> dict:
    n = len(df)
    if n == 0:
        return {"n_total": 0, "empty": True}

    if su_threshold is None:
        su_threshold = float(df["se"].median())

    refusal_mask = df["is_refusal"]
    labelled = df["is_correct"].notna()
    correct_mask = labelled & (df["is_correct"] == True) & ~refusal_mask  # noqa: E712
    # Hallucinated = labelled as incorrect AND not a refusal. Use an explicit
    # `== False` rather than `~fillna(True)`: an unparsed accuracy answer
    # (NaN) would silently flip to False after `~`, contradicting `labelled`.
    hall_mask = labelled & (df["is_correct"] == False) & ~refusal_mask  # noqa: E712
    confident_mask = hall_mask & (df["vu"] < vu_threshold)

    vu = df["vu"].to_numpy(dtype=float)
    se = df["se"].to_numpy(dtype=float)

    vu_high = df["vu"] > vu_threshold
    su_high = df["se"] > su_threshold
    disagreement = (vu_high != su_high).mean()

    return {
        "n_total": int(n),
        "n_refusal": int(refusal_mask.sum()),
        "n_correct": int(correct_mask.sum()),
        "n_hallucinated": int(hall_mask.sum()),
        "n_confident_hallucinated": int(confident_mask.sum()),
        "hallucination_rate": float(hall_mask.mean()),
        "confident_hallucination_rate": float(confident_mask.mean()),
        "correct_rate": float(correct_mask.mean()),
        "refusal_rate": float(refusal_mask.mean()),
        "vu_su_disagreement_rate": float(disagreement),
        "correlation": _pearson(vu, se),
        "vu_correct_mean": _safe_mean(df.loc[correct_mask, "vu"]),
        "vu_incorrect_mean": _safe_mean(df.loc[hall_mask, "vu"]),
        "thresholds": {"vu": float(vu_threshold), "su": float(su_threshold)},
    }


def _build_frame(ctx: PipelineContext) -> pd.DataFrame:
    return build_frame_from_paths(
        ctx,
        generations_path="generations.parquet",
        accuracy_path="accuracy.parquet",
        judge_path="judge_scores.parquet",
        se_path="semantic_entropy.parquet",
    )


def run(ctx: PipelineContext) -> list[str]:
    df = _build_frame(ctx)
    log.info(
        "computing Tab.3 metrics on %d samples (%d correct, %d refusals)",
        len(df),
        int(df["is_correct"].fillna(False).sum()),
        int(df["is_refusal"].sum()),
    )
    # Default thresholds: VU 0.5 (paper-ish), SU = median (balanced split).
    metrics = _compute_metrics(df, vu_threshold=0.5, su_threshold=None)
    ctx.store.save_json(OUTPUT, metrics)
    return [OUTPUT]
