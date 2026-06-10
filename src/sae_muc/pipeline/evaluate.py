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

Threshold modes come from `cfg.stages.evaluate`:
  vu_threshold_mode = "kossen" | "fixed" (default kossen, paper-faithful;
                       Kossen et al. 2024 picks t minimising
                       sum((vu - t)**2), which collapses to mean(vu)).
  su_threshold_mode = "kossen" | "fixed" | "median" (default kossen).
  vu_threshold / su_threshold are the explicit values used in fixed mode.

Post-intervention metrics (per α) require re-running judge / semantic_entropy
/ accuracy_judge on the intervened generations; that chain is in evaluate_post.

Refusal classification reuses `cfg.stages.detect.refusal_vu_threshold`
(default 0.85 — OUR calibration, not paper). Numbers under a non-paper
threshold are not directly comparable to Tab.3.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from sae_muc.pipeline._utils import PROMPT_PLAIN, select_prompt_kind
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
    # Plain most-likely answer for the rate metrics (paper App C / §2.3); the
    # eliciting samples drive VU(x). Falls through on the single-set post gens.
    greedy = select_prompt_kind(
        gens[gens["kind"] == "greedy"], PROMPT_PLAIN
    ).set_index("sample_id")
    accuracy = ctx.store.load_parquet(accuracy_path).set_index("sample_id")
    judge = ctx.store.load_parquet(judge_path)
    se = ctx.store.load_parquet(se_path).set_index("sample_id")
    vu_per_q = judge[judge["kind"] == "sample"].groupby("sample_id")["vu_score"].mean()
    vu_greedy_per_q = (
        select_prompt_kind(judge[judge["kind"] == "greedy"], PROMPT_PLAIN)
        .set_index("sample_id")["vu_score"]
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


def _kossen_threshold(values: np.ndarray) -> float:
    """t* = argmin_t sum((value - t)**2). Closed form: t* = mean(value).

    Empty input falls back to 0.0 (the metric is uncomputable; the caller
    will have already returned the empty-frame stub above, but guard here
    too in case someone wires this elsewhere).
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return 0.0
    return float(arr.mean())


def _resolve_thresholds(
    df: pd.DataFrame,
    *,
    vu_mode: str,
    vu_fallback: float,
    su_mode: str,
    su_fallback: float | None,
) -> tuple[float, float]:
    if vu_mode == "kossen":
        vu_t = _kossen_threshold(df["vu"].to_numpy(dtype=float))
    else:  # fixed
        vu_t = float(vu_fallback)
    if su_mode == "kossen":
        su_t = _kossen_threshold(df["se"].to_numpy(dtype=float))
    elif su_mode == "median":
        su_t = float(df["se"].median())
    else:  # fixed
        if su_fallback is None:
            su_t = float(df["se"].median())
        else:
            su_t = float(su_fallback)
    return vu_t, su_t


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


def _abstention_categories(
    ctx: PipelineContext, df: pd.DataFrame, refusal_threshold: float
) -> dict:
    """Behavioural abstained/complying categorization (paper §2.3 / Table 6).

    §2.3: abstained-vs-complying is keyed on the most-likely answer; an
    abstained question is "consistently abstained" if ALL sampled answers
    abstain, else "partly abstained" (complies at least once). Complying
    questions split correct vs hallucinated. Per-sample abstention reuses the
    same VU ≥ refusal_vu_threshold cut (OUR calibration — the paper pins no
    number). Diagnostic only: it does not feed the Table-3 rate metrics.
    """
    judge = ctx.store.load_parquet("judge_scores.parquet")
    samples = judge[judge["kind"] == "sample"]
    n_samples = samples.groupby("sample_id")["vu_score"].size()
    n_abstain = (
        samples.assign(_ab=samples["vu_score"] >= refusal_threshold)
        .groupby("sample_id")["_ab"]
        .sum()
    )
    counts = {
        "consistently_abstained": 0,
        "partly_abstained": 0,
        "complying_correct": 0,
        "complying_hallucinated": 0,
        "complying_unlabelled": 0,
    }
    for _, row in df.iterrows():
        sid = row["sample_id"]
        if bool(row["is_refusal"]):
            ns = int(n_samples.get(sid, 0))
            na = int(n_abstain.get(sid, 0))
            # consistently = every sample abstains; partly = >=1 complies. With
            # no samples (na==ns==0) none complies → consistently (vacuous).
            key = "consistently_abstained" if na == ns else "partly_abstained"
            counts[key] += 1
        elif row["is_correct"] is True:
            counts["complying_correct"] += 1
        elif row["is_correct"] is False:
            counts["complying_hallucinated"] += 1
        else:
            counts["complying_unlabelled"] += 1
    total = sum(counts.values())
    proportions = {k: (v / total if total else float("nan")) for k, v in counts.items()}
    return {
        "counts": counts,
        "proportions": proportions,
        "n": total,
        "refusal_vu_threshold": refusal_threshold,
    }


def run(ctx: PipelineContext) -> list[str]:
    df = _build_frame(ctx)
    log.info(
        "computing Tab.3 metrics on %d samples (%d correct, %d refusals)",
        len(df),
        int(df["is_correct"].fillna(False).sum()),
        int(df["is_refusal"].sum()),
    )
    eval_cfg = ctx.cfg.stages.evaluate
    vu_t, su_t = _resolve_thresholds(
        df,
        vu_mode=eval_cfg.vu_threshold_mode,
        vu_fallback=eval_cfg.vu_threshold,
        su_mode=eval_cfg.su_threshold_mode,
        su_fallback=eval_cfg.su_threshold,
    )
    log.info(
        "thresholds: vu_mode=%s → %.3f, su_mode=%s → %.3f",
        eval_cfg.vu_threshold_mode, vu_t, eval_cfg.su_threshold_mode, su_t,
    )
    metrics = _compute_metrics(df, vu_threshold=vu_t, su_threshold=su_t)
    metrics["abstention_categories"] = _abstention_categories(
        ctx, df, float(ctx.cfg.stages.detect.refusal_vu_threshold)
    )
    ctx.store.save_json(OUTPUT, metrics)
    return [OUTPUT]
