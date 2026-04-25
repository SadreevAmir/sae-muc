"""detect: LR hallucination detector on (SU, VU) features (paper Tab.1 / Tab.2).

Input artefacts (all produced by earlier stages):
  - generations.parquet  — greedy answer per sample
  - accuracy.parquet     — is_correct from LLM-as-judge
  - judge_scores.parquet — VU per generation (greedy + samples)
  - semantic_entropy.parquet — SE per question

For each question we compute:
  - vu        = mean judge VU over the N high-T samples (§2.2)
  - vu_greedy = judge VU on the greedy answer
  - se        = semantic entropy of the N samples
  - is_refusal       = vu_greedy ≥ cfg.stages.detect.refusal_vu_threshold
                       (paper §3.2: refusal classification is derived from the
                       judge's own VU score, not a regex list. The threshold
                       value itself — default 0.85 — is OUR calibration: the
                       paper does not pin a specific cut-off.)
  - is_hallucinated  = (not is_correct) AND (not is_refusal)

The trainable set drops refusals and any sample with a missing label. We
do an 80/20 stratified split (seeded via cfg.seed) and fit three logistic
regressions: verbal-only, semantic-only, combined. Metrics (AUROC, ACC)
are reported on both train and test splits. Predictions for every sample
(including refusals) are written to detection.parquet.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from sae_muc.pipeline.context import PipelineContext

log = logging.getLogger(__name__)

OUTPUT_PRED = "detection.parquet"
OUTPUT_METRICS = "detection_metrics.json"


def _score(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auroc = float("nan")
    return {"auroc": auroc, "acc": float(accuracy_score(y_true, y_pred))}


def _build_feature_frame(ctx: PipelineContext) -> pd.DataFrame:
    gens = ctx.store.load_parquet("generations.parquet")
    greedy = gens[gens["kind"] == "greedy"].set_index("sample_id")
    accuracy = ctx.store.load_parquet("accuracy.parquet").set_index("sample_id")
    judge = ctx.store.load_parquet("judge_scores.parquet")
    se = ctx.store.load_parquet("semantic_entropy.parquet").set_index("sample_id")

    vu_per_q = judge[judge["kind"] == "sample"].groupby("sample_id")["vu_score"].mean()
    vu_greedy_per_q = (
        judge[judge["kind"] == "greedy"].set_index("sample_id")["vu_score"]
    )
    refusal_threshold = float(ctx.cfg.stages.detect.refusal_vu_threshold)

    rows: list[dict[str, Any]] = []
    for sid in greedy.index:
        if sid not in vu_per_q.index or sid not in se.index:
            continue
        vu_g = vu_greedy_per_q.get(sid)
        refusal = (vu_g is not None) and pd.notna(vu_g) and (float(vu_g) >= refusal_threshold)
        correct_raw = accuracy.loc[sid, "is_correct"] if sid in accuracy.index else None
        is_correct = bool(correct_raw) if pd.notna(correct_raw) else None
        if is_correct is None:
            hall = None
        else:
            hall = (not is_correct) and (not refusal)
        rows.append(
            {
                "sample_id": sid,
                "vu": float(vu_per_q.loc[sid]),
                "vu_greedy": float(vu_g) if (vu_g is not None and pd.notna(vu_g)) else float("nan"),
                "se": float(se.loc[sid, "semantic_entropy"]),
                "is_correct": is_correct,
                "is_refusal": bool(refusal),
                "is_hallucinated": hall,
                "greedy_text": greedy.loc[sid, "text"],
            }
        )
    return pd.DataFrame(rows)


def _split(y: np.ndarray, seed: int, test_size: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    try:
        return train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    except ValueError:
        # Stratification fails when some class has too few members; fall back.
        return train_test_split(idx, test_size=test_size, random_state=seed)


_FEATURE_SPEC: tuple[tuple[str, list[str]], ...] = (
    ("verbal", ["vu"]),
    ("semantic", ["se"]),
    ("combined", ["vu", "se"]),
)


def _nan_probs(n: int) -> np.ndarray:
    return np.full(n, float("nan"))


def run(ctx: PipelineContext) -> list[str]:
    df = _build_feature_frame(ctx)

    trainable = df[(~df["is_refusal"]) & df["is_hallucinated"].notna()].copy()
    log.info(
        "fitting LR detector on %d trainable (%d refusals excluded, %d hallucinated in train set)",
        len(trainable), int(df["is_refusal"].sum()),
        int(trainable["is_hallucinated"].sum()) if not trainable.empty else 0,
    )
    metrics: dict[str, Any] = {
        "n_total": int(len(df)),
        "n_refusal": int(df["is_refusal"].sum()),
        "n_trainable": int(len(trainable)),
        "n_hallucinated": int(trainable["is_hallucinated"].sum())
        if not trainable.empty
        else 0,
        "split": {"train_fraction": 0.8, "seed": int(ctx.cfg.seed)},
    }

    # Default: no predictions, filled as stage runs.
    for name, _ in _FEATURE_SPEC:
        df[f"prob_hallucinate_{name}"] = _nan_probs(len(df))

    if len(trainable) < 4 or trainable["is_hallucinated"].nunique() < 2:
        log.warning(
            "detect: not enough labelled, non-refusal samples to fit a detector "
            "(trainable=%d, classes=%d)",
            len(trainable),
            trainable["is_hallucinated"].nunique() if not trainable.empty else 0,
        )
        metrics["skipped"] = "insufficient data"
        ctx.store.save_parquet(OUTPUT_PRED, df)
        ctx.store.save_json(OUTPUT_METRICS, metrics)
        return [OUTPUT_PRED, OUTPUT_METRICS]

    y = trainable["is_hallucinated"].astype(int).values
    idx_train, idx_test = _split(y, seed=int(ctx.cfg.seed))

    # Even after a stratified split, a tiny imbalanced dataset can leave a
    # fold with a single class — LogisticRegression refuses to fit in that
    # case. Detect that and skip gracefully.
    if len(np.unique(y[idx_train])) < 2:
        log.warning(
            "detect: train split has a single class (train=%d/%d, classes=%s); skipping fit",
            len(idx_train), len(y), sorted(set(y[idx_train].tolist())),
        )
        metrics["skipped"] = "single_class_in_train_split"
        ctx.store.save_parquet(OUTPUT_PRED, df)
        ctx.store.save_json(OUTPUT_METRICS, metrics)
        return [OUTPUT_PRED, OUTPUT_METRICS]

    for name, cols in _FEATURE_SPEC:
        X = trainable[cols].values
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[idx_train], y[idx_train])

        train_prob = clf.predict_proba(X[idx_train])[:, 1]
        train_pred = clf.predict(X[idx_train])
        test_prob = clf.predict_proba(X[idx_test])[:, 1]
        test_pred = clf.predict(X[idx_test])

        metrics[name] = {
            "train": _score(y[idx_train], train_prob, train_pred),
            "test": _score(y[idx_test], test_prob, test_pred),
            "n_features": len(cols),
        }

        # Predict on every row (trainable + refusals); all have vu and se.
        df[f"prob_hallucinate_{name}"] = clf.predict_proba(df[cols].values)[:, 1]

    ctx.store.save_parquet(OUTPUT_PRED, df)
    ctx.store.save_json(OUTPUT_METRICS, metrics)
    return [OUTPUT_PRED, OUTPUT_METRICS]
