"""detect: LR hallucination detector on (SU, VU) features (paper Tab.1 / Tab.2).

Input artefacts (all produced by earlier stages):
  - generations.parquet  — greedy answer per sample
  - accuracy.parquet     — is_correct from LLM-as-judge
  - judge_scores.parquet — VU per generation (greedy + samples)
  - semantic_entropy.parquet — SE per question
  - hidden_states/layer_{detector_layer}.safetensors  (only when
    detector_method ∈ {"lr_hidden", "combined"})

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
do an 80/20 stratified split (seeded via cfg.seed) and fit logistic
regressions per `cfg.stages.detect.detector_method`:
  * lr_vu_se (default): verbal-only, semantic-only, combined (vu, se).
  * lr_hidden          : the three above + hidden-state probe (paper §4.1).
  * combined           : the four above + combined_full (vu, se, hidden).

Metrics (AUROC, ACC) are reported on both train and test splits.
Predictions for every sample (including refusals) and the bool column
`is_at_risk` (1 ⇔ probability under the active gating method ≥
intervene.detector_threshold) are written to detection.parquet.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from sae_muc.pipeline._utils import (
    PROMPT_PLAIN,
    _pool,
    _resolve_layers,
    select_prompt_kind,
)
from sae_muc.pipeline.context import PipelineContext

log = logging.getLogger(__name__)

OUTPUT_PRED = "detection.parquet"
OUTPUT_METRICS = "detection_metrics.json"


# Maps detector_method → which detection column drives is_at_risk by default.
_GATE_DEFAULTS: dict[str, str] = {
    "lr_vu_se": "combined",
    "lr_hidden": "hidden",
    "combined": "combined_full",
}


def _score(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auroc = float("nan")
    return {"auroc": auroc, "acc": float(accuracy_score(y_true, y_pred))}


def _build_feature_frame(ctx: PipelineContext) -> pd.DataFrame:
    gens = ctx.store.load_parquet("generations.parquet")
    # The most-likely answer (accuracy + abstention signal) is the plain greedy
    # (paper App C / §2.3); VU(x) averages the eliciting samples (§2.2).
    greedy = select_prompt_kind(
        gens[gens["kind"] == "greedy"], PROMPT_PLAIN
    ).set_index("sample_id")
    accuracy = ctx.store.load_parquet("accuracy.parquet").set_index("sample_id")
    judge = ctx.store.load_parquet("judge_scores.parquet")
    se = ctx.store.load_parquet("semantic_entropy.parquet").set_index("sample_id")

    vu_per_q = judge[judge["kind"] == "sample"].groupby("sample_id")["vu_score"].mean()
    vu_greedy_per_q = (
        select_prompt_kind(judge[judge["kind"] == "greedy"], PROMPT_PLAIN)
        .set_index("sample_id")["vu_score"]
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


_BASE_FEATURE_SPEC: tuple[tuple[str, list[str]], ...] = (
    ("verbal", ["vu"]),
    ("semantic", ["se"]),
    ("combined", ["vu", "se"]),
)


def _nan_probs(n: int) -> np.ndarray:
    return np.full(n, float("nan"))


def _load_pooled_hidden(
    ctx: PipelineContext, sample_ids: list[str], layer: int,
) -> np.ndarray:
    """Pool the residual stream at `layer` per sample, returning [n, d_model]."""
    import torch  # local import — keep top of module light

    tensors = ctx.store.load_safetensors(f"hidden_states/layer_{layer}.safetensors")
    meta = ctx.store.load_parquet("hidden_states/meta.parquet").set_index("sample_id")
    pooling = ctx.cfg.stages.vuf.pooling
    pooled: list[torch.Tensor] = []
    for sid in sample_ids:
        pooled.append(
            _pool(
                tensors[sid], pooling,
                int(meta.loc[sid, "question_len"]),
                int(meta.loc[sid, "seq_len"]),
            )
        )
    return torch.stack(pooled).float().numpy()


def run(ctx: PipelineContext) -> list[str]:
    df = _build_feature_frame(ctx)
    detect_cfg = ctx.cfg.stages.detect
    intervene_cfg = ctx.cfg.stages.intervene
    method = detect_cfg.detector_method
    use_hidden = method in ("lr_hidden", "combined")

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

    # Resolve detector_layer up front so the lr_hidden / combined branches
    # below can pull the matching pooled tensor.
    if use_hidden:
        vuf_meta = ctx.store.load_parquet("vuf/meta.parquet")
        available = sorted(int(x) for x in vuf_meta["layer"].tolist())
        detector_layer = _resolve_layers(detect_cfg.detector_layer, available)[0]
        metrics["detector_layer"] = int(detector_layer)
    else:
        detector_layer = None

    # Default: no predictions, filled as stage runs.
    feature_specs: list[tuple[str, list[str]]] = list(_BASE_FEATURE_SPEC)
    for name, _ in feature_specs:
        df[f"prob_hallucinate_{name}"] = _nan_probs(len(df))
    if use_hidden:
        df["prob_hallucinate_hidden"] = _nan_probs(len(df))
    if method == "combined":
        df["prob_hallucinate_combined_full"] = _nan_probs(len(df))

    metrics["detector_method"] = method

    if len(trainable) < 4 or trainable["is_hallucinated"].nunique() < 2:
        log.warning(
            "detect: not enough labelled, non-refusal samples to fit a detector "
            "(trainable=%d, classes=%d)",
            len(trainable),
            trainable["is_hallucinated"].nunique() if not trainable.empty else 0,
        )
        metrics["skipped"] = "insufficient data"
        df["is_at_risk"] = False
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
        df["is_at_risk"] = False
        ctx.store.save_parquet(OUTPUT_PRED, df)
        ctx.store.save_json(OUTPUT_METRICS, metrics)
        return [OUTPUT_PRED, OUTPUT_METRICS]

    for name, cols in feature_specs:
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

    if use_hidden:
        # Train a hidden-state probe on `detector_layer`, then optionally
        # the (vu, se, hidden) full-combined LR for detector_method=combined.
        hidden_all = _load_pooled_hidden(
            ctx, df["sample_id"].tolist(), detector_layer,
        )
        sid_to_idx = {sid: i for i, sid in enumerate(df["sample_id"])}
        trainable_idx = np.asarray([sid_to_idx[sid] for sid in trainable["sample_id"]])
        H = hidden_all[trainable_idx]

        clf_h = LogisticRegression(max_iter=1000)
        clf_h.fit(H[idx_train], y[idx_train])
        metrics["hidden"] = {
            "train": _score(
                y[idx_train],
                clf_h.predict_proba(H[idx_train])[:, 1],
                clf_h.predict(H[idx_train]),
            ),
            "test": _score(
                y[idx_test],
                clf_h.predict_proba(H[idx_test])[:, 1],
                clf_h.predict(H[idx_test]),
            ),
            "n_features": int(hidden_all.shape[1]),
            "layer": int(detector_layer),
        }
        df["prob_hallucinate_hidden"] = clf_h.predict_proba(hidden_all)[:, 1]

        if method == "combined":
            full_all = np.concatenate(
                [df[["vu", "se"]].values, hidden_all], axis=1
            )
            F = full_all[trainable_idx]
            clf_full = LogisticRegression(max_iter=1000)
            clf_full.fit(F[idx_train], y[idx_train])
            metrics["combined_full"] = {
                "train": _score(
                    y[idx_train],
                    clf_full.predict_proba(F[idx_train])[:, 1],
                    clf_full.predict(F[idx_train]),
                ),
                "test": _score(
                    y[idx_test],
                    clf_full.predict_proba(F[idx_test])[:, 1],
                    clf_full.predict(F[idx_test]),
                ),
                "n_features": int(full_all.shape[1]),
            }
            df["prob_hallucinate_combined_full"] = clf_full.predict_proba(full_all)[:, 1]

    # Decide which probability column drives the gate.
    gate_method = intervene_cfg.gate_detector_method
    if gate_method == "auto":
        gate_col = f"prob_hallucinate_{_GATE_DEFAULTS[method]}"
    else:
        gate_col = f"prob_hallucinate_{gate_method}"
    if gate_col not in df.columns:
        log.warning(
            "detect: gate_detector_method=%s -> column %s not produced under "
            "detector_method=%s; falling back to %s",
            gate_method, gate_col, method, f"prob_hallucinate_{_GATE_DEFAULTS[method]}",
        )
        gate_col = f"prob_hallucinate_{_GATE_DEFAULTS[method]}"
    threshold = float(intervene_cfg.detector_threshold)
    df["is_at_risk"] = df[gate_col].fillna(0.0) >= threshold
    metrics["gate"] = {
        "method": gate_col.removeprefix("prob_hallucinate_"),
        "threshold": threshold,
        "n_at_risk": int(df["is_at_risk"].sum()),
    }

    ctx.store.save_parquet(OUTPUT_PRED, df)
    ctx.store.save_json(OUTPUT_METRICS, metrics)
    return [OUTPUT_PRED, OUTPUT_METRICS]
