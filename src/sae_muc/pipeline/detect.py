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
                       (paper §2.3: refusal classification is derived from the
                       judge's own VU on the most-likely answer, not a regex
                       list. The threshold value itself — default 0.85 — is OUR
                       calibration: §2.3 defines abstention behaviourally and
                       pins no specific cut-off.)
  - is_hallucinated  = (not is_correct) AND (not is_refusal)

The trainable set drops refusals and any sample with a missing label. We
do an 80/20 stratified split (seeded via cfg.seed) and fit logistic
regressions per `cfg.stages.detect.detector_method`:
  * lr_vu_se (default): verbal-only, semantic-only, combined (vu, se).
  * lr_hidden          : the three above + hidden-state probe (paper §4.1).
  * combined           : the four above + combined_full (vu, se, hidden).

`cfg.stages.detect.detector_input` selects the verbal/semantic/combined LR
features (paper §4.1 / Table 2):
  * calculated (default): the calculated judge VU + NLI SE.
  * probe_predicted     : VU/SU predicted by two linear regressor probes over
                          the question's last-token hidden state (App F.1 per-
                          uncertainty ranges) — no sampling needed at inference.

`cfg.stages.detect.detector_baselines` additionally fits external Table-2
baselines (each needs hidden states):
  * sep — Semantic Entropy Probe: an LR over the TBG (last-question-token)
          hidden state predicting binarized SU, abstained→non-hallucinated.

EigenScore (Table-2's other baseline) is DEFERRED, not implemented: the paper
cites it but gives no formula — it is the INSIDE method (Chen et al. 2024). It
needs the covariance of the K *sampled-answer* embeddings, which this pipeline
does not store (hidden_states forwards the question, not each sampled answer),
so it would require a fresh forward pass over generations.parquet's sample
texts. Addable any time without data loss; out of scope here, see TODO.md.

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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from sae_muc.pipeline._utils import (
    PROMPT_PLAIN,
    _pool,
    _resolve_layers,
    select_prompt_kind,
)
from sae_muc.pipeline.context import PipelineContext
from sae_muc.pipeline.probe_layer_ranges import probe_layer_range, probe_layer_union

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
    ctx: PipelineContext,
    sample_ids: list[str],
    layers: int | list[int],
    pooling: str | None = None,
) -> np.ndarray:
    """Pool the residual stream at `layers` per sample and concatenate.

    Returns [n, d_model * len(layers)]. The paper's probes source from a
    *range* of layers (App F.1 "sourced from multiple layers"), concatenated
    into one feature vector — a single int keeps the legacy single-layer probe.

    `pooling` defaults to cfg.stages.vuf.pooling (the classifier probe inherits
    the extraction pooling); the App F.1 regressor / SEP probes pass an explicit
    "last_token_q" since the paper pins their token to the last question token
    (TBG), independent of the VUF-extraction pooling knob.
    """
    import torch  # local import — keep top of module light

    layer_list = [int(layers)] if isinstance(layers, int) else [int(l) for l in layers]
    meta = ctx.store.load_parquet("hidden_states/meta.parquet").set_index("sample_id")
    pooling = pooling or ctx.cfg.stages.vuf.pooling
    per_layer: list[np.ndarray] = []
    for layer in layer_list:
        tensors = ctx.store.load_safetensors(f"hidden_states/layer_{layer}.safetensors")
        pooled: list[torch.Tensor] = []
        for sid in sample_ids:
            pooled.append(
                _pool(
                    tensors[sid], pooling,
                    int(meta.loc[sid, "question_len"]),
                    int(meta.loc[sid, "seq_len"]),
                )
            )
        per_layer.append(torch.stack(pooled).float().numpy())
    return np.concatenate(per_layer, axis=1)


def _resolve_detector_layers(
    detector_layer, dataset: str, available: list[int]
) -> list[int]:
    """Resolve detect.detector_layer to a layer list (classifier hidden probe).

    "paper_range" → the App F.1 per-dataset VU∪SU probe range; otherwise int /
    list / "auto" via the shared resolver (auto = single middle layer).
    """
    if detector_layer == "paper_range":
        union = [l for l in probe_layer_union(dataset) if l in available]
        if not union:
            raise ValueError(
                f"detect.detector_layer='paper_range' resolved to "
                f"{probe_layer_union(dataset)} for dataset {dataset!r}, but none "
                f"are in available={available}."
            )
        return union
    return _resolve_layers(detector_layer, available)


def _fit_regressor_probes(
    ctx: PipelineContext,
    df: pd.DataFrame,
    train_pos: np.ndarray,
    available: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Two linear regressor probes (paper §4.1 / Table 2 Probe-Predicted).

    Fit Ridge regressors on the question's last-token hidden state (App F.1
    per-uncertainty ranges) to predict VU and SU, trained on `train_pos` (the
    detector's train split) to avoid leakage. Returns (pred_vu, pred_se) over
    every row of `df`.
    """
    dataset = ctx.cfg.dataset.name
    sids = df["sample_id"].tolist()
    preds: list[np.ndarray] = []
    for uncertainty, target_col in (("vu", "vu"), ("su", "se")):
        layers = [l for l in probe_layer_range(dataset, uncertainty) if l in available]
        if not layers:
            raise ValueError(
                f"detector_input='probe_predicted' but App F.1 {uncertainty} range "
                f"{probe_layer_range(dataset, uncertainty)} has no layers in "
                f"available={available}."
            )
        # last_token_q == TBG: the paper pins the probe input to the question's
        # last token, independent of cfg.stages.vuf.pooling.
        H = _load_pooled_hidden(ctx, sids, layers, pooling="last_token_q")
        y_target = df[target_col].to_numpy(dtype=float)
        ridge = Ridge()
        ridge.fit(H[train_pos], y_target[train_pos])
        preds.append(ridge.predict(H))
    return preds[0], preds[1]


def _fit_sep_baseline(
    ctx: PipelineContext,
    df: pd.DataFrame,
    y: np.ndarray,
    trainable_idx: np.ndarray,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
    available: list[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    """SEP baseline (paper §4.1 Baselines / Kossen 2024).

    A logistic-regression probe over the TBG (token-before-generating =
    last-question-token) hidden state predicting *binarized* SU; the predicted
    P(high-SU) is the hallucination score. Abstained samples are already
    excluded from the trainable population (classified non-hallucinated), per
    the paper's adaptation. Uses the App F.1 SU range. Returns (prob_all,
    metrics) — AUROC/ACC evaluated against the hallucination labels on the same
    train/test split as the proposed detector.

    This reproduces the TBG setting; the paper also reports a sentence-form
    (answer-pooled) SEP, which is not reproduced here (it needs answer-token
    pooling). The hidden-state classifier probe (lr_hidden, App F.2 / Table 7)
    is the closest in-pipeline relative.
    """
    dataset = ctx.cfg.dataset.name
    layers = [l for l in probe_layer_range(dataset, "su") if l in available]
    if not layers:
        raise ValueError(
            f"SEP baseline but App F.1 SU range {probe_layer_range(dataset, 'su')} "
            f"has no layers in available={available}."
        )
    # TBG = last token before generating = last question token.
    H = _load_pooled_hidden(ctx, df["sample_id"].tolist(), layers, pooling="last_token_q")
    train_pos = trainable_idx[idx_train]
    test_pos = trainable_idx[idx_test]
    # Binarize SU at the train-split median (data-derived, no paper constant).
    su_thresh = float(np.median(df["se"].to_numpy(float)[train_pos]))
    y_su = (df["se"].to_numpy(float) > su_thresh).astype(int)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(H[train_pos], y_su[train_pos])
    prob_all = clf.predict_proba(H)[:, 1]
    metrics = {
        "train": _score(y[idx_train], prob_all[train_pos], (prob_all[train_pos] >= 0.5).astype(int)),
        "test": _score(y[idx_test], prob_all[test_pos], (prob_all[test_pos] >= 0.5).astype(int)),
        "n_features": int(H.shape[1]),
        "su_layers": layers,
        "su_binarize_threshold": su_thresh,
    }
    return prob_all, metrics


def run(ctx: PipelineContext) -> list[str]:
    df = _build_feature_frame(ctx)
    detect_cfg = ctx.cfg.stages.detect
    intervene_cfg = ctx.cfg.stages.intervene
    method = detect_cfg.detector_method
    detector_input = detect_cfg.detector_input
    baselines = list(detect_cfg.detector_baselines)
    use_classifier_probe = method in ("lr_hidden", "combined")
    need_hidden = (
        use_classifier_probe
        or detector_input == "probe_predicted"
        or bool(baselines)
    )

    trainable = df[(~df["is_refusal"]) & df["is_hallucinated"].notna()].copy()
    log.info(
        "fitting LR detector on %d trainable (%d refusals excluded, %d hallucinated; "
        "input=%s, method=%s, baselines=%s)",
        len(trainable), int(df["is_refusal"].sum()),
        int(trainable["is_hallucinated"].sum()) if not trainable.empty else 0,
        detector_input, method, baselines or "none",
    )
    metrics: dict[str, Any] = {
        "n_total": int(len(df)),
        "n_refusal": int(df["is_refusal"].sum()),
        "n_trainable": int(len(trainable)),
        "n_hallucinated": int(trainable["is_hallucinated"].sum())
        if not trainable.empty
        else 0,
        "split": {"train_fraction": 0.8, "seed": int(ctx.cfg.seed)},
        "detector_method": method,
        "detector_input": detector_input,
    }

    # Resolve the hidden-state layer set up front (the classifier probe, the
    # regressor probes, and the SEP baseline all read from the same `available`).
    available: list[int] = []
    detector_layers: list[int] = []
    if need_hidden:
        vuf_meta = ctx.store.load_parquet("vuf/meta.parquet")
        available = sorted(int(x) for x in vuf_meta["layer"].tolist())
        if use_classifier_probe:
            detector_layers = _resolve_detector_layers(
                detect_cfg.detector_layer, ctx.cfg.dataset.name, available
            )
            metrics["detector_layers"] = [int(l) for l in detector_layers]

    # Pre-create every prediction column (filled as the stage runs / NaN on skip).
    for name in ("verbal", "semantic", "combined"):
        df[f"prob_hallucinate_{name}"] = _nan_probs(len(df))
    if use_classifier_probe:
        df["prob_hallucinate_hidden"] = _nan_probs(len(df))
    if method == "combined":
        df["prob_hallucinate_combined_full"] = _nan_probs(len(df))
    if "sep" in baselines:
        df["prob_hallucinate_sep"] = _nan_probs(len(df))

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

    sid_to_idx = {sid: i for i, sid in enumerate(df["sample_id"])}
    trainable_idx = np.asarray([sid_to_idx[sid] for sid in trainable["sample_id"]])
    train_pos = trainable_idx[idx_train]

    # Base features: calculated VU/SU, or VU/SU predicted by two regressor
    # probes from the question's last-token hidden state (paper Table 2).
    if detector_input == "probe_predicted":
        pred_vu, pred_se = _fit_regressor_probes(ctx, df, train_pos, available)
        df["pred_vu"] = pred_vu
        df["pred_se"] = pred_se
        feature_specs = (
            ("verbal", ["pred_vu"]),
            ("semantic", ["pred_se"]),
            ("combined", ["pred_vu", "pred_se"]),
        )
    else:
        feature_specs = _BASE_FEATURE_SPEC

    for name, cols in feature_specs:
        X_all = df[cols].to_numpy(dtype=float)
        X = X_all[trainable_idx]
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[idx_train], y[idx_train])
        metrics[name] = {
            "train": _score(
                y[idx_train], clf.predict_proba(X[idx_train])[:, 1], clf.predict(X[idx_train])
            ),
            "test": _score(
                y[idx_test], clf.predict_proba(X[idx_test])[:, 1], clf.predict(X[idx_test])
            ),
            "n_features": len(cols),
        }
        # Predict on every row (trainable + refusals); all features are filled.
        df[f"prob_hallucinate_{name}"] = clf.predict_proba(X_all)[:, 1]

    if use_classifier_probe:
        # Train a hidden-state classifier probe over `detector_layers`, then
        # optionally the (vu, se, hidden) full-combined LR for method=combined.
        hidden_all = _load_pooled_hidden(ctx, df["sample_id"].tolist(), detector_layers)
        H = hidden_all[trainable_idx]

        clf_h = LogisticRegression(max_iter=1000)
        clf_h.fit(H[idx_train], y[idx_train])
        metrics["hidden"] = {
            "train": _score(
                y[idx_train], clf_h.predict_proba(H[idx_train])[:, 1], clf_h.predict(H[idx_train])
            ),
            "test": _score(
                y[idx_test], clf_h.predict_proba(H[idx_test])[:, 1], clf_h.predict(H[idx_test])
            ),
            "n_features": int(hidden_all.shape[1]),
            "layers": [int(l) for l in detector_layers],
        }
        df["prob_hallucinate_hidden"] = clf_h.predict_proba(hidden_all)[:, 1]

        if method == "combined":
            full_all = np.concatenate(
                [df[["vu", "se"]].to_numpy(dtype=float), hidden_all], axis=1
            )
            F = full_all[trainable_idx]
            clf_full = LogisticRegression(max_iter=1000)
            clf_full.fit(F[idx_train], y[idx_train])
            metrics["combined_full"] = {
                "train": _score(
                    y[idx_train], clf_full.predict_proba(F[idx_train])[:, 1], clf_full.predict(F[idx_train])
                ),
                "test": _score(
                    y[idx_test], clf_full.predict_proba(F[idx_test])[:, 1], clf_full.predict(F[idx_test])
                ),
                "n_features": int(full_all.shape[1]),
            }
            df["prob_hallucinate_combined_full"] = clf_full.predict_proba(full_all)[:, 1]

    if "sep" in baselines:
        prob_sep, sep_metrics = _fit_sep_baseline(
            ctx, df, y, trainable_idx, idx_train, idx_test, available
        )
        df["prob_hallucinate_sep"] = prob_sep
        metrics["sep"] = sep_metrics

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
