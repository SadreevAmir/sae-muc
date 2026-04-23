#!/usr/bin/env python3
"""
Two-stage pipeline for hallucination detection via linear probes.

  Stage 1 — Regression probes (trained on train split):
      hidden_states  →  predicted VU  (verbal uncertainty)
      hidden_states  →  predicted SU  (semantic entropy)

  Stage 2 — Hallucination detection (evaluated on test split):
      Use predicted VU/SU scores to predict hallucination (acc=0 AND not-refusal).
      Compare AUROC against:
        (a) Oracle: true VU / true SU directly as scores
        (b) Oracle LR:  LR trained on (true VU, true SU)
        (c) Probe:      predicted VU / predicted SU as scores
        (d) Probe LR:   LR trained on (pred VU, pred SU) — main result
        (e) Direct LR:  LR trained directly on hidden states

  Paper baseline (Meta's LogisticRegression.py, old checkpoint):
      SU alone: AUROC ≈ 73.9
      VU alone: AUROC ≈ 67.0
      SU + VU:  AUROC ≈ 75.4

Data:
  vuf_checkpoint/datasets/nq_open/Mistral-7B-Instruct-v0.3/
    {split}.csv                                 — VU, SU, question/answer
    pipeline_used_layers_last/{split}/{id}.pt   — hidden states (layers 15-32)
  vuf_checkpoint/sem_uncertainty/sentence/
    {split}_accuracy.json                       — accuracy label per question
    {split}_refusal_rate.json                   — refusal flag per question

Usage:
  python train_uncertainty_probes.py
  python train_uncertainty_probes.py --layer_agg last
  python train_uncertainty_probes.py --skip_layer_analysis   # skip per-layer table

Outputs → vuf_checkpoint/detection/uncertainty_probes/
  probe_vu.pkl, probe_su.pkl   — regression probes
  metrics.json                 — all AUROC / R² results
  layer_analysis.json          — per-layer R² (if not skipped)
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import LinAlgWarning
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Hidden-state matrices are highly collinear → Ridge's least-squares sometimes
# hits near-singular sub-problems.  Regularisation handles it; warning is noise.
warnings.filterwarnings("ignore", category=LinAlgWarning)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CHECKPOINT = Path(__file__).parent / "vuf_checkpoint"
MODEL_NAME = "Mistral-7B-Instruct-v0.3"
DATASET    = "nq_open"
EXTRACTED_LAYERS = list(range(15, 33))   # 18 layers

RIDGE_ALPHAS = np.logspace(-3, 5, 40)
LR_CS        = np.logspace(-4, 4, 20)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_labels(checkpoint_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ids_sorted, vu, su) arrays aligned by sorted question id."""
    import csv
    csv_path = checkpoint_dir / "datasets" / DATASET / MODEL_NAME / f"{split}.csv"
    rows = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows[row["id"]] = row
    ids = sorted(rows, key=lambda x: int(x.split("_")[1]))
    vu  = np.array([float(rows[i]["verbal_uncertainty"])       for i in ids], dtype=np.float32)
    su  = np.array([float(rows[i]["sentence_semantic_entropy"]) for i in ids], dtype=np.float32)
    return ids, vu, su


def load_hallucination_labels(checkpoint_dir: Path, split: str, ids: list[str]) -> np.ndarray:
    """
    Binary label: hallucinated = accuracy==0 AND NOT refusal.
    Returns int array aligned with `ids`.
    """
    su_dir = checkpoint_dir / "sem_uncertainty" / "sentence"

    acc_raw  = json.load(open(su_dir / f"{split}_accuracy.json"))
    ref_data = json.load(open(su_dir / f"{split}_refusal_rate.json"))
    # refusal is a list in dataset order (same as ids sorted numerically)
    refusal  = ref_data["refusal"]

    labels = np.zeros(len(ids), dtype=int)
    for i, sid in enumerate(ids):
        acc      = acc_raw.get(sid, 1.0)   # default: correct if missing
        is_refusal = refusal[i] if i < len(refusal) else False
        labels[i]  = int(acc == 0.0 and not is_refusal)
    return labels


def _load_one(pt_path: Path, layer_agg: str) -> np.ndarray:
    raw = torch.load(pt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        h = torch.stack([raw[k] for k in sorted(raw.keys())], dim=0).float().numpy()
    else:
        h = raw.float().numpy()

    if   layer_agg == "mean":    return h.mean(axis=0)
    elif layer_agg == "last":    return h[-1]
    elif layer_agg == "concat":  return h.reshape(-1)
    elif layer_agg.startswith("layer_"):
        return h[int(layer_agg.split("_")[1])]
    raise ValueError(f"Unknown layer_agg: {layer_agg!r}")


def load_hidden_states(
    checkpoint_dir: Path, split: str, ids: list[str], layer_agg: str
) -> np.ndarray:
    hs_dir = (checkpoint_dir / "datasets" / DATASET / MODEL_NAME
              / "pipeline_used_layers_last" / split)
    out, missing = [], []
    for sid in ids:
        p = hs_dir / f"{sid}.pt"
        if p.exists():
            out.append(_load_one(p, layer_agg))
        else:
            missing.append(sid)
    if missing:
        print(f"  [WARNING] {len(missing)} hidden-state files missing for split={split}")
    return np.array(out, dtype=np.float32)


def peek_format(checkpoint_dir: Path, first_id: str) -> list:
    hs_dir = (checkpoint_dir / "datasets" / DATASET / MODEL_NAME
              / "pipeline_used_layers_last" / "train")
    raw  = torch.load(hs_dir / f"{first_id}.pt", map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        keys = sorted(raw.keys())
        print(f"  format : dict  {len(keys)} layers "
              f"(keys {keys[0]}…{keys[-1]}), each {tuple(raw[keys[0]].shape)}")
        return keys
    print(f"  format : tensor {tuple(raw.shape)}")
    return EXTRACTED_LAYERS


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — Regression probes
# ──────────────────────────────────────────────────────────────────────────────

def train_ridge(X_train: np.ndarray, y_train: np.ndarray, name: str) -> Pipeline:
    pipe = Pipeline([("sc", StandardScaler()), ("ridge", RidgeCV(alphas=RIDGE_ALPHAS, cv=5))])
    pipe.fit(X_train, y_train)
    alpha = pipe.named_steps["ridge"].alpha_
    y_pred = pipe.predict(X_train)
    r2_tr = float(pipe.score(X_train, y_train))
    print(f"  {name}: α={alpha:.3g}  train R²={r2_tr:.3f}")
    return pipe


def eval_regression(pipe: Pipeline, X: np.ndarray, y: np.ndarray, name: str) -> dict:
    y_pred = pipe.predict(X)
    r2   = float(pipe.score(X, y))
    r, p = pearsonr(y, y_pred)
    print(f"  {name}: R²={r2:+.4f}  r={float(r):+.4f}  (p={float(p):.1e})")
    return {"r2": r2, "pearson_r": float(r), "pearson_pval": float(p)}


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — Hallucination detection
# ──────────────────────────────────────────────────────────────────────────────

def auroc(y_true: np.ndarray, scores: np.ndarray, name: str, flip: bool = False) -> float:
    """Compute and print AUROC. flip=True → negate scores before computing."""
    s = -scores if flip else scores
    auc = float(roc_auc_score(y_true, s))
    print(f"  {name}: AUROC = {auc*100:.2f}")
    return auc


def train_lr_detector(
    X_train: np.ndarray, y_train: np.ndarray
) -> Pipeline:
    """Logistic regression detector (2-D or full feature input)."""
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegressionCV(
            Cs=LR_CS, cv=5, max_iter=2000,
            class_weight="balanced", scoring="roc_auc", n_jobs=-1,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def eval_lr_detector(
    pipe: Pipeline, X_test: np.ndarray, y_test: np.ndarray, name: str
) -> float:
    scores = pipe.predict_proba(X_test)[:, 1]
    return auroc(y_test, scores, name)


# ──────────────────────────────────────────────────────────────────────────────
# Per-layer R² analysis
# ──────────────────────────────────────────────────────────────────────────────

def layer_analysis(
    checkpoint_dir: Path,
    train_ids: list[str], test_ids: list[str],
    y_tr_vu: np.ndarray, y_te_vu: np.ndarray,
    y_tr_su: np.ndarray, y_te_su: np.ndarray,
    actual_layers: list,
) -> dict:
    print(f"\n  {'Layer':>6}  {'VU R²':>8}  {'SU R²':>8}")
    results = {}
    for i, layer in enumerate(actual_layers):
        agg  = f"layer_{i}"
        X_tr = load_hidden_states(checkpoint_dir, "train", train_ids, agg)
        X_te = load_hidden_states(checkpoint_dir, "test",  test_ids,  agg)
        sc   = StandardScaler().fit(X_tr)
        Xtr, Xte = sc.transform(X_tr), sc.transform(X_te)
        r2_vu = RidgeCV(alphas=RIDGE_ALPHAS, cv=5).fit(Xtr, y_tr_vu).score(Xte, y_te_vu)
        r2_su = RidgeCV(alphas=RIDGE_ALPHAS, cv=5).fit(Xtr, y_tr_su).score(Xte, y_te_su)
        print(f"  {layer:>6}  {r2_vu:>8.4f}  {r2_su:>8.4f}")
        results[str(layer)] = {"vu_r2": float(r2_vu), "su_r2": float(r2_su)}
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint_dir", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--layer_agg", default="mean",
        choices=["mean", "last", "concat"] + [f"layer_{i}" for i in range(18)],
    )
    parser.add_argument("--skip_layer_analysis", action="store_true")
    args = parser.parse_args()

    out_dir = args.checkpoint_dir / "detection" / "uncertainty_probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict = {"layer_agg": args.layer_agg}

    # ── Labels ────────────────────────────────────────────────────────────────
    print("=" * 65)
    print("Loading labels …")
    tr_ids, y_tr_vu, y_tr_su = load_labels(args.checkpoint_dir, "train")
    te_ids, y_te_vu, y_te_su = load_labels(args.checkpoint_dir, "test")
    y_tr_hallu = load_hallucination_labels(args.checkpoint_dir, "train", tr_ids)
    y_te_hallu = load_hallucination_labels(args.checkpoint_dir, "test",  te_ids)
    print(f"  train: {len(tr_ids)} examples  |  test: {len(te_ids)} examples")
    print(f"  hallucinated — train: {y_tr_hallu.mean():.1%}  test: {y_te_hallu.mean():.1%}")
    print(f"  VU  μ/σ  train: {y_tr_vu.mean():.3f}/{y_tr_vu.std():.3f}   "
          f"test: {y_te_vu.mean():.3f}/{y_te_vu.std():.3f}")
    print(f"  SU  μ/σ  train: {y_tr_su.mean():.3f}/{y_tr_su.std():.3f}   "
          f"test: {y_te_su.mean():.3f}/{y_te_su.std():.3f}")

    # ── Hidden states ─────────────────────────────────────────────────────────
    print(f"\nLoading hidden states (layer_agg={args.layer_agg!r}) …")
    actual_layers = peek_format(args.checkpoint_dir, tr_ids[0])
    X_train = load_hidden_states(args.checkpoint_dir, "train", tr_ids, args.layer_agg)
    X_test  = load_hidden_states(args.checkpoint_dir, "test",  te_ids, args.layer_agg)
    print(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1 — Regression probes: hidden states → VU, SU
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 65)
    print("STAGE 1 — Regression probes (train fit)")
    probe_vu = train_ridge(X_train, y_tr_vu, "VU-Ridge")
    probe_su = train_ridge(X_train, y_tr_su, "SU-Ridge")

    print("\nTest evaluation:")
    metrics["vu_regression"] = eval_regression(probe_vu, X_test, y_te_vu, "VU")
    metrics["su_regression"] = eval_regression(probe_su, X_test, y_te_su, "SU")

    pred_tr_vu = probe_vu.predict(X_train).astype(np.float32)
    pred_te_vu = probe_vu.predict(X_test).astype(np.float32)
    pred_tr_su = probe_su.predict(X_train).astype(np.float32)
    pred_te_su = probe_su.predict(X_test).astype(np.float32)

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2 — Hallucination detection
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 65)
    print("STAGE 2 — Hallucination detection (test AUROC × 100)")

    print("\n  [Oracle — using TRUE VU/SU as scores]")
    # High SU → hallucination; Low VU (confident) → hallucination
    auc_su_oracle  = auroc(y_te_hallu, y_te_su,       "  Oracle SU")
    auc_vu_oracle  = auroc(y_te_hallu, y_te_vu, flip=True, name="  Oracle 1-VU")
    # LR trained on true (VU, SU)
    lr_oracle = train_lr_detector(
        np.stack([y_tr_vu, y_tr_su], axis=1), y_tr_hallu
    )
    auc_lr_oracle  = eval_lr_detector(
        lr_oracle, np.stack([y_te_vu, y_te_su], axis=1), y_te_hallu,
        "  Oracle LR(VU, SU)"
    )

    print("\n  [Probe — using PREDICTED VU/SU as scores]")
    auc_su_probe   = auroc(y_te_hallu, pred_te_su,       "  Probe SU")
    auc_vu_probe   = auroc(y_te_hallu, pred_te_vu, flip=True, name="  Probe 1-VU")
    # LR trained on predicted (VU, SU)  — *** main result ***
    lr_probe = train_lr_detector(
        np.stack([pred_tr_vu, pred_tr_su], axis=1), y_tr_hallu
    )
    auc_lr_probe   = eval_lr_detector(
        lr_probe, np.stack([pred_te_vu, pred_te_su], axis=1), y_te_hallu,
        "  Probe LR(pred_VU, pred_SU)"
    )

    print("\n  [Direct LR on hidden states — upper bound]")
    lr_direct = train_lr_detector(X_train, y_tr_hallu)
    auc_lr_direct  = eval_lr_detector(lr_direct, X_test, y_te_hallu, "  Direct LR(hidden)")

    print("\n  [Paper baseline — Meta LR.py, old checkpoint (Mistral judge)]")
    paper_dir = args.checkpoint_dir / "detection" / "LR_outputs" / DATASET / MODEL_NAME
    for fname, label in [
        ("test_sentence_semantic_entropy.json",             "  Paper SU"),
        ("test_verbal_uncertainty.json",                    "  Paper VU"),
        ("test_verbal_uncertainty_sentence_semantic_entropy.json", "  Paper LR(VU,SU)"),
    ]:
        fp = paper_dir / fname
        if fp.exists():
            d = json.load(open(fp))
            print(f"  {label}: AUROC = {d['auroc']:.2f}  (old checkpoint, Mistral judge)")

    # ── Per-layer analysis ─────────────────────────────────────────────────────
    layer_results = None
    if not args.skip_layer_analysis:
        print("\n" + "─" * 65)
        print("Per-layer R² analysis (single layer → Ridge, test set)")
        layer_results = layer_analysis(
            args.checkpoint_dir,
            tr_ids, te_ids,
            y_tr_vu, y_te_vu,
            y_tr_su, y_te_su,
            actual_layers=actual_layers,
        )
    else:
        print("\n[per-layer analysis skipped]")

    # ── Save ──────────────────────────────────────────────────────────────────
    metrics["detection"] = {
        "oracle_su":          auc_su_oracle,
        "oracle_1_minus_vu":  auc_vu_oracle,
        "oracle_lr_vu_su":    auc_lr_oracle,
        "probe_su":           auc_su_probe,
        "probe_1_minus_vu":   auc_vu_probe,
        "probe_lr_vu_su":     auc_lr_probe,   # main result
        "direct_lr_hidden":   auc_lr_direct,
    }

    with open(out_dir / "probe_vu.pkl", "wb") as f: pickle.dump(probe_vu, f)
    with open(out_dir / "probe_su.pkl", "wb") as f: pickle.dump(probe_su, f)
    with open(out_dir / "lr_probe_detector.pkl", "wb") as f: pickle.dump(lr_probe, f)
    with open(out_dir / "lr_direct_detector.pkl", "wb") as f: pickle.dump(lr_direct, f)
    with open(out_dir / "metrics.json", "w") as f: json.dump(metrics, f, indent=2)
    if layer_results is not None:
        with open(out_dir / "layer_analysis.json", "w") as f:
            json.dump({"extracted_layers": actual_layers, "results": layer_results}, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SUMMARY")
    print(f"  Stage 1 regression")
    print(f"    VU  R²={metrics['vu_regression']['r2']:+.3f}   r={metrics['vu_regression']['pearson_r']:+.3f}")
    print(f"    SU  R²={metrics['su_regression']['r2']:+.3f}   r={metrics['su_regression']['pearson_r']:+.3f}")
    print(f"  Stage 2 AUROC ×100 (test)")
    print(f"    Oracle  SU         {auc_su_oracle*100:5.2f}")
    print(f"    Oracle  1-VU       {auc_vu_oracle*100:5.2f}")
    print(f"    Oracle  LR(VU,SU)  {auc_lr_oracle*100:5.2f}")
    print(f"    --------------------------------")
    print(f"    Probe   SU         {auc_su_probe*100:5.2f}")
    print(f"    Probe   1-VU       {auc_vu_probe*100:5.2f}")
    print(f"    Probe   LR(VU,SU)  {auc_lr_probe*100:5.2f}  ← main result")
    print(f"    Direct  LR(hidden) {auc_lr_direct*100:5.2f}  ← upper bound")
    print("=" * 65)


if __name__ == "__main__":
    main()
