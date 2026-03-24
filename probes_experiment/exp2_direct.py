"""
Experiment 2: Direct LR / PCA+LR on hidden states (no intermediate SE/VU).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import json, joblib, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score

import config
from data_utils import load_all

warnings.filterwarnings("ignore")

data = load_all(
    config.TRAIN_CSV, config.TEST_CSV,
    config.TRAIN_PT_DIR, config.TEST_PT_DIR,
    config.TRAIN_VU_JUDGE, config.TEST_VU_JUDGE,
    config.TRAIN_ACC_JSON, config.TEST_ACC_JSON,
)
hs_tr, hs_te = data["hs_tr"], data["hs_te"]
hal_tr, hal_te = data["hal_tr"], data["hal_te"]
LAYERS = config.LAYERS

results = []

# ── Per-layer direct LR ────────────────────────────────────────────────────────
print("\n── Per-layer direct LR ─────────────────────────────────────────")
for li, layer in enumerate(LAYERS):
    pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
    pipe.fit(hs_tr[:, li, :], hal_tr)
    proba = pipe.predict_proba(hs_te[:, li, :])[:, 1]
    auroc = roc_auc_score(hal_te, proba)
    acc   = accuracy_score(hal_te, pipe.predict(hs_te[:, li, :]))
    results.append(dict(method=f"direct_L{layer}", auroc=auroc, acc=acc))
    print(f"  L{layer:<4} AUROC={auroc:.4f}  ACC={acc:.4f}")

# ── Mean across layers ─────────────────────────────────────────────────────────
X_tr = hs_tr.mean(axis=1)
X_te = hs_te.mean(axis=1)
pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
pipe.fit(X_tr, hal_tr)
proba = pipe.predict_proba(X_te)[:, 1]
auroc = roc_auc_score(hal_te, proba)
acc   = accuracy_score(hal_te, pipe.predict(X_te))
results.append(dict(method="mean_layers", auroc=auroc, acc=acc))
print(f"\n  mean_layers   AUROC={auroc:.4f}  ACC={acc:.4f}")

# ── PCA + LR (various n_components) ───────────────────────────────────────────
print("\n── PCA + LR on best single layer ──────────────────────────────")
best_direct_layer = sorted(results, key=lambda r: r["auroc"], reverse=True)[0]["method"]
best_li = int(best_direct_layer.split("_L")[1]) - LAYERS[0]

X_tr_l = hs_tr[:, best_li, :]
X_te_l = hs_te[:, best_li, :]

for n_comp in [8, 16, 32, 64, 128]:
    pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("pca", PCA(n_components=n_comp, random_state=42)),
        ("lr",  LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(X_tr_l, hal_tr)
    proba = pipe.predict_proba(X_te_l)[:, 1]
    auroc = roc_auc_score(hal_te, proba)
    acc   = accuracy_score(hal_te, pipe.predict(X_te_l))
    results.append(dict(method=f"PCA{n_comp}+LR_L{LAYERS[best_li]}", auroc=auroc, acc=acc))
    print(f"  PCA(n={n_comp:<4}) L{LAYERS[best_li]}  AUROC={auroc:.4f}  ACC={acc:.4f}")

# ── PCA on all layers concat ───────────────────────────────────────────────────
print("\n── PCA + LR on all layers concat ───────────────────────────────")
X_tr_all = hs_tr.reshape(config.N_TRAIN, -1)
X_te_all = hs_te.reshape(config.N_TEST,  -1)
for n_comp in [32, 64, 128]:
    pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("pca", PCA(n_components=n_comp, random_state=42)),
        ("lr",  LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(X_tr_all, hal_tr)
    proba = pipe.predict_proba(X_te_all)[:, 1]
    auroc = roc_auc_score(hal_te, proba)
    acc   = accuracy_score(hal_te, pipe.predict(X_te_all))
    results.append(dict(method=f"PCA{n_comp}+LR_all_layers", auroc=auroc, acc=acc))
    print(f"  PCA(n={n_comp:<4}) all layers  AUROC={auroc:.4f}  ACC={acc:.4f}")

# ── Save ───────────────────────────────────────────────────────────────────────
results.sort(key=lambda r: r["auroc"], reverse=True)
print("\n── Top results ─────────────────────────────────────────────────")
for r in results[:5]:
    print(f"  {r['method']:<30} AUROC={r['auroc']:.4f}  ACC={r['acc']:.4f}")

# Save best model
best = results[0]
best_li_idx = int(best["method"].split("_L")[-1]) - LAYERS[0] if "_L" in best["method"] else 0
if "PCA" in best["method"] and "all" not in best["method"]:
    n_comp = int(best["method"].split("PCA")[1].split("+")[0])
    pipe_best = Pipeline([("sc", StandardScaler()), ("pca", PCA(n_components=n_comp, random_state=42)), ("lr", LogisticRegression(max_iter=1000))])
    pipe_best.fit(hs_tr[:, best_li_idx, :], hal_tr)
elif "direct" in best["method"]:
    pipe_best = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
    pipe_best.fit(hs_tr[:, best_li_idx, :], hal_tr)
else:
    pipe_best = None

os.makedirs(config.MODELS_DIR, exist_ok=True)
if pipe_best:
    joblib.dump(pipe_best, f"{config.MODELS_DIR}/direct_{best['method']}.pkl")

os.makedirs(config.RESULTS_DIR, exist_ok=True)
with open(f"{config.RESULTS_DIR}/exp2_direct.json", "w") as f:
    json.dump({"experiment": "exp2_direct", "results": results}, f, indent=2)
print(f"\nSaved to {config.RESULTS_DIR}/exp2_direct.json")
