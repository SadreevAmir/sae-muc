"""
Experiment 1: Linear probes SE/VU → LR detector
(Reproduces Table 2 / Table 4 from the paper)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import json, joblib, warnings
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, accuracy_score

import config
from data_utils import load_all

warnings.filterwarnings("ignore")

# ── Load data ─────────────────────────────────────────────────────────────────
data = load_all(
    config.TRAIN_CSV, config.TEST_CSV,
    config.TRAIN_PT_DIR, config.TEST_PT_DIR,
    config.TRAIN_VU_JUDGE, config.TEST_VU_JUDGE,
    config.TRAIN_ACC_JSON, config.TEST_ACC_JSON,
)
hs_tr, hs_te = data["hs_tr"], data["hs_te"]
se_tr, se_te = data["se_tr"], data["se_te"]
vu_tr, vu_te = data["vu_tr"], data["vu_te"]
hal_tr, hal_te = data["hal_tr"], data["hal_te"]

LAYERS = config.LAYERS
results = {}

# ── Per-layer probes ───────────────────────────────────────────────────────────
print("\n── Per-layer Ridge probes ──────────────────────────────────────")
print(f"{'Layer':<8} {'SE MAE':<10} {'SE R²':<10} {'VU MAE':<10} {'VU R²'}")

se_probes, vu_probes = {}, {}
probe_rows = []

for li, layer in enumerate(LAYERS):
    X_tr, X_te = hs_tr[:, li, :], hs_te[:, li, :]

    se_pipe = Pipeline([("sc", StandardScaler()), ("r", Ridge(alpha=1.0))])
    se_pipe.fit(X_tr, se_tr)
    se_pred = se_pipe.predict(X_te)
    se_mae, se_r2 = mean_absolute_error(se_te, se_pred), r2_score(se_te, se_pred)

    vu_pipe = Pipeline([("sc", StandardScaler()), ("r", Ridge(alpha=1.0))])
    vu_pipe.fit(X_tr, vu_tr)
    vu_pred = vu_pipe.predict(X_te)
    vu_mae, vu_r2 = mean_absolute_error(vu_te, vu_pred), r2_score(vu_te, vu_pred)

    se_probes[layer] = se_pipe
    vu_probes[layer] = vu_pipe
    probe_rows.append(dict(layer=layer, se_mae=se_mae, se_r2=se_r2, vu_mae=vu_mae, vu_r2=vu_r2))
    print(f"  {layer:<6} {se_mae:<10.4f} {se_r2:<10.4f} {vu_mae:<10.4f} {vu_r2:.4f}")

best_se_layer = max(probe_rows, key=lambda r: r["se_r2"])["layer"]
best_vu_layer = max(probe_rows, key=lambda r: r["vu_r2"])["layer"]
print(f"\nBest SE probe: layer {best_se_layer}")
print(f"Best VU probe: layer {best_vu_layer}")

# ── Detection: probe-predicted SE+VU → LR ─────────────────────────────────────
print("\n── Detection: probe-predicted SE+VU ──────────────────────────")
print(f"{'Features':<22} {'AUROC':<8} {'ACC'}")
det_rows = []

for li, layer in enumerate(LAYERS):
    se_pred_tr = se_probes[layer].predict(hs_tr[:, li, :])
    vu_pred_tr = vu_probes[layer].predict(hs_tr[:, li, :])
    se_pred_te = se_probes[layer].predict(hs_te[:, li, :])
    vu_pred_te = vu_probes[layer].predict(hs_te[:, li, :])

    for feats_tr, feats_te, tag in [
        (np.c_[se_pred_tr, vu_pred_tr], np.c_[se_pred_te, vu_pred_te], "SE+VU"),
        (se_pred_tr[:, None],           se_pred_te[:, None],           "SE only"),
        (vu_pred_tr[:, None],           vu_pred_te[:, None],           "VU only"),
    ]:
        lr = LogisticRegression(max_iter=1000)
        lr.fit(feats_tr, hal_tr)
        proba = lr.predict_proba(feats_te)[:, 1]
        auroc = roc_auc_score(hal_te, proba)
        acc   = accuracy_score(hal_te, lr.predict(feats_te))
        det_rows.append(dict(layer=layer, features=tag, auroc=auroc, acc=acc))

det_rows.sort(key=lambda r: r["auroc"], reverse=True)
for r in det_rows[:10]:
    print(f"  L{r['layer']} {r['features']:<10} {r['auroc']:.4f}   {r['acc']:.4f}")

# ── Baseline: raw SE/VU → LR ──────────────────────────────────────────────────
print("\n── Baseline: raw SE+VU (calculated) ──────────────────────────")
baseline_rows = []
for feats_tr, feats_te, tag in [
    (np.c_[se_tr, vu_tr], np.c_[se_te, vu_te], "raw SE+VU"),
    (se_tr[:, None], se_te[:, None],            "raw SE only"),
    (vu_tr[:, None], vu_te[:, None],            "raw VU only"),
]:
    lr = LogisticRegression(max_iter=1000)
    lr.fit(feats_tr, hal_tr)
    proba = lr.predict_proba(feats_te)[:, 1]
    auroc = roc_auc_score(hal_te, proba)
    acc   = accuracy_score(hal_te, lr.predict(feats_te))
    baseline_rows.append(dict(features=tag, auroc=auroc, acc=acc))
    print(f"  {tag:<18} AUROC={auroc:.4f}  ACC={acc:.4f}")

# ── Save best models ───────────────────────────────────────────────────────────
best_det = det_rows[0]
best_li  = LAYERS.index(best_det["layer"])
best_se_pred_tr = se_probes[best_det["layer"]].predict(hs_tr[:, best_li, :])
best_vu_pred_tr = vu_probes[best_det["layer"]].predict(hs_tr[:, best_li, :])
best_lr = LogisticRegression(max_iter=1000)
best_lr.fit(np.c_[best_se_pred_tr, best_vu_pred_tr], hal_tr)

os.makedirs(config.MODELS_DIR, exist_ok=True)
joblib.dump(se_probes[best_se_layer], f"{config.MODELS_DIR}/se_probe_l{best_se_layer}.pkl")
joblib.dump(vu_probes[best_vu_layer], f"{config.MODELS_DIR}/vu_probe_l{best_vu_layer}.pkl")
joblib.dump(best_lr,                  f"{config.MODELS_DIR}/lr_detector_probe_l{best_det['layer']}.pkl")

# raw SE+VU detector
lr_raw = LogisticRegression(max_iter=1000)
lr_raw.fit(np.c_[se_tr, vu_tr], hal_tr)
joblib.dump(lr_raw, f"{config.MODELS_DIR}/lr_detector_raw_se_vu.pkl")

print(f"\nSaved models to {config.MODELS_DIR}/")

# ── Save results ───────────────────────────────────────────────────────────────
os.makedirs(config.RESULTS_DIR, exist_ok=True)
output = {
    "experiment": "exp1_probes",
    "probe_per_layer": probe_rows,
    "best_se_probe":   {"layer": best_se_layer, **next(r for r in probe_rows if r["layer"]==best_se_layer)},
    "best_vu_probe":   {"layer": best_vu_layer, **next(r for r in probe_rows if r["layer"]==best_vu_layer)},
    "detection_probe": det_rows,
    "detection_baseline": baseline_rows,
}
with open(f"{config.RESULTS_DIR}/exp1_probes.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved results to {config.RESULTS_DIR}/exp1_probes.json")
