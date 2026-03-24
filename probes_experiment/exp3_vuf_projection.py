"""
Experiment 3: VUF Projection approach.

Key idea (motivated by the paper §3):
  The paper shows that verbal uncertainty is governed by a single linear direction r_VU(l).
  Instead of predicting SE/VU numerically, we project each layer's hidden state onto the
  VUF direction and use those projections as features.

  score(x, l) = h(l)(x) · r_VU(l)          # scalar per layer

  We then train LR on these 17 projections → hallucination.
  This is interpretable: it directly measures how "certain" the model is per layer.

Variants:
  A. VUF projection only (17 scalars) → LR
  B. VUF projection per layer → LR (best single layer)
  C. VUF projection + cosine similarity → LR
  D. MLP (2 hidden layers) on VUF projections

Also included: small MLP directly on hidden states of best layer (with dropout).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import json, joblib, warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score

import config
from data_utils import load_all

warnings.filterwarnings("ignore")

# ── Load data ──────────────────────────────────────────────────────────────────
data = load_all(
    config.TRAIN_CSV, config.TEST_CSV,
    config.TRAIN_PT_DIR, config.TEST_PT_DIR,
    config.TRAIN_VU_JUDGE, config.TEST_VU_JUDGE,
    config.TRAIN_ACC_JSON, config.TEST_ACC_JSON,
)
hs_tr, hs_te = data["hs_tr"], data["hs_te"]
hal_tr, hal_te = data["hal_tr"], data["hal_te"]
LAYERS = config.LAYERS  # [15..31]

# ── Load VUF vectors ──────────────────────────────────────────────────────────
print("\nLoading VUF vectors...")
vuf_all = torch.load(config.VUF_MERGED, map_location="cpu", weights_only=True).float().numpy()
# shape [32, 4096] — one direction per layer
# Extract only layers 15-31
vuf = vuf_all[LAYERS, :]   # [17, 4096]
# L2-normalize each layer's direction
vuf_norm = vuf / (np.linalg.norm(vuf, axis=1, keepdims=True) + 1e-8)  # [17, 4096]
print(f"VUF shape: {vuf_norm.shape}")

# ── A. VUF projections (all layers) → LR ──────────────────────────────────────
# proj[i, l] = h(i, l) · r_VU(l)
def compute_projections(hs: np.ndarray) -> np.ndarray:
    """hs: [n, 17, 4096] → projections: [n, 17]"""
    return np.einsum("nld,ld->nl", hs, vuf_norm)

proj_tr = compute_projections(hs_tr)   # [500, 17]
proj_te = compute_projections(hs_te)

print("\n── A. VUF projections → LR ─────────────────────────────────────")
results = []

# All 17 projections at once
pipe_all = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
pipe_all.fit(proj_tr, hal_tr)
proba = pipe_all.predict_proba(proj_te)[:, 1]
auroc = roc_auc_score(hal_te, proba)
acc   = accuracy_score(hal_te, pipe_all.predict(proj_te))
results.append(dict(method="VUF_proj_all17", auroc=auroc, acc=acc))
print(f"  all 17 layers:  AUROC={auroc:.4f}  ACC={acc:.4f}")

# Per-layer projection
print("\n  Per-layer projection → LR:")
for li, layer in enumerate(LAYERS):
    X_tr = proj_tr[:, li:li+1]
    X_te = proj_te[:, li:li+1]
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_tr, hal_tr)
    proba = lr.predict_proba(X_te)[:, 1]
    auroc = roc_auc_score(hal_te, proba)
    acc   = accuracy_score(hal_te, lr.predict(X_te))
    results.append(dict(method=f"VUF_proj_L{layer}", auroc=auroc, acc=acc))
    print(f"    L{layer:<4} AUROC={auroc:.4f}  ACC={acc:.4f}")

# ── B. VUF residual: h - (h·r̂)r̂ → remaining signal ──────────────────────────
print("\n── B. VUF residual (h - proj component) best layers → LR ──────")
for li, layer in enumerate(LAYERS[:6]):   # check first few layers
    h_tr = hs_tr[:, li, :]
    h_te = hs_te[:, li, :]
    p_tr = proj_tr[:, li:li+1] * vuf_norm[li]  # [n, 4096]
    p_te = proj_te[:, li:li+1] * vuf_norm[li]
    resid_tr = h_tr - p_tr
    resid_te = h_te - p_te
    # Combine: projection scalar + residual norm as 2 features
    feat_tr = np.c_[proj_tr[:, li], np.linalg.norm(resid_tr, axis=1)]
    feat_te = np.c_[proj_te[:, li], np.linalg.norm(resid_te, axis=1)]
    lr = LogisticRegression(max_iter=1000)
    lr.fit(feat_tr, hal_tr)
    proba = lr.predict_proba(feat_te)[:, 1]
    auroc = roc_auc_score(hal_te, proba)
    acc   = accuracy_score(hal_te, lr.predict(feat_te))
    results.append(dict(method=f"VUF_proj+resid_norm_L{layer}", auroc=auroc, acc=acc))
    print(f"  L{layer:<4} proj+resid_norm  AUROC={auroc:.4f}  ACC={acc:.4f}")

# ── C. MLP on VUF projections ─────────────────────────────────────────────────
print("\n── C. MLP on VUF projections (17 features) ────────────────────")

class SmallMLP(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)

def train_mlp(X_tr, y_tr, X_te, y_te, in_dim, hidden=32, epochs=200, lr=1e-3, name=""):
    sc = StandardScaler()
    X_tr_s = torch.tensor(sc.fit_transform(X_tr), dtype=torch.float32)
    X_te_s = torch.tensor(sc.transform(X_te),     dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)

    model = SmallMLP(in_dim, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    crit = nn.BCEWithLogitsLoss()

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = crit(model(X_tr_s), y_tr_t)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(X_te_s).numpy()
    proba = 1 / (1 + np.exp(-logits))
    preds = (proba > 0.5).astype(int)
    auroc = roc_auc_score(y_te, proba)
    acc   = accuracy_score(y_te, preds)
    return auroc, acc, model, sc

for hidden in [16, 32, 64]:
    auroc, acc, mlp, sc = train_mlp(proj_tr, hal_tr, proj_te, hal_te,
                                     in_dim=17, hidden=hidden, name=f"MLP_h{hidden}")
    results.append(dict(method=f"MLP_h{hidden}_VUF_proj", auroc=auroc, acc=acc))
    print(f"  MLP hidden={hidden:<4}  AUROC={auroc:.4f}  ACC={acc:.4f}")

# ── D. MLP on best single-layer hidden states ─────────────────────────────────
print("\n── D. MLP on hidden states (best layer) ────────────────────────")
# Find best layer from direct LR result  — use L17 (from exp2)
for li, layer in enumerate([17, 18, 15]):
    li_idx = LAYERS.index(layer)
    for hidden in [64, 128]:
        auroc, acc, _, _ = train_mlp(
            hs_tr[:, li_idx, :], hal_tr,
            hs_te[:, li_idx, :], hal_te,
            in_dim=4096, hidden=hidden, epochs=300, lr=5e-4,
            name=f"MLP_h{hidden}_L{layer}"
        )
        results.append(dict(method=f"MLP_h{hidden}_hs_L{layer}", auroc=auroc, acc=acc))
        print(f"  MLP hidden={hidden:<4} L{layer}  AUROC={auroc:.4f}  ACC={acc:.4f}")

# ── Summary ────────────────────────────────────────────────────────────────────
results.sort(key=lambda r: r["auroc"], reverse=True)
print("\n── Summary (sorted by AUROC) ───────────────────────────────────")
print(f"{'Method':<35} {'AUROC':<8} {'ACC'}")
print("-"*55)
for r in results:
    print(f"  {r['method']:<33} {r['auroc']:.4f}   {r['acc']:.4f}")

# ── Save best VUF projection model ────────────────────────────────────────────
os.makedirs(config.MODELS_DIR, exist_ok=True)
joblib.dump(pipe_all, f"{config.MODELS_DIR}/vuf_proj_all17_lr.pkl")
np.save(f"{config.MODELS_DIR}/vuf_norm.npy", vuf_norm)
print(f"\nSaved VUF projection model to {config.MODELS_DIR}/")

os.makedirs(config.RESULTS_DIR, exist_ok=True)
with open(f"{config.RESULTS_DIR}/exp3_vuf_projection.json", "w") as f:
    json.dump({"experiment": "exp3_vuf_projection", "results": results}, f, indent=2)
print(f"Saved results to {config.RESULTS_DIR}/exp3_vuf_projection.json")
