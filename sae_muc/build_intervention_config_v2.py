#!/usr/bin/env python3
"""
Build per-layer intervention vectors for all 3 SAE steering methods:

  Method 1 (emd):           Encode-Modify-Decode + Error Term with consensus features
  Method 2 (projected_vuf): SAE-projected VUF (cleaned MUC)
  Method 3 (clamp):         Feature clamping (raise uncertainty features, suppress certainty features)

All methods share the same analysis step: load hidden states, encode via SAE, compute
feature-level statistics (mean-diff, t-test, Cohen's d, bootstrap stability), then
select consensus features.

Input:
  --repo_root:  root with vuf_checkpoint/ subtree
  --split:      train / test / both
  --release:    SAE release name (mistral-7b-res-wg)

Output:
  --out_path:   .pt file with intervention config for all 3 methods

Example:
  python -m sae_muc.build_intervention_config_v2 \
      --repo_root /path/to/sae \
      --split both \
      --release mistral-7b-res-wg \
      --out_path sae_muc/artifacts/intervention_v2.pt
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ttest_ind
from sae_lens import SAE

_PKG_DIR = Path(__file__).resolve().parent
_REPO_PARENT = _PKG_DIR.parent
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

from sae_muc.layer_map import hf_layers_for_release

# --------------------------------------------------------------------------- #
#  Paper defaults                                                              #
# --------------------------------------------------------------------------- #
VU_CERTAIN_MAX = 0.05
VU_UNCERTAIN_MIN = 0.80


# --------------------------------------------------------------------------- #
#  Statistics helpers                                                           #
# --------------------------------------------------------------------------- #

def benjamini_hochberg_mask(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    n = pvals.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresholds = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= thresholds
    if not np.any(passed):
        return np.zeros(n, dtype=bool)
    kmax = np.max(np.where(passed)[0])
    cutoff = ranked[kmax]
    return pvals <= cutoff


def cohens_d(Xu: np.ndarray, Xc: np.ndarray) -> np.ndarray:
    """Per-feature Cohen's d: positive means uncertain > certain."""
    mu_u = Xu.mean(axis=0)
    mu_c = Xc.mean(axis=0)
    var_u = Xu.var(axis=0, ddof=1)
    var_c = Xc.var(axis=0, ddof=1)
    n_u, n_c = Xu.shape[0], Xc.shape[0]
    pooled = np.sqrt(((n_u - 1) * var_u + (n_c - 1) * var_c) / max(1, (n_u + n_c - 2)))
    return (mu_u - mu_c) / (pooled + 1e-12)


def bootstrap_topk_stability(
    Xu: np.ndarray,
    Xc: np.ndarray,
    top_k: int,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    d = Xu.shape[1]
    hit_u = np.zeros(d, dtype=np.int32)
    hit_c = np.zeros(d, dtype=np.int32)
    n_u, n_c = Xu.shape[0], Xc.shape[0]
    k = min(top_k, d)

    for _ in range(n_boot):
        bu = Xu[rng.integers(0, n_u, size=n_u)]
        bc = Xc[rng.integers(0, n_c, size=n_c)]
        diff = bu.mean(axis=0) - bc.mean(axis=0)
        hit_u[np.argsort(diff)[-k:]] += 1
        hit_c[np.argsort(-diff)[-k:]] += 1

    return hit_u / float(n_boot), hit_c / float(n_boot)


# --------------------------------------------------------------------------- #
#  Data loading                                                                #
# --------------------------------------------------------------------------- #

def load_vu_labels(csv_path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            out[row["id"]] = float(row["verbal_uncertainty"])
    return out


def load_hidden_states_and_encode(
    sae: SAE,
    layer: int,
    sample_ids: list[str],
    id_to_hs_dir: dict[str, Path],
    device: str,
) -> np.ndarray:
    """Encode hidden states through SAE for given samples. Returns [n_samples, d_sae]."""
    zs = []
    for sid in sample_ids:
        hs = torch.load(id_to_hs_dir[sid] / f"{sid}.pt", map_location="cpu", weights_only=False)
        h = hs[layer].to(dtype=torch.float32).unsqueeze(0).to(device=device)
        z = sae.encode(h).squeeze(0).detach().cpu().numpy().astype(np.float32)
        zs.append(z)
    return np.stack(zs, axis=0)


# --------------------------------------------------------------------------- #
#  Per-layer analysis + config building                                        #
# --------------------------------------------------------------------------- #

def analyze_and_build_layer(
    layer: int,
    sae_id: str,
    release: str,
    id_to_hs_dir: dict[str, Path],
    certain_ids: list[str],
    uncertain_ids: list[str],
    hedge_vec: torch.Tensor | None,
    device: str,
    sae_dtype: str,
    top_k: int,
    fdr_alpha: float,
    n_bootstrap: int,
    bootstrap_freq_threshold: float,
    cohens_d_threshold: float,
    seed: int,
) -> dict:
    """
    Analyze one layer and return config for all 3 methods.

    Returns dict with keys:
      sae_id, layer,
      method_emd:           {delta: Tensor[d_sae], feature_indices, weights}
      method_projected_vuf: {delta: Tensor[d_sae], feature_mask: Tensor[d_sae]}  (or None if no hedge)
      method_clamp:         {uncertainty_features, certainty_features,
                             target_uncertain_values, mean_certain_values}
      stats: summary statistics
    """
    print(f"\n{'='*60}")
    print(f"Layer {layer} | {sae_id}")

    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device, dtype=sae_dtype)
    d_sae = sae.cfg.d_sae

    # Encode all samples
    print(f"  Encoding {len(certain_ids)} certain + {len(uncertain_ids)} uncertain samples...")
    Xc = load_hidden_states_and_encode(sae, layer, certain_ids, id_to_hs_dir, device)
    Xu = load_hidden_states_and_encode(sae, layer, uncertain_ids, id_to_hs_dir, device)

    rng = np.random.default_rng(seed + int(layer))

    mean_c = np.nan_to_num(Xc.mean(axis=0), nan=0.0)
    mean_u = np.nan_to_num(Xu.mean(axis=0), nan=0.0)

    # ---------- Feature-level statistics ----------

    # 1. Mean diff
    diff_u_minus_c = mean_u - mean_c

    # 2. Welch t-test + BH-FDR
    tt = ttest_ind(Xu, Xc, axis=0, equal_var=False, nan_policy="omit")
    pvals = np.nan_to_num(np.asarray(tt.pvalue, dtype=np.float64), nan=1.0)
    tvals = np.nan_to_num(np.asarray(tt.statistic, dtype=np.float64), nan=0.0)
    sig = benjamini_hochberg_mask(pvals, alpha=fdr_alpha)

    # 3. Cohen's d
    dvals = np.nan_to_num(cohens_d(Xu, Xc), nan=0.0)

    # 4. Bootstrap stability
    print(f"  Bootstrap ({n_bootstrap} iters)...")
    stab_u, stab_c = bootstrap_topk_stability(Xu, Xc, top_k=top_k, n_boot=n_bootstrap, rng=rng)

    # 5. Frequency diff
    freq_c = (Xc > 0).mean(axis=0)
    freq_u = (Xu > 0).mean(axis=0)

    # ---------- Consensus feature selection ----------

    # Uncertainty-up features: bootstrap-stable AND (FDR-significant OR |d| > threshold)
    unc_bootstrap = stab_u >= bootstrap_freq_threshold
    unc_sig = sig & (tvals > 0)
    unc_d_large = dvals > cohens_d_threshold
    unc_consensus = unc_bootstrap & (unc_sig | unc_d_large)

    # Certainty-up features: bootstrap-stable AND (FDR-significant OR |d| > threshold)
    cert_bootstrap = stab_c >= bootstrap_freq_threshold
    cert_sig = sig & (tvals < 0)
    cert_d_large = dvals < -cohens_d_threshold
    cert_consensus = cert_bootstrap & (cert_sig | cert_d_large)

    unc_idx = np.where(unc_consensus)[0].tolist()
    cert_idx = np.where(cert_consensus)[0].tolist()

    print(f"  Consensus uncertainty-up features: {len(unc_idx)}")
    print(f"  Consensus certainty-up features:   {len(cert_idx)}")

    # ---------- Method 1: EMD (Encode-Modify-Decode) ----------
    # delta weighted by Cohen's d: positive for uncertainty-up, negative for certainty-up
    delta_emd = torch.zeros(d_sae, dtype=torch.float32)
    emd_features = []
    emd_weights = []
    for i in unc_idx:
        w = float(dvals[i])  # positive Cohen's d
        delta_emd[i] = w
        emd_features.append(i)
        emd_weights.append(w)
    for i in cert_idx:
        w = float(dvals[i])  # negative Cohen's d (suppresses certainty)
        delta_emd[i] = -w  # flip sign: we want to increase uncertainty
        emd_features.append(i)
        emd_weights.append(-w)

    # L2 normalize
    norm = delta_emd.norm()
    if norm > 1e-8:
        delta_emd = delta_emd / norm

    # ---------- Method 2: Projected VUF ----------
    method_projected_vuf = None
    if hedge_vec is not None:
        h = hedge_vec.to(device=device, dtype=torch.float32)
        h = h / (h.norm() + 1e-8)
        w_dec = sae.W_dec.detach().to(device=device, dtype=torch.float32)
        # Project VUF onto SAE decoder directions
        vuf_scores = ((h @ w_dec.T) / (w_dec.norm(dim=1) + 1e-8)).detach().cpu()

        # Feature mask: keep only features that are FDR-significant OR in consensus
        feature_mask = torch.zeros(d_sae, dtype=torch.float32)
        all_relevant = set(unc_idx) | set(cert_idx)
        # Also add top-k by hedge projection that are FDR-significant
        hedge_topk_idx = torch.topk(vuf_scores.abs(), k=min(top_k * 2, d_sae)).indices.tolist()
        for i in hedge_topk_idx:
            if sig[i]:
                all_relevant.add(i)
        for i in all_relevant:
            feature_mask[i] = 1.0

        # Build clean delta: masked hedge scores, normalized
        delta_proj = vuf_scores * feature_mask
        proj_norm = delta_proj.norm()
        if proj_norm > 1e-8:
            delta_proj = delta_proj / proj_norm

        method_projected_vuf = {
            "delta": delta_proj,
            "feature_mask": feature_mask,
            "n_features_kept": int(feature_mask.sum().item()),
        }
        print(f"  Projected VUF: {method_projected_vuf['n_features_kept']} features retained")

    # ---------- Method 3: Clamp ----------
    # Target values for uncertainty features: mean activation on uncertain samples
    target_unc_vals = {}
    for i in unc_idx:
        target_unc_vals[i] = float(mean_u[i])

    # Mean values for certainty features (to know how much to suppress)
    mean_cert_vals = {}
    for i in cert_idx:
        mean_cert_vals[i] = float(mean_c[i])

    # ---------- Summary stats ----------
    stats = {
        "layer": layer,
        "sae_id": sae_id,
        "n_certain": int(Xc.shape[0]),
        "n_uncertain": int(Xu.shape[0]),
        "d_sae": int(d_sae),
        "n_unc_consensus": len(unc_idx),
        "n_cert_consensus": len(cert_idx),
        "fdr_significant_total": int(sig.sum()),
        "fdr_unc_up": int(unc_sig.sum()),
        "fdr_cert_up": int(cert_sig.sum()),
        "bootstrap_stable_unc": int(unc_bootstrap.sum()),
        "bootstrap_stable_cert": int(cert_bootstrap.sum()),
    }

    result = {
        "layer": layer,
        "sae_id": sae_id,
        "method_emd": {
            "delta": delta_emd,
            "feature_indices": emd_features,
            "weights": emd_weights,
        },
        "method_projected_vuf": method_projected_vuf,
        "method_clamp": {
            "uncertainty_features": unc_idx,
            "certainty_features": cert_idx,
            "target_uncertain_values": target_unc_vals,
            "mean_certain_values": mean_cert_vals,
            "freq_uncertain": {i: float(freq_u[i]) for i in unc_idx},
            "freq_certain": {i: float(freq_c[i]) for i in cert_idx},
        },
        "stats": stats,
    }

    return result


# --------------------------------------------------------------------------- #
#  Main                                                                        #
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build intervention config for all 3 SAE steering methods")
    p.add_argument("--repo_root", type=str, default=".")
    p.add_argument("--split", type=str, default="both", choices=["train", "test", "both"])
    p.add_argument("--release", type=str, default="mistral-7b-res-wg")
    p.add_argument("--sae_device", type=str, default="cpu")
    p.add_argument("--sae_dtype", type=str, default="float32", choices=("float32", "float16", "bfloat16"))
    p.add_argument("--vu_certain_max", type=float, default=VU_CERTAIN_MAX)
    p.add_argument("--vu_uncertain_min", type=float, default=VU_UNCERTAIN_MIN)
    p.add_argument("--top_k", type=int, default=64)
    p.add_argument("--fdr_alpha", type=float, default=0.05)
    p.add_argument("--n_bootstrap", type=int, default=200)
    p.add_argument("--bootstrap_freq_threshold", type=float, default=0.8)
    p.add_argument("--cohens_d_threshold", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hedge_path", type=str, default="",
                   help="Path to Hs_hedge_universal.pt for projected VUF method")
    p.add_argument("--out_path", type=str, default="sae_muc/artifacts/intervention_v2.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    ckpt = repo_root / "vuf_checkpoint"

    base_csv = ckpt / "datasets" / "nq_open" / "Mistral-7B-Instruct-v0.3"
    base_hs = ckpt / "datasets" / "nq_open" / "Mistral-7B-Instruct-v0.3" / "pipeline_used_layers_last"

    if args.split == "both":
        split_names = ["train", "test"]
    else:
        split_names = [args.split]

    # Build merged VU map and per-id path map
    vu: dict[str, float] = {}
    id_to_hs_dir: dict[str, Path] = {}
    for sp in split_names:
        csv_path = base_csv / f"{sp}.csv"
        vu_sp = load_vu_labels(csv_path)
        hs_dir_sp = base_hs / sp
        for sid, v in vu_sp.items():
            vu[sid] = v
            id_to_hs_dir[sid] = hs_dir_sp

    certain_ids = [sid for sid, v in vu.items() if v <= args.vu_certain_max]
    uncertain_ids = [sid for sid, v in vu.items() if v >= args.vu_uncertain_min]

    if not certain_ids or not uncertain_ids:
        raise RuntimeError(
            f"No data: certain={len(certain_ids)}, uncertain={len(uncertain_ids)}"
        )

    print(f"Split={args.split}: certain={len(certain_ids)}, uncertain={len(uncertain_ids)}")

    # Determine usable layers
    sample_id = next(iter(vu.keys()))
    raw = torch.load(id_to_hs_dir[sample_id] / f"{sample_id}.pt", map_location="cpu", weights_only=False)
    stored_layers = sorted(raw.keys())
    mapping = dict(hf_layers_for_release(args.release))
    usable_layers = [l for l in stored_layers if l in mapping]

    if not usable_layers:
        raise RuntimeError(f"No overlap between stored layers {stored_layers} and mapping {sorted(mapping.keys())}")

    print(f"Usable layers: {usable_layers}")

    # Load hedge vector for projected VUF method
    hedge = None
    if args.hedge_path:
        hp = Path(args.hedge_path)
        if not hp.is_file():
            hp = repo_root / args.hedge_path
        if hp.is_file():
            hedge = torch.load(hp, map_location="cpu", weights_only=False).float()
            print(f"Loaded hedge from {hp}, shape={tuple(hedge.shape)}")

    # Process each layer
    layer_configs = {}
    for layer in usable_layers:
        layer_hedge = None
        if hedge is not None and layer < hedge.shape[0]:
            layer_hedge = hedge[layer]

        config = analyze_and_build_layer(
            layer=layer,
            sae_id=mapping[layer],
            release=args.release,
            id_to_hs_dir=id_to_hs_dir,
            certain_ids=certain_ids,
            uncertain_ids=uncertain_ids,
            hedge_vec=layer_hedge,
            device=args.sae_device,
            sae_dtype=args.sae_dtype,
            top_k=args.top_k,
            fdr_alpha=args.fdr_alpha,
            n_bootstrap=args.n_bootstrap,
            bootstrap_freq_threshold=args.bootstrap_freq_threshold,
            cohens_d_threshold=args.cohens_d_threshold,
            seed=args.seed,
        )
        layer_configs[layer] = config

    # Save
    output = {
        "release": args.release,
        "meta": {
            "split": args.split,
            "vu_certain_max": args.vu_certain_max,
            "vu_uncertain_min": args.vu_uncertain_min,
            "n_certain": len(certain_ids),
            "n_uncertain": len(uncertain_ids),
            "top_k": args.top_k,
            "fdr_alpha": args.fdr_alpha,
            "n_bootstrap": args.n_bootstrap,
            "bootstrap_freq_threshold": args.bootstrap_freq_threshold,
            "cohens_d_threshold": args.cohens_d_threshold,
        },
        "layers": layer_configs,
    }

    out_path = Path(args.out_path)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out_path)
    print(f"\nSaved intervention config to {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    for layer, cfg in layer_configs.items():
        s = cfg["stats"]
        print(
            f"  Layer {layer}: "
            f"unc_consensus={s['n_unc_consensus']}, "
            f"cert_consensus={s['n_cert_consensus']}, "
            f"fdr_sig={s['fdr_significant_total']}"
        )
        if cfg["method_projected_vuf"]:
            print(f"    projected_vuf: {cfg['method_projected_vuf']['n_features_kept']} features")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
