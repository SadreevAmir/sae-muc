#!/usr/bin/env python3
"""
SAE Feature Analysis: find interpretable features distinguishing
certain (low VU) vs uncertain (high VU) model activations.

Uses pre-computed hidden states — NO GPU model inference required.
SAE encode/decode runs on CPU (or CUDA if available).

Usage:
    python probes_experiment/sae_feature_analysis.py [--device cpu|cuda]
"""

import argparse
import json
import os
import sys

import numpy as np
import requests
import torch

# Add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from sae_lens import SAE

# ── Config ────────────────────────────────────────────────────────────────────

# SAE layers available for Mistral-7B
SAE_RELEASE = "mistral-7b-res-wg"
LAYER_MAP = {
    7:  "blocks.8.hook_resid_pre",
    15: "blocks.16.hook_resid_pre",
    23: "blocks.24.hook_resid_pre",
}

# Pre-computed activations [n_samples, 32_layers, 4096_d_model]
ACT_PATHS = {
    "certain_test":    f"{ROOT}/calibration__outputs/nq_open/Mistral-7B-Instruct-v0.3/uncertainty/test/certain_verbal.pt",
    "uncertain_test":  f"{ROOT}/calibration__outputs/nq_open/Mistral-7B-Instruct-v0.3/uncertainty/test/uncertain_verbal.pt",
    "certain_train":   f"{ROOT}/calibration__outputs/nq_open/Mistral-7B-Instruct-v0.3/uncertainty/train/certain_verbal.pt",
    "uncertain_train": f"{ROOT}/calibration__outputs/nq_open/Mistral-7B-Instruct-v0.3/uncertainty/train/uncertain_verbal.pt",
}

# VUF direction vector [32, 4096]
HEDGE_PATH = f"{ROOT}/calibration__outputs/merged/Mistral-7B-Instruct-v0.3/uncertainty/Hs_hedge_universal.pt"

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Neuronpedia ───────────────────────────────────────────────────────────────

def fetch_neuronpedia(model_id: str, sae_id: str, feature_idx: int) -> dict | None:
    """Fetch feature interpretation from Neuronpedia API."""
    url = f"https://www.neuronpedia.org/api/feature/{model_id}/{sae_id}/{feature_idx}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def get_explanation(data: dict) -> str:
    """Extract human-readable explanation from Neuronpedia response."""
    if not data:
        return ""
    expl = data.get("explanations") or data.get("explanation")
    if isinstance(expl, list) and expl:
        e = expl[0]
        return e.get("description", "") if isinstance(e, dict) else str(e)
    if isinstance(expl, str):
        return expl
    return ""


# ── Core Analysis ─────────────────────────────────────────────────────────────

def load_activations(device: str) -> dict:
    """Load all pre-computed activations."""
    acts = {}
    for name, path in ACT_PATHS.items():
        t = torch.load(path, map_location="cpu", weights_only=False)
        acts[name] = t.float()
        print(f"  {name}: {t.shape}")
    return acts


def encode_through_sae(sae: SAE, activations: torch.Tensor, device: str) -> torch.Tensor:
    """Encode activations [n, d_model] through SAE → [n, d_sae]."""
    x = activations.to(device=sae.device, dtype=sae.dtype)
    with torch.no_grad():
        features = sae.encode(x)
    return features.float().cpu()


def analyze_layer(
    hf_layer: int,
    sae_id: str,
    certain_acts: torch.Tensor,    # [n_certain, d_model]
    uncertain_acts: torch.Tensor,  # [n_uncertain, d_model]
    hedge_vec: torch.Tensor,       # [d_model]
    device: str,
    top_k: int = 50,
) -> dict:
    """
    Full analysis for one layer:
    1. Encode certain/uncertain through SAE
    2. Compute mean activation per feature for each group
    3. Find features with largest difference (uncertain - certain)
    4. Also project hedge vector onto SAE decoder (delta_from_hedge)
    """
    print(f"\n{'='*60}")
    print(f"Layer {hf_layer} (SAE: {sae_id})")
    print(f"  Certain: {certain_acts.shape[0]} samples, Uncertain: {uncertain_acts.shape[0]} samples")
    print(f"{'='*60}")

    # Load SAE
    print("  Loading SAE...")
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=sae_id, device=device)
    d_sae = sae.cfg.d_sae
    print(f"  SAE d_sae={d_sae}, d_in={sae.cfg.d_in}")

    # Encode both groups
    print("  Encoding certain activations...")
    feat_certain = encode_through_sae(sae, certain_acts, device)     # [n_c, d_sae]
    print("  Encoding uncertain activations...")
    feat_uncertain = encode_through_sae(sae, uncertain_acts, device) # [n_u, d_sae]

    # ── Approach 1: Mean activation difference ────────────────────────────
    mean_certain   = feat_certain.mean(dim=0)    # [d_sae]
    mean_uncertain = feat_uncertain.mean(dim=0)  # [d_sae]
    diff = mean_uncertain - mean_certain          # positive = more active for uncertain

    # Top features by absolute difference
    top_idx = torch.topk(diff.abs(), k=top_k).indices
    top_features = []
    for idx in top_idx:
        i = idx.item()
        top_features.append({
            "feature_idx": i,
            "diff": float(diff[i]),
            "mean_certain": float(mean_certain[i]),
            "mean_uncertain": float(mean_uncertain[i]),
            "direction": "uncertain" if diff[i] > 0 else "certain",
        })

    # ── Approach 2: Frequency difference ──────────────────────────────────
    # How often is feature active (>0) in each group?
    freq_certain   = (feat_certain > 0).float().mean(dim=0)
    freq_uncertain = (feat_uncertain > 0).float().mean(dim=0)
    freq_diff = freq_uncertain - freq_certain

    top_freq_idx = torch.topk(freq_diff.abs(), k=top_k).indices
    top_freq_features = []
    for idx in top_freq_idx:
        i = idx.item()
        top_freq_features.append({
            "feature_idx": i,
            "freq_diff": float(freq_diff[i]),
            "freq_certain": float(freq_certain[i]),
            "freq_uncertain": float(freq_uncertain[i]),
            "direction": "uncertain" if freq_diff[i] > 0 else "certain",
        })

    # ── Approach 3: Welch's t-test per feature ────────────────────────────
    n_c, n_u = feat_certain.shape[0], feat_uncertain.shape[0]
    var_c = feat_certain.var(dim=0)   # [d_sae]
    var_u = feat_uncertain.var(dim=0) # [d_sae]
    se = torch.sqrt(var_c / n_c + var_u / n_u + 1e-12)
    t_stat = diff / se  # Welch's t-statistic

    top_t_idx = torch.topk(t_stat.abs(), k=top_k).indices
    top_t_features = []
    for idx in top_t_idx:
        i = idx.item()
        top_t_features.append({
            "feature_idx": i,
            "t_stat": float(t_stat[i]),
            "diff": float(diff[i]),
            "mean_certain": float(mean_certain[i]),
            "mean_uncertain": float(mean_uncertain[i]),
            "direction": "uncertain" if diff[i] > 0 else "certain",
        })

    # ── Approach 4: Hedge vector projection onto SAE ──────────────────────
    h = hedge_vec.to(device=sae.device, dtype=sae.dtype)
    h = h / (h.norm() + 1e-8)
    w_dec = sae.W_dec  # [d_sae, d_in]
    scores = (h @ w_dec.T) / (w_dec.norm(dim=1) + 1e-8)
    scores = scores.float().cpu()

    top_hedge_idx = torch.topk(scores.abs(), k=top_k).indices
    top_hedge_features = []
    for idx in top_hedge_idx:
        i = idx.item()
        top_hedge_features.append({
            "feature_idx": i,
            "hedge_score": float(scores[i].detach()),
            "mean_certain": float(mean_certain[i]),
            "mean_uncertain": float(mean_uncertain[i]),
            "direction": "uncertain" if scores[i] > 0 else "certain",
        })

    # ── Print summary ─────────────────────────────────────────────────────
    print(f"\n  Top features by MEAN ACTIVATION DIFFERENCE (uncertain - certain):")
    print(f"  {'Idx':<8} {'Diff':>8} {'Certain':>10} {'Uncertain':>10} {'Dir'}")
    print(f"  {'-'*50}")
    for f in top_features[:20]:
        print(f"  {f['feature_idx']:<8} {f['diff']:>8.3f} {f['mean_certain']:>10.3f} {f['mean_uncertain']:>10.3f} {f['direction']}")

    print(f"\n  Top features by T-STATISTIC:")
    print(f"  {'Idx':<8} {'t-stat':>8} {'Diff':>8} {'Dir'}")
    print(f"  {'-'*35}")
    for f in top_t_features[:20]:
        print(f"  {f['feature_idx']:<8} {f['t_stat']:>8.2f} {f['diff']:>8.3f} {f['direction']}")

    print(f"\n  Top features by HEDGE PROJECTION:")
    print(f"  {'Idx':<8} {'Score':>8} {'Certain':>10} {'Uncertain':>10} {'Dir'}")
    print(f"  {'-'*50}")
    for f in top_hedge_features[:20]:
        print(f"  {f['feature_idx']:<8} {f['hedge_score']:>8.4f} {f['mean_certain']:>10.3f} {f['mean_uncertain']:>10.3f} {f['direction']}")

    # ── Overlap analysis ──────────────────────────────────────────────────
    set_diff  = set(f["feature_idx"] for f in top_features[:20])
    set_tstat = set(f["feature_idx"] for f in top_t_features[:20])
    set_hedge = set(f["feature_idx"] for f in top_hedge_features[:20])

    print(f"\n  Overlap (top-20):")
    print(f"    diff ∩ t-stat: {len(set_diff & set_tstat)} features")
    print(f"    diff ∩ hedge:  {len(set_diff & set_hedge)} features")
    print(f"    t-stat ∩ hedge: {len(set_tstat & set_hedge)} features")
    print(f"    all three:      {len(set_diff & set_tstat & set_hedge)} features")

    # Clean up GPU memory
    del sae
    if device != "cpu":
        torch.cuda.empty_cache()

    return {
        "hf_layer": hf_layer,
        "sae_id": sae_id,
        "n_certain": n_c,
        "n_uncertain": n_u,
        "d_sae": d_sae,
        "top_by_mean_diff": top_features,
        "top_by_frequency": top_freq_features,
        "top_by_tstat": top_t_features,
        "top_by_hedge": top_hedge_features,
    }


def fetch_interpretations(results: dict, limit: int = 10) -> dict:
    """Try to fetch Neuronpedia interpretations for top features."""
    # Neuronpedia model/sae IDs for Mistral may not be available
    # Try common patterns
    model_ids = ["mistral-7b-instruct-v0.3", "mistral-7b", "mistral-7b-v0.3"]

    interpretations = {}
    for layer_result in results["layers"]:
        hf_layer = layer_result["hf_layer"]
        sae_id_candidates = [
            f"blocks.{hf_layer + 1}.hook_resid_pre",
            layer_result["sae_id"],
        ]

        top_indices = [f["feature_idx"] for f in layer_result["top_by_tstat"][:limit]]

        for feat_idx in top_indices:
            found = False
            for model_id in model_ids:
                for sid in sae_id_candidates:
                    data = fetch_neuronpedia(model_id, sid, feat_idx)
                    expl = get_explanation(data)
                    if expl:
                        interpretations[f"L{hf_layer}_{feat_idx}"] = {
                            "feature_idx": feat_idx,
                            "layer": hf_layer,
                            "explanation": expl,
                            "model_id": model_id,
                            "sae_id": sid,
                        }
                        found = True
                        break
                if found:
                    break

            if not found:
                interpretations[f"L{hf_layer}_{feat_idx}"] = {
                    "feature_idx": feat_idx,
                    "layer": hf_layer,
                    "explanation": "(not available on Neuronpedia)",
                }

        print(f"  Layer {hf_layer}: fetched {len(top_indices)} feature interpretations")

    return interpretations


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SAE Feature Analysis: certain vs uncertain")
    parser.add_argument("--device", default="cpu", help="cpu or cuda (default: cpu)")
    parser.add_argument("--top_k", type=int, default=50, help="Top K features per method")
    parser.add_argument("--fetch_neuronpedia", action="store_true", help="Fetch interpretations from Neuronpedia API")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print("=" * 60)
    print("SAE Feature Analysis: Certain vs Uncertain")
    print("=" * 60)

    # Load activations
    print("\nLoading pre-computed activations...")
    acts = load_activations(device)

    # Combine train+test for more samples
    certain_all   = torch.cat([acts["certain_train"], acts["certain_test"]], dim=0)
    uncertain_all = torch.cat([acts["uncertain_train"], acts["uncertain_test"]], dim=0)
    print(f"\n  Combined: certain={certain_all.shape[0]}, uncertain={uncertain_all.shape[0]}")

    # Load hedge vector
    hedge = torch.load(HEDGE_PATH, map_location="cpu", weights_only=False).float()
    print(f"  Hedge vector: {hedge.shape}")

    # Analyze each SAE layer
    all_results = {"layers": []}
    for hf_layer, sae_id in LAYER_MAP.items():
        certain_layer   = certain_all[:, hf_layer, :]    # [n, 4096]
        uncertain_layer = uncertain_all[:, hf_layer, :]  # [n, 4096]
        hedge_layer     = hedge[hf_layer]                 # [4096]

        result = analyze_layer(
            hf_layer=hf_layer,
            sae_id=sae_id,
            certain_acts=certain_layer,
            uncertain_acts=uncertain_layer,
            hedge_vec=hedge_layer,
            device=device,
            top_k=args.top_k,
        )
        all_results["layers"].append(result)

    # Neuronpedia interpretations
    if args.fetch_neuronpedia:
        print("\n\nFetching Neuronpedia interpretations...")
        interps = fetch_interpretations(all_results, limit=10)
        all_results["neuronpedia"] = interps

        print("\n  Interpretations found:")
        for key, val in interps.items():
            expl = val.get("explanation", "")[:80]
            print(f"    {key}: {expl}")

    # Save results
    out_path = os.path.join(RESULTS_DIR, "sae_feature_analysis.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for lr in all_results["layers"]:
        print(f"\nLayer {lr['hf_layer']} (d_sae={lr['d_sae']}):")
        # Top uncertain-direction features
        unc_feats = [f for f in lr["top_by_tstat"][:10] if f["direction"] == "uncertain"]
        cert_feats = [f for f in lr["top_by_tstat"][:10] if f["direction"] == "certain"]
        print(f"  Top uncertainty features (t-stat): {[f['feature_idx'] for f in unc_feats]}")
        print(f"  Top certainty features (t-stat):   {[f['feature_idx'] for f in cert_feats]}")


if __name__ == "__main__":
    main()
