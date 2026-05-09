#!/usr/bin/env python3
"""
Compare two ways to compute VUF in SAE latent space.

Variant 1 (hidden-first):
  VUF_h = mean(hidden_certain) - mean(hidden_uncertain)
  VUF_z1 = SAE.encode(VUF_h)

Variant 2 (latent-first):
  z_i = SAE.encode(hidden_i)
  VUF_z2 = mean(z_certain) - mean(z_uncertain)

Thresholds follow the paper defaults:
  certain   if VU <= 0.05
  uncertain if VU >= 0.90

Data expected in repo_root/vuf_checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import torch
import numpy as np
from scipy.stats import ttest_ind
from sae_lens import SAE

from sae_muc.layer_map import hf_layers_for_release

# Paper defaults (Ji et al. 2025)
VU_CERTAIN_MAX = 0.05
VU_UNCERTAIN_MIN = 0.90


def load_vu_labels(csv_path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            out[row["id"]] = float(row["verbal_uncertainty"])
    return out


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm() + 1e-8)


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.dot(l2_normalize(a), l2_normalize(b)).item())


def topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int = 64) -> float:
    ka = torch.topk(a.abs(), k=min(k, a.numel())).indices.tolist()
    kb = torch.topk(b.abs(), k=min(k, b.numel())).indices.tolist()
    sa, sb = set(ka), set(kb)
    return len(sa & sb) / max(1, len(sa | sb))


def topk_indices(x: torch.Tensor, k: int) -> list[int]:
    return torch.topk(x, k=min(k, x.numel())).indices.tolist()


def benjamini_hochberg_mask(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Return boolean mask of significant hypotheses after BH-FDR correction."""
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


def linear_encode_if_available(sae: SAE, h: torch.Tensor, device: str) -> torch.Tensor | None:
    """
    Compute pre-nonlinearity SAE encoder projection if weights are exposed.
    This check helps confirm implementation sanity:
      linear(mean(h_c)-mean(h_u))  ==  mean(linear(h_c))-mean(linear(h_u))
    """
    w_enc = getattr(sae, "W_enc", None)
    b_enc = getattr(sae, "b_enc", None)
    if w_enc is None:
        return None
    x = h.to(device=device, dtype=torch.float32)
    w = w_enc.detach().to(device=device, dtype=torch.float32)
    # Support both [d_in, d_sae] and [d_sae, d_in] conventions
    if w.ndim != 2:
        return None
    if w.shape[0] == x.numel():
        z = x @ w
    elif w.shape[1] == x.numel():
        z = x @ w.T
    else:
        return None
    if b_enc is not None:
        z = z + b_enc.detach().to(device=device, dtype=torch.float32)
    return z.detach().cpu().to(torch.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo_root", type=str, default=".")
    p.add_argument("--split", type=str, default="train", choices=["train", "test"])
    p.add_argument("--release", type=str, default="mistral-7b-res-wg")
    p.add_argument("--sae_device", type=str, default="cpu")
    p.add_argument("--sae_dtype", type=str, default="float32", choices=("float32", "float16", "bfloat16"))
    p.add_argument("--vu_certain_max", type=float, default=VU_CERTAIN_MAX)
    p.add_argument("--vu_uncertain_min", type=float, default=VU_UNCERTAIN_MIN)
    p.add_argument("--out_path", type=str, default="sae_muc/artifacts/vuf_sae_order_compare.pt")
    p.add_argument("--stats_json", type=str, default="sae_muc/artifacts/vuf_sae_order_compare_stats.json")
    p.add_argument("--top_k", type=int, default=64)
    p.add_argument("--fdr_alpha", type=float, default=0.05)
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    ckpt = repo_root / "vuf_checkpoint"

    csv_path = ckpt / "datasets" / "nq_open" / "Mistral-7B-Instruct-v0.3" / f"{args.split}.csv"
    hs_dir = ckpt / "datasets" / "nq_open" / "Mistral-7B-Instruct-v0.3" / "pipeline_used_layers_last" / args.split

    vu = load_vu_labels(csv_path)

    certain_ids = [sid for sid, v in vu.items() if v <= args.vu_certain_max]
    uncertain_ids = [sid for sid, v in vu.items() if v >= args.vu_uncertain_min]

    if not certain_ids or not uncertain_ids:
        raise RuntimeError(
            f"No data for thresholds: certain<= {args.vu_certain_max}, "
            f"uncertain>= {args.vu_uncertain_min}. "
            f"Found certain={len(certain_ids)}, uncertain={len(uncertain_ids)}"
        )

    print(f"Split={args.split}: certain={len(certain_ids)}, uncertain={len(uncertain_ids)}")

    # Determine stored layer keys from first sample
    sample_id = next(iter(vu.keys()))
    raw = torch.load(hs_dir / f"{sample_id}.pt", map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise RuntimeError("Expected hidden-state .pt as dict[layer_idx -> tensor].")
    stored_layers = sorted(raw.keys())
    d_model = int(raw[stored_layers[0]].numel())

    # We only compare layers where an SAE exists for this release
    mapping = dict(hf_layers_for_release(args.release))  # hf_layer -> sae_id
    usable_layers = [l for l in stored_layers if l in mapping]
    if not usable_layers:
        raise RuntimeError(
            f"No overlap between stored layers {stored_layers} and SAE mapping {sorted(mapping.keys())}."
        )

    print(f"Stored layers: {stored_layers}")
    print(f"SAE-usable layers: {usable_layers}")

    # Load SAEs
    saes: Dict[int, SAE] = {}
    for layer in usable_layers:
        sae_id = mapping[layer]
        print(f"Loading SAE for HF layer {layer}: {sae_id}")
        sae = SAE.from_pretrained(args.release, sae_id, device=args.sae_device, dtype=args.sae_dtype)
        if sae.cfg.d_in != d_model:
            raise ValueError(f"Layer {layer}: d_in mismatch, SAE d_in={sae.cfg.d_in}, hidden d_model={d_model}")
        saes[layer] = sae

    # Accumulators
    sum_hidden_certain = {l: torch.zeros(d_model, dtype=torch.float32) for l in usable_layers}
    sum_hidden_uncertain = {l: torch.zeros(d_model, dtype=torch.float32) for l in usable_layers}

    d_sae_by_layer = {l: int(saes[l].cfg.d_sae) for l in usable_layers}
    sum_latent_certain = {l: torch.zeros(d_sae_by_layer[l], dtype=torch.float32) for l in usable_layers}
    sum_latent_uncertain = {l: torch.zeros(d_sae_by_layer[l], dtype=torch.float32) for l in usable_layers}
    # Keep per-sample latent vectors for feature-level statistics (variant 2)
    latents_certain = {l: [] for l in usable_layers}
    latents_uncertain = {l: [] for l in usable_layers}
    # Optional pre-nonlinearity linear encoder accumulators (for debugging)
    sum_linear_certain = {l: torch.zeros(d_sae_by_layer[l], dtype=torch.float32) for l in usable_layers}
    sum_linear_uncertain = {l: torch.zeros(d_sae_by_layer[l], dtype=torch.float32) for l in usable_layers}

    n_certain = 0
    n_uncertain = 0

    # Iterate all samples once
    for sid, v in vu.items():
        is_certain = v <= args.vu_certain_max
        is_uncertain = v >= args.vu_uncertain_min
        if not (is_certain or is_uncertain):
            continue

        hs = torch.load(hs_dir / f"{sid}.pt", map_location="cpu", weights_only=False)

        for layer in usable_layers:
            h = hs[layer].to(dtype=torch.float32).reshape(-1)

            # For variant 1
            if is_certain:
                sum_hidden_certain[layer] += h
            if is_uncertain:
                sum_hidden_uncertain[layer] += h

            # For variant 2
            sae = saes[layer]
            z = sae.encode(h.to(device=args.sae_device).unsqueeze(0)).squeeze(0).detach().cpu().to(torch.float32)
            z_lin = linear_encode_if_available(sae, h, args.sae_device)
            if is_certain:
                sum_latent_certain[layer] += z
                latents_certain[layer].append(z.numpy())
                if z_lin is not None:
                    sum_linear_certain[layer] += z_lin
            if is_uncertain:
                sum_latent_uncertain[layer] += z
                latents_uncertain[layer].append(z.numpy())
                if z_lin is not None:
                    sum_linear_uncertain[layer] += z_lin

        if is_certain:
            n_certain += 1
        if is_uncertain:
            n_uncertain += 1

    if n_certain == 0 or n_uncertain == 0:
        raise RuntimeError(f"Invalid counts: certain={n_certain}, uncertain={n_uncertain}")

    results = {
        "meta": {
            "split": args.split,
            "release": args.release,
            "vu_certain_max": args.vu_certain_max,
            "vu_uncertain_min": args.vu_uncertain_min,
            "n_certain": n_certain,
            "n_uncertain": n_uncertain,
            "usable_layers": usable_layers,
        },
        "layers": {},
    }

    stats = {"meta": results["meta"], "layers": {}}
    aggregate_feature_hits = {
        "uncertainty_topk": {},
        "confidence_topk": {},
        "uncertainty_only_topk": {},
        "confidence_only_topk": {},
        "ttest_uncertainty_up_fdr": {},
        "ttest_confidence_up_fdr": {},
    }

    for layer in usable_layers:
        # Hidden means
        mu_h_c = sum_hidden_certain[layer] / float(n_certain)
        mu_h_u = sum_hidden_uncertain[layer] / float(n_uncertain)
        vuf_hidden = mu_h_c - mu_h_u

        # Variant 1: encode(hidden-difference)
        z1 = saes[layer].encode(vuf_hidden.to(device=args.sae_device).unsqueeze(0)).squeeze(0).detach().cpu().to(torch.float32)

        # Variant 2: difference(encode(hidden))
        mu_z_c = sum_latent_certain[layer] / float(n_certain)
        mu_z_u = sum_latent_uncertain[layer] / float(n_uncertain)
        z2 = mu_z_c - mu_z_u
        # Positive z2 => more active in certain; negative z2 => more active in uncertain
        z2_uncertainty = -z2

        # Comparison
        cosine = cos_sim(z1, z2)
        overlap64 = topk_overlap(z1, z2, k=64)
        overlap256 = topk_overlap(z1, z2, k=256)
        overlap1024 = topk_overlap(z1, z2, k=1024)
        cosine_normed = cos_sim(l2_normalize(z1), l2_normalize(z2))

        # Linearized sanity check (before nonlinearity), if W_enc/b_enc is available
        z1_lin = linear_encode_if_available(saes[layer], vuf_hidden, args.sae_device)
        linear_cos = None
        linear_rel_l2 = None
        if z1_lin is not None:
            mu_lin_c = sum_linear_certain[layer] / float(n_certain)
            mu_lin_u = sum_linear_uncertain[layer] / float(n_uncertain)
            z2_lin = mu_lin_c - mu_lin_u
            linear_cos = cos_sim(z1_lin, z2_lin)
            linear_rel_l2 = float((z1_lin - z2_lin).norm().item() / (z1_lin.norm().item() + 1e-8))

        # ---------------- Variant 2 feature analysis ----------------
        Xc = np.stack(latents_certain[layer], axis=0)    # [n_certain, d_sae]
        Xu = np.stack(latents_uncertain[layer], axis=0)  # [n_uncertain, d_sae]
        mean_c = Xc.mean(axis=0)
        mean_u = Xu.mean(axis=0)
        diff_u_minus_c = mean_u - mean_c
        diff_c_minus_u = mean_c - mean_u

        # Top features by raw mean-difference
        top_uncertainty_idx = topk_indices(torch.from_numpy(diff_u_minus_c).float(), args.top_k)
        top_confidence_idx = topk_indices(torch.from_numpy(diff_c_minus_u).float(), args.top_k)
        set_u = set(top_uncertainty_idx)
        set_c = set(top_confidence_idx)
        only_u = sorted(set_u - set_c)
        only_c = sorted(set_c - set_u)

        # T-tests per feature
        tt = ttest_ind(Xu, Xc, axis=0, equal_var=False, nan_policy="omit")
        pvals = np.asarray(tt.pvalue, dtype=np.float64)
        tvals = np.asarray(tt.statistic, dtype=np.float64)
        pvals = np.nan_to_num(pvals, nan=1.0, posinf=1.0, neginf=1.0)
        tvals = np.nan_to_num(tvals, nan=0.0, posinf=0.0, neginf=0.0)
        sig_mask = benjamini_hochberg_mask(pvals, alpha=args.fdr_alpha)
        sig_uncertainty = np.where(sig_mask & (tvals > 0))[0].tolist()  # Xu > Xc
        sig_confidence = np.where(sig_mask & (tvals < 0))[0].tolist()   # Xc > Xu

        # Add per-layer tallies to aggregate counters
        for i in top_uncertainty_idx:
            aggregate_feature_hits["uncertainty_topk"][str(i)] = aggregate_feature_hits["uncertainty_topk"].get(str(i), 0) + 1
        for i in top_confidence_idx:
            aggregate_feature_hits["confidence_topk"][str(i)] = aggregate_feature_hits["confidence_topk"].get(str(i), 0) + 1
        for i in only_u:
            aggregate_feature_hits["uncertainty_only_topk"][str(i)] = aggregate_feature_hits["uncertainty_only_topk"].get(str(i), 0) + 1
        for i in only_c:
            aggregate_feature_hits["confidence_only_topk"][str(i)] = aggregate_feature_hits["confidence_only_topk"].get(str(i), 0) + 1
        for i in sig_uncertainty:
            aggregate_feature_hits["ttest_uncertainty_up_fdr"][str(i)] = aggregate_feature_hits["ttest_uncertainty_up_fdr"].get(str(i), 0) + 1
        for i in sig_confidence:
            aggregate_feature_hits["ttest_confidence_up_fdr"][str(i)] = aggregate_feature_hits["ttest_confidence_up_fdr"].get(str(i), 0) + 1

        results["layers"][layer] = {
            "sae_id": mapping[layer],
            "vuf_hidden": vuf_hidden,
            "variant1_encode_of_hidden_diff": z1,
            "variant2_diff_of_encoded_hidden": z2,
            "variant1_linear_of_hidden_diff": z1_lin,
            "variant2_top_uncertainty_feature_idx": top_uncertainty_idx,
            "variant2_top_confidence_feature_idx": top_confidence_idx,
            "variant2_uncertainty_only_topk_idx": only_u,
            "variant2_confidence_only_topk_idx": only_c,
            "variant2_ttest_statistic": torch.from_numpy(tvals.astype(np.float32)),
            "variant2_ttest_pvalue": torch.from_numpy(pvals.astype(np.float32)),
            "variant2_ttest_fdr_significant_mask": torch.from_numpy(sig_mask.astype(np.bool_)),
        }

        stats["layers"][str(layer)] = {
            "sae_id": mapping[layer],
            "latent_dim": int(z1.numel()),
            "norm_variant1": float(z1.norm().item()),
            "norm_variant2": float(z2.norm().item()),
            "cosine_variant1_vs_variant2": cosine,
            "top64_jaccard": overlap64,
            "top256_jaccard": overlap256,
            "top1024_jaccard": overlap1024,
            "cosine_l2normed_variant1_vs_variant2": cosine_normed,
            "linear_cosine_variant1_vs_variant2": linear_cos,
            "linear_relative_l2_error_variant1_vs_variant2": linear_rel_l2,
            "variant2_feature_analysis": {
                "top_k": args.top_k,
                "top_uncertainty_feature_idx": top_uncertainty_idx,
                "top_confidence_feature_idx": top_confidence_idx,
                "overlap_topk_count": len(set_u & set_c),
                "uncertainty_only_topk_count": len(only_u),
                "confidence_only_topk_count": len(only_c),
                "uncertainty_only_topk_idx": only_u,
                "confidence_only_topk_idx": only_c,
                "fdr_alpha": args.fdr_alpha,
                "ttest_significant_total": int(sig_mask.sum()),
                "ttest_uncertainty_up_count": int(len(sig_uncertainty)),
                "ttest_confidence_up_count": int(len(sig_confidence)),
                "ttest_uncertainty_up_top50_by_abs_t": sorted(
                    sig_uncertainty, key=lambda i: abs(tvals[i]), reverse=True
                )[:50],
                "ttest_confidence_up_top50_by_abs_t": sorted(
                    sig_confidence, key=lambda i: abs(tvals[i]), reverse=True
                )[:50],
            },
        }

        print(
            f"Layer {layer}: cos(z1,z2)={cosine:.4f}, "
            f"top64/256/1024={overlap64:.4f}/{overlap256:.4f}/{overlap1024:.4f}, "
            f"||z1||={z1.norm().item():.3f}, ||z2||={z2.norm().item():.3f}"
        )
        if linear_cos is not None:
            print(
                f"  linear check: cos={linear_cos:.6f}, rel_l2_err={linear_rel_l2:.6e} "
                "(should be ~1.0 cosine and ~0 error)"
            )

    out_path = (repo_root / args.out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, out_path)

    stats_path = (repo_root / args.stats_json).resolve()
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    # Aggregate summary across layers
    def top_from_counter(counter: dict[str, int], n: int = 100) -> list[dict]:
        items = sorted(counter.items(), key=lambda kv: (-kv[1], int(kv[0])))
        return [{"feature_idx": int(k), "hits_across_layers": int(v)} for k, v in items[:n]]

    stats["aggregate"] = {
        "top_k": args.top_k,
        "fdr_alpha": args.fdr_alpha,
        "uncertainty_topk_most_frequent": top_from_counter(aggregate_feature_hits["uncertainty_topk"]),
        "confidence_topk_most_frequent": top_from_counter(aggregate_feature_hits["confidence_topk"]),
        "uncertainty_only_topk_most_frequent": top_from_counter(aggregate_feature_hits["uncertainty_only_topk"]),
        "confidence_only_topk_most_frequent": top_from_counter(aggregate_feature_hits["confidence_only_topk"]),
        "ttest_uncertainty_up_fdr_most_frequent": top_from_counter(aggregate_feature_hits["ttest_uncertainty_up_fdr"]),
        "ttest_confidence_up_fdr_most_frequent": top_from_counter(aggregate_feature_hits["ttest_confidence_up_fdr"]),
    }

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved tensors: {out_path}")
    print(f"Saved stats:   {stats_path}")


if __name__ == "__main__":
    main()
