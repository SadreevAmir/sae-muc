#!/usr/bin/env python3
"""
Unified SAE feature analysis for METHOD 2 only:

    z_i = SAE.encode(h_i)
    VUF_z = mean(z_certain) - mean(z_uncertain)

This script combines:
- our robust setup: explicit split control, no leakage by default, FDR-corrected stats,
- your friend's exploratory analysis: mean-diff, frequency-diff, hedge-projection,
  overlap reporting, clear human-readable summary.

Outputs:
- JSON summary for easy inspection
- optional PT with tensors (full vectors)

Example:
  python -m sae_muc.sae_feature_analysis_merged \
      --repo_root . --split train --release mistral-7b-res-wg --top_k 64
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sae_lens import SAE
from scipy.stats import ttest_ind

# Add project root to path when running as a script.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sae_muc.layer_map import hf_layers_for_release

# Paper defaults
VU_CERTAIN_MAX = 0.05
VU_UNCERTAIN_MIN = 0.90


def load_vu_labels(csv_path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            out[row["id"]] = float(row["verbal_uncertainty"])
    return out


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


def topk_indices(x: np.ndarray, k: int) -> list[int]:
    k = min(k, x.shape[0])
    return np.argsort(np.abs(x))[-k:][::-1].tolist()


def topk_positive_indices(x: np.ndarray, k: int) -> list[int]:
    idx = np.where(x > 0)[0]
    if idx.size == 0:
        return []
    vals = x[idx]
    order = np.argsort(vals)[::-1]
    take = idx[order][: min(k, idx.size)]
    return take.tolist()


def topk_overlap(a_idx: list[int], b_idx: list[int]) -> float:
    sa, sb = set(a_idx), set(b_idx)
    return len(sa & sb) / max(1, len(sa | sb))


def l2norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cohens_d_uncertain_vs_certain(Xu: np.ndarray, Xc: np.ndarray) -> np.ndarray:
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
    """
    Per-feature inclusion frequency in top-k over bootstrap resamples.
    Returns:
      stability_uncertain_up, stability_confidence_up
    """
    d = Xu.shape[1]
    hit_u = np.zeros(d, dtype=np.int32)
    hit_c = np.zeros(d, dtype=np.int32)
    n_u, n_c = Xu.shape[0], Xc.shape[0]
    k = min(top_k, d)

    for _ in range(n_boot):
        bu = Xu[rng.integers(0, n_u, size=n_u)]
        bc = Xc[rng.integers(0, n_c, size=n_c)]
        diff = bu.mean(axis=0) - bc.mean(axis=0)
        idx_u = np.argsort(diff)[-k:]        # largest positive (uncertain-up)
        idx_c = np.argsort(-diff)[-k:]       # largest negative -> confidence-up
        hit_u[idx_u] += 1
        hit_c[idx_c] += 1

    return hit_u / float(n_boot), hit_c / float(n_boot)


def permutation_pvalues_mean_diff(
    Xu: np.ndarray,
    Xc: np.ndarray,
    feature_idx: list[int],
    n_perm: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """
    Two-sided permutation p-values for mean-difference on a selected feature subset.
    """
    if n_perm <= 0 or not feature_idx:
        return {}
    X = np.concatenate([Xu, Xc], axis=0)
    n_u = Xu.shape[0]
    out: dict[str, float] = {}
    for fi in feature_idx:
        x = X[:, fi]
        obs = abs(Xu[:, fi].mean() - Xc[:, fi].mean())
        cnt = 0
        for _ in range(n_perm):
            perm = rng.permutation(x.shape[0])
            a = x[perm[:n_u]]
            b = x[perm[n_u:]]
            stat = abs(a.mean() - b.mean())
            if stat >= obs:
                cnt += 1
        out[str(int(fi))] = float((cnt + 1) / (n_perm + 1))
    return out


def summarize_top_table(title: str, rows: list[dict], n: int = 10, score_key: str = "score") -> None:
    print(f"\n  {title} (top-{min(n, len(rows))}):")
    print("    idx      score      dir")
    print("    ---------------------------")
    for r in rows[:n]:
        print(f"    {r['feature_idx']:<8} {r[score_key]:>8.4f}   {r['direction']}")


def analyze_layer(
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
    n_permutation: int,
    seed: int,
) -> tuple[dict, dict]:
    # Load SAE
    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device, dtype=sae_dtype)

    # Collect encoded features per sample
    z_certain = []
    z_uncertain = []
    z_all = []

    for sid in certain_ids:
        hs = torch.load(id_to_hs_dir[sid] / f"{sid}.pt", map_location="cpu", weights_only=False)
        h = hs[layer].to(dtype=torch.float32).unsqueeze(0).to(device=device)
        z = sae.encode(h).squeeze(0).detach().cpu().numpy().astype(np.float32)
        z_certain.append(z)
        z_all.append(z)

    for sid in uncertain_ids:
        hs = torch.load(id_to_hs_dir[sid] / f"{sid}.pt", map_location="cpu", weights_only=False)
        h = hs[layer].to(dtype=torch.float32).unsqueeze(0).to(device=device)
        z = sae.encode(h).squeeze(0).detach().cpu().numpy().astype(np.float32)
        z_uncertain.append(z)
        z_all.append(z)

    Xc = np.stack(z_certain, axis=0)  # [n_c, d_sae]
    Xu = np.stack(z_uncertain, axis=0)  # [n_u, d_sae]
    rng = np.random.default_rng(seed + int(layer))

    mean_c = np.nan_to_num(Xc.mean(axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    mean_u = np.nan_to_num(Xu.mean(axis=0), nan=0.0, posinf=0.0, neginf=0.0)

    # Method 2 VUF in SAE latent space (certain - uncertain)
    vuf_z = mean_c - mean_u

    # (A) mean-diff rankings
    diff_u_minus_c = mean_u - mean_c
    diff_c_minus_u = mean_c - mean_u
    top_uncertainty = topk_positive_indices(diff_u_minus_c, top_k)
    top_confidence = topk_positive_indices(diff_c_minus_u, top_k)

    rows_mean = []
    for i in topk_indices(diff_u_minus_c, top_k) + topk_indices(diff_c_minus_u, top_k):
        d = float(diff_u_minus_c[i])
        rows_mean.append({
            "feature_idx": int(i),
            "score": abs(d),
            "mean_diff_u_minus_c": d,
            "mean_certain": float(mean_c[i]),
            "mean_uncertain": float(mean_u[i]),
            "direction": "uncertain" if d > 0 else "certain",
        })
    # Deduplicate by idx and sort by |diff|
    dedup_mean = {}
    for r in rows_mean:
        dedup_mean[r["feature_idx"]] = r
    rows_mean = sorted(dedup_mean.values(), key=lambda r: r["score"], reverse=True)

    # (B) frequency-diff rankings (feature active if >0)
    freq_c = np.nan_to_num((Xc > 0).mean(axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    freq_u = np.nan_to_num((Xu > 0).mean(axis=0), nan=0.0, posinf=0.0, neginf=0.0)
    freq_diff = freq_u - freq_c

    rows_freq = []
    for i in topk_indices(freq_diff, top_k) + topk_indices(-freq_diff, top_k):
        d = float(freq_diff[i])
        rows_freq.append({
            "feature_idx": int(i),
            "score": abs(d),
            "freq_diff_u_minus_c": d,
            "freq_certain": float(freq_c[i]),
            "freq_uncertain": float(freq_u[i]),
            "direction": "uncertain" if d > 0 else "certain",
        })
    dedup_freq = {}
    for r in rows_freq:
        dedup_freq[r["feature_idx"]] = r
    rows_freq = sorted(dedup_freq.values(), key=lambda r: r["score"], reverse=True)

    # (C) Welch t-test + BH-FDR
    tt = ttest_ind(Xu, Xc, axis=0, equal_var=False, nan_policy="omit")
    pvals = np.asarray(tt.pvalue, dtype=np.float64)
    tvals = np.asarray(tt.statistic, dtype=np.float64)
    pvals = np.nan_to_num(pvals, nan=1.0, posinf=1.0, neginf=1.0)
    tvals = np.nan_to_num(tvals, nan=0.0, posinf=0.0, neginf=0.0)

    sig = benjamini_hochberg_mask(pvals, alpha=fdr_alpha)
    sig_u = np.where(sig & (tvals > 0))[0].tolist()   # uncertain up
    sig_c = np.where(sig & (tvals < 0))[0].tolist()   # certain up

    rows_t = []
    for i in np.argsort(np.abs(tvals))[-top_k:][::-1].tolist():
        d = float(diff_u_minus_c[i])
        rows_t.append({
            "feature_idx": int(i),
            "score": float(abs(tvals[i])),
            "t_stat": float(tvals[i]),
            "p_value": float(pvals[i]),
            "fdr_significant": bool(sig[i]),
            "direction": "uncertain" if d > 0 else "certain",
        })

    # (E) Effect size (Cohen's d)
    dvals = np.nan_to_num(cohens_d_uncertain_vs_certain(Xu, Xc), nan=0.0, posinf=0.0, neginf=0.0)
    rows_d = []
    for i in np.argsort(np.abs(dvals))[-top_k:][::-1].tolist():
        d = float(dvals[i])
        rows_d.append({
            "feature_idx": int(i),
            "score": abs(d),
            "cohens_d": d,
            "direction": "uncertain" if d > 0 else "certain",
        })

    # (F) Bootstrap top-k stability (selected frequency)
    stab_u, stab_c = bootstrap_topk_stability(Xu, Xc, top_k=top_k, n_boot=n_bootstrap, rng=rng)
    stable_u_idx = np.where(stab_u >= bootstrap_freq_threshold)[0].tolist()
    stable_c_idx = np.where(stab_c >= bootstrap_freq_threshold)[0].tolist()

    # (G) Permutation p-values on top-k candidates
    perm_candidates = sorted(set(top_uncertainty[:top_k] + top_confidence[:top_k]))
    perm_pvals = permutation_pvalues_mean_diff(
        Xu, Xc, feature_idx=perm_candidates, n_perm=n_permutation, rng=rng
    )

    # (D) Hedge projection ranking (friend's useful view)
    rows_hedge = []
    if hedge_vec is not None:
        h = hedge_vec.to(device=device, dtype=torch.float32)
        h = h / (h.norm() + 1e-8)
        w_dec = sae.W_dec.detach().to(device=device, dtype=torch.float32)
        hedge_scores = ((h @ w_dec.T) / (w_dec.norm(dim=1) + 1e-8)).detach().cpu().numpy()

        for i in np.argsort(np.abs(hedge_scores))[-top_k:][::-1].tolist():
            s = float(hedge_scores[i])
            rows_hedge.append({
                "feature_idx": int(i),
                "score": abs(s),
                "hedge_score": s,
                "direction": "uncertain" if s > 0 else "certain",
            })

    set_u = set(top_uncertainty)
    set_c = set(top_confidence)
    overlap_topk = len(set_u & set_c)

    # Consistency checks among ranking methods
    set_mean = set([r["feature_idx"] for r in rows_mean[:top_k]])
    set_t = set([r["feature_idx"] for r in rows_t[:top_k]])
    set_h = set([r["feature_idx"] for r in rows_hedge[:top_k]]) if rows_hedge else set()

    summary = {
        "layer": layer,
        "sae_id": sae_id,
        "n_certain": int(Xc.shape[0]),
        "n_uncertain": int(Xu.shape[0]),
        "d_sae": int(Xc.shape[1]),
        "vuf_z_norm": l2norm(vuf_z),
        "mean_c_vs_u_cosine": cos_sim(mean_c, mean_u),
        "top_k": top_k,
        "top_uncertainty_feature_idx": top_uncertainty,
        "top_confidence_feature_idx": top_confidence,
        "topk_overlap_count": int(overlap_topk),
        "topk_overlap_jaccard": topk_overlap(top_uncertainty, top_confidence),
        "uncertainty_only_topk_count": int(len(set_u - set_c)),
        "confidence_only_topk_count": int(len(set_c - set_u)),
        "ttest": {
            "fdr_alpha": fdr_alpha,
            "significant_total": int(sig.sum()),
            "uncertainty_up_count": int(len(sig_u)),
            "confidence_up_count": int(len(sig_c)),
            "uncertainty_up_top50_by_abs_t": sorted(sig_u, key=lambda i: abs(tvals[i]), reverse=True)[:50],
            "confidence_up_top50_by_abs_t": sorted(sig_c, key=lambda i: abs(tvals[i]), reverse=True)[:50],
        },
        "method_overlap_topk": {
            "mean_vs_t": int(len(set_mean & set_t)),
            "mean_vs_hedge": int(len(set_mean & set_h)) if rows_hedge else None,
            "t_vs_hedge": int(len(set_t & set_h)) if rows_hedge else None,
            "all_three": int(len(set_mean & set_t & set_h)) if rows_hedge else None,
        },
        "cohens_d": {
            "top_uncertainty_up_by_abs_d": [r["feature_idx"] for r in rows_d if r["direction"] == "uncertain"][:50],
            "top_confidence_up_by_abs_d": [r["feature_idx"] for r in rows_d if r["direction"] == "certain"][:50],
        },
        "bootstrap": {
            "n_bootstrap": int(n_bootstrap),
            "selection_frequency_threshold": float(bootstrap_freq_threshold),
            "stable_uncertainty_up_count": int(len(stable_u_idx)),
            "stable_confidence_up_count": int(len(stable_c_idx)),
            "stable_uncertainty_up_top50": sorted(stable_u_idx, key=lambda i: stab_u[i], reverse=True)[:50],
            "stable_confidence_up_top50": sorted(stable_c_idx, key=lambda i: stab_c[i], reverse=True)[:50],
        },
        "permutation": {
            "n_permutation": int(n_permutation),
            "topk_candidate_count": int(len(perm_candidates)),
            "candidate_perm_pvalues": perm_pvals,
            "candidate_perm_significant_005": [
                int(k) for k, v in perm_pvals.items() if v <= 0.05
            ],
        },
    }

    details = {
        "layer": layer,
        "sae_id": sae_id,
        "vuf_z": vuf_z,
        "mean_diff_rows": rows_mean,
        "frequency_diff_rows": rows_freq,
        "tstat_rows": rows_t,
        "cohens_d_rows": rows_d,
        "bootstrap_selection_frequency_uncertainty": stab_u.astype(np.float32),
        "bootstrap_selection_frequency_confidence": stab_c.astype(np.float32),
        "hedge_rows": rows_hedge,
    }

    # Print concise readable summary
    print(f"\n{'='*72}")
    print(f"Layer {layer} | {sae_id} | n_certain={Xc.shape[0]}, n_uncertain={Xu.shape[0]}")
    print(f"VUF_z norm={summary['vuf_z_norm']:.4f} | mean_c_vs_u_cosine={summary['mean_c_vs_u_cosine']:.4f}")
    print(
        f"Top-{top_k}: overlap={summary['topk_overlap_count']} "
        f"(jaccard={summary['topk_overlap_jaccard']:.4f}), "
        f"unc_only={summary['uncertainty_only_topk_count']}, conf_only={summary['confidence_only_topk_count']}"
    )
    print(
        f"t-test(FDR={fdr_alpha}): total={summary['ttest']['significant_total']}, "
        f"unc_up={summary['ttest']['uncertainty_up_count']}, conf_up={summary['ttest']['confidence_up_count']}"
    )
    print(
        f"bootstrap(n={n_bootstrap}, thr={bootstrap_freq_threshold:.2f}): "
        f"stable_unc={summary['bootstrap']['stable_uncertainty_up_count']}, "
        f"stable_conf={summary['bootstrap']['stable_confidence_up_count']}"
    )
    print(
        f"permutation(n={n_permutation}) on top-k candidates: "
        f"significant@0.05={len(summary['permutation']['candidate_perm_significant_005'])}"
    )

    summarize_top_table("Mean-diff features", rows_mean, n=10, score_key="score")
    summarize_top_table("Frequency-diff features", rows_freq, n=10, score_key="score")
    summarize_top_table("T-stat features", rows_t, n=10, score_key="score")
    summarize_top_table("Cohen's d features", rows_d, n=10, score_key="score")
    if rows_hedge:
        summarize_top_table("Hedge-projection features", rows_hedge, n=10, score_key="score")

    return summary, details


def build_aggregate(layer_summaries: list[dict], top_n: int = 100) -> dict:
    counters = {
        "uncertainty_top": {},
        "confidence_top": {},
        "ttest_uncertainty_up": {},
        "ttest_confidence_up": {},
    }

    for s in layer_summaries:
        for i in s["top_uncertainty_feature_idx"]:
            counters["uncertainty_top"][str(i)] = counters["uncertainty_top"].get(str(i), 0) + 1
        for i in s["top_confidence_feature_idx"]:
            counters["confidence_top"][str(i)] = counters["confidence_top"].get(str(i), 0) + 1
        for i in s["ttest"]["uncertainty_up_top50_by_abs_t"]:
            counters["ttest_uncertainty_up"][str(i)] = counters["ttest_uncertainty_up"].get(str(i), 0) + 1
        for i in s["ttest"]["confidence_up_top50_by_abs_t"]:
            counters["ttest_confidence_up"][str(i)] = counters["ttest_confidence_up"].get(str(i), 0) + 1

    def top_from_counter(counter: dict[str, int]) -> list[dict]:
        items = sorted(counter.items(), key=lambda kv: (-kv[1], int(kv[0])))
        return [{"feature_idx": int(k), "hits_across_layers": int(v)} for k, v in items[:top_n]]

    return {
        "uncertainty_top_most_frequent": top_from_counter(counters["uncertainty_top"]),
        "confidence_top_most_frequent": top_from_counter(counters["confidence_top"]),
        "ttest_uncertainty_up_most_frequent": top_from_counter(counters["ttest_uncertainty_up"]),
        "ttest_confidence_up_most_frequent": top_from_counter(counters["ttest_confidence_up"]),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merged SAE feature analysis (method 2 only)")
    p.add_argument("--repo_root", type=str, default=".")
    p.add_argument("--split", type=str, default="train", choices=["train", "test", "both"])
    p.add_argument("--release", type=str, default="mistral-7b-res-wg")
    p.add_argument("--sae_device", type=str, default="cpu")
    p.add_argument("--sae_dtype", type=str, default="float32", choices=("float32", "float16", "bfloat16"))
    p.add_argument("--vu_certain_max", type=float, default=VU_CERTAIN_MAX)
    p.add_argument("--vu_uncertain_min", type=float, default=VU_UNCERTAIN_MIN)
    p.add_argument("--top_k", type=int, default=64)
    p.add_argument("--fdr_alpha", type=float, default=0.05)
    p.add_argument("--n_bootstrap", type=int, default=200, help="Bootstrap repeats for stability")
    p.add_argument("--bootstrap_freq_threshold", type=float, default=0.8, help="Stable if selected in >= this fraction")
    p.add_argument("--n_permutation", type=int, default=500, help="Permutation repeats for top-k candidate p-values")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stats_json", type=str, default="sae_muc/artifacts/sae_feature_analysis_merged_stats.json")
    p.add_argument("--save_tensors_pt", type=str, default="", help="Optional .pt path to save full vectors")
    p.add_argument("--hedge_path", type=str, default="", help="Optional Hs_hedge path for projection ranking")
    p.add_argument(
        "--csv_root",
        type=str,
        default="",
        help="Optional root with train.csv/test.csv. If empty, auto-detects known locations.",
    )
    p.add_argument(
        "--hs_root",
        type=str,
        default="",
        help="Optional root with hidden states dirs train/ and test/ (id.pt files).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    ckpt = repo_root / "vuf_checkpoint"

    if args.csv_root:
        base_csv = Path(args.csv_root).resolve()
    else:
        ds_glob = sorted(repo_root.glob("datasets__nq_open*/Mistral-7B-Instruct-v0.3"))
        csv_candidates = [
            ckpt / "datasets" / "nq_open" / "Mistral-7B-Instruct-v0.3",
            repo_root / "datasets__nq_open" / "Mistral-7B-Instruct-v0.3",
            repo_root / "datasets" / "nq_open" / "Mistral-7B-Instruct-v0.3",
        ] + ds_glob
        base_csv = next((p for p in csv_candidates if (p / "train.csv").is_file() or (p / "test.csv").is_file()), csv_candidates[0])

    if args.hs_root:
        base_hs = Path(args.hs_root).resolve()
    else:
        hs_glob = sorted(repo_root.glob("datasets__nq_open*/Mistral-7B-Instruct-v0.3/pipeline_used_layers_last"))
        hs_candidates = [
            ckpt / "datasets" / "nq_open" / "Mistral-7B-Instruct-v0.3" / "pipeline_used_layers_last",
            repo_root / "vuf_checkpoint" / "datasets" / "nq_open" / "Mistral-7B-Instruct-v0.3" / "pipeline_used_layers_last",
        ] + hs_glob
        base_hs = next((p for p in hs_candidates if (p / "train").is_dir() or (p / "test").is_dir()), hs_candidates[0])

    if args.split == "both":
        split_names = ["train", "test"]
    else:
        split_names = [args.split]

    # Build merged VU map and per-id path map
    vu: dict[str, float] = {}
    id_to_hs_dir: dict[str, Path] = {}
    for sp in split_names:
        csv_path = base_csv / f"{sp}.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"CSV not found: {csv_path}. "
                "Pass --csv_root explicitly (folder with train.csv/test.csv)."
            )
        vu_sp = load_vu_labels(csv_path)
        hs_dir_sp = base_hs / sp
        if not hs_dir_sp.is_dir():
            raise FileNotFoundError(
                f"Hidden states dir not found: {hs_dir_sp}. "
                "Pass --hs_root explicitly (folder containing train/ and test/)."
            )
        for sid, v in vu_sp.items():
            vu[sid] = v
            id_to_hs_dir[sid] = hs_dir_sp

    certain_ids = [sid for sid, v in vu.items() if v <= args.vu_certain_max]
    uncertain_ids = [sid for sid, v in vu.items() if v >= args.vu_uncertain_min]

    if not certain_ids or not uncertain_ids:
        raise RuntimeError(
            f"No data for thresholds: certain<={args.vu_certain_max}, "
            f"uncertain>={args.vu_uncertain_min}. "
            f"Found certain={len(certain_ids)}, uncertain={len(uncertain_ids)}"
        )

    print(f"Split={args.split}: certain={len(certain_ids)}, uncertain={len(uncertain_ids)}")

    sample_id = next(iter(vu.keys()))
    raw = torch.load(id_to_hs_dir[sample_id] / f"{sample_id}.pt", map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise RuntimeError("Expected hidden-state files as dict[layer_idx -> vector].")
    stored_layers = sorted(raw.keys())

    mapping = dict(hf_layers_for_release(args.release))
    usable_layers = [l for l in stored_layers if l in mapping]
    if not usable_layers:
        raise RuntimeError(f"No overlap between stored layers {stored_layers} and mapping {sorted(mapping.keys())}")

    print(f"Stored layers: {stored_layers}")
    print(f"SAE-usable layers: {usable_layers}")

    hedge = None
    if args.hedge_path:
        h = torch.load(Path(args.hedge_path), map_location="cpu", weights_only=False)
        hedge = h.float()

    layer_summaries = []
    layer_details = []

    for layer in usable_layers:
        layer_hedge = None
        if hedge is not None:
            layer_hedge = hedge[layer]

        summary, details = analyze_layer(
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
            n_permutation=args.n_permutation,
            seed=args.seed,
        )
        layer_summaries.append(summary)
        layer_details.append(details)

    aggregate = build_aggregate(layer_summaries)

    result = {
        "meta": {
            "split": args.split,
            "release": args.release,
            "vu_certain_max": args.vu_certain_max,
            "vu_uncertain_min": args.vu_uncertain_min,
            "n_certain": len(certain_ids),
            "n_uncertain": len(uncertain_ids),
            "usable_layers": usable_layers,
            "top_k": args.top_k,
            "fdr_alpha": args.fdr_alpha,
        },
        "layers": layer_summaries,
        "aggregate": aggregate,
    }

    out_stats = (repo_root / args.stats_json).resolve()
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    with open(out_stats, "w") as f:
        json.dump(result, f, indent=2)

    if args.save_tensors_pt:
        out_pt = (repo_root / args.save_tensors_pt).resolve()
        out_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"meta": result["meta"], "layers": layer_details}, out_pt)
        print(f"Saved tensor details: {out_pt}")

    print(f"\nSaved stats: {out_stats}")

    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    for s in layer_summaries:
        print(
            f"Layer {s['layer']}: overlap@{s['top_k']}={s['topk_overlap_count']} "
            f"(J={s['topk_overlap_jaccard']:.4f}), "
            f"ttest sig={s['ttest']['significant_total']} "
            f"(unc={s['ttest']['uncertainty_up_count']}, conf={s['ttest']['confidence_up_count']})"
        )
    print("=" * 72)
