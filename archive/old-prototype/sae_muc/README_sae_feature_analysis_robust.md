# SAE Feature Analysis (Robust, Method 2)

This document describes `sae_muc/sae_feature_analysis_robust.py`.

The script performs **feature analysis in SAE latent space** for a fixed binary setup:
- **certain**: `VU <= 0.05`
- **uncertain**: `VU >= 0.90`

It does **not** train a classifier.  
It compares feature activations between groups and reports statistically robust feature sets.

---

## 1) What method is used

For each sample:
1. Load hidden state vector `h` from precomputed activations.
2. Encode with SAE: `z = SAE.encode(h)`.

For each SAE layer and feature:
- `mean_certain`
- `mean_uncertain`
- `diff = mean_uncertain - mean_certain`

Main latent direction for method 2:
- `VUF_z = mean(z_certain) - mean(z_uncertain)`

The script then computes several analysis views:
- **Mean-diff ranking** (top features by `|mean_uncertain - mean_certain|`)
- **Frequency-diff ranking** (top features by activation frequency gap, `z > 0`)
- **Welch t-test** for each feature (`uncertain` vs `certain`)
- **BH-FDR correction** over all SAE features (`fdr_alpha`, default `0.05`)
- Optional **hedge projection ranking** (`--hedge_path`)
- Cross-method overlaps (`mean vs t`, etc.)

---

## 2) Why this is “robust”

Compared to simple top-k by mean difference:
- keeps split control (`train`, `test`, `both`)
- avoids leakage by default unless you explicitly choose `--split both`
- uses **Welch t-test** (unequal variances)
- uses **multiple testing correction** (Benjamini-Hochberg FDR)

This is important because SAE space is high-dimensional (`d_sae = 65536`) and raw top-k can contain noise.

---

## 3) How to run

From repo root:

```bash
cd /Users/kirillfrolov/Documents/ML_2026/sae-muc
```

### Basic run (recommended, more uncertain examples)

```bash
python -m sae_muc.sae_feature_analysis_robust \
  --repo_root . \
  --split both \
  --release mistral-7b-res-wg \
  --sae_device cpu \
  --top_k 64 \
  --fdr_alpha 0.05 \
  --stats_json sae_muc/artifacts/sae_feature_analysis_robust_stats_both.json
```

### Full robust run (bootstrap + permutation + seed)

```bash
python -m sae_muc.sae_feature_analysis_robust \
  --repo_root . \
  --split both \
  --release mistral-7b-res-wg \
  --sae_device cpu \
  --top_k 64 \
  --fdr_alpha 0.05 \
  --n_bootstrap 200 \
  --bootstrap_freq_threshold 0.8 \
  --n_permutation 500 \
  --seed 42 \
  --stats_json sae_muc/artifacts/sae_feature_analysis_robust_stats_both.json
```

### Save detailed tensors too

```bash
python -m sae_muc.sae_feature_analysis_robust \
  --repo_root . \
  --split both \
  --release mistral-7b-res-wg \
  --sae_device cpu \
  --top_k 64 \
  --fdr_alpha 0.05 \
  --stats_json sae_muc/artifacts/sae_feature_analysis_robust_stats_both.json \
  --save_tensors_pt sae_muc/artifacts/sae_feature_analysis_robust_both.pt
```

### Add hedge projection view

```bash
python -m sae_muc.sae_feature_analysis_robust \
  --repo_root . \
  --split both \
  --release mistral-7b-res-wg \
  --sae_device cpu \
  --top_k 64 \
  --fdr_alpha 0.05 \
  --hedge_path vuf_checkpoint/calibration/outputs/merged/Mistral-7B-Instruct-v0.3/uncertainty/Hs_hedge_universal.pt \
  --stats_json sae_muc/artifacts/sae_feature_analysis_robust_stats_both.json
```

---

## 4) Input data expected

The script expects:
- CSV with VU labels:
  - `vuf_checkpoint/datasets/nq_open/Mistral-7B-Instruct-v0.3/{train|test}.csv`
- Hidden states:
  - `vuf_checkpoint/datasets/nq_open/Mistral-7B-Instruct-v0.3/pipeline_used_layers_last/{train|test}/{id}.pt`

For `release=mistral-7b-res-wg`, SAE is available for mapped layers (currently `15`, `23`).

---

## 5) Output files

Primary output:
- `--stats_json` (JSON summary)

Optional:
- `--save_tensors_pt` with full vectors and ranking rows

---

## 6) How to read results

Open `meta`:
- `n_certain`, `n_uncertain`: group sizes
- `top_k`, `fdr_alpha`: analysis settings

For each layer (`layers[]`):
- `vuf_z_norm`: norm of method-2 latent direction
- `mean_c_vs_u_cosine`: cosine between group means in latent space
- `top_uncertainty_feature_idx`: top-k features where uncertain mean is higher
- `top_confidence_feature_idx`: top-k features where certain mean is higher
- `topk_overlap_count/jaccard`: overlap between uncertain-topk and confidence-topk

### t-test block

Inside `ttest`:
- `significant_total`: number of FDR-significant features
- `uncertainty_up_count`: significant features with higher uncertain mean
- `confidence_up_count`: significant features with higher certain mean

Interpretation example:
- `significant_total=136, uncertainty_up=4, confidence_up=132`
  means 136 features differ significantly; most are certain-up, few are uncertain-up.

---

## 7) Practical guidance

For downstream steering/inspection:
- treat `top_*` lists as candidate sets
- treat `ttest + FDR` as reliability filter
- best stable set is usually:
  - features present in top-k
  - and FDR-significant
  - and direction-consistent (`uncertain` or `certain`)

With strong class imbalance, uncertainty-up significant features may be few.  
This is expected and does not invalidate mean-diff rankings.

---

## 8) What robust blocks change (and what they do not)

Robust blocks are **post-filters** over the same base analysis.

They **do not change**:
- `top_uncertainty_feature_idx` / `top_confidence_feature_idx`
- `ttest.significant_total`, `uncertainty_up_count`, `confidence_up_count`
- base layer statistics (`vuf_z_norm`, `mean_c_vs_u_cosine`)

They **add confidence checks**:
- `cohens_d`: effect size ranking (magnitude of difference)
- `bootstrap`: how often feature is selected across resampled datasets
- `permutation`: non-parametric p-values by label shuffling

Recommended interpretation order:
1. Use direction-aware top-k as candidate pool.
2. Apply `ttest + FDR` to remove likely false positives.
3. Keep features stable in bootstrap and significant in permutation.
4. Use intersection `t-test ∩ bootstrap ∩ permutation` as final robust core.

---

## 9) Reading robust blocks in JSON

Per layer:
- `bootstrap.stable_uncertainty_up_count`: count of uncertainty-up features with selection frequency >= threshold
- `bootstrap.stable_confidence_up_count`: same for confidence-up
- `permutation.candidate_perm_significant_005`: candidate features with permutation p <= 0.05

Quick practical recipe:
- **uncertainty core**: `ttest.uncertainty_up_top50_by_abs_t ∩ bootstrap.stable_uncertainty_up_top50 ∩ permutation.candidate_perm_significant_005`
- **confidence core**: `ttest.confidence_up_top50_by_abs_t ∩ bootstrap.stable_confidence_up_top50 ∩ permutation.candidate_perm_significant_005`

If these intersections are much smaller than raw top-k, robust blocks are filtering out unstable/noisy features.

---

## 10) Example (current `split=both` run)

For `sae_muc/artifacts/sae_feature_analysis_robust_stats_both.json`:

- Layer 15:
  - uncertainty core = 4 features (`24193, 25276, 56697, 64000`)
  - confidence core = 18 features
- Layer 23:
  - uncertainty core = 3 features (`30568, 36931, 44822`)
  - confidence core = 13 features

Interpretation:
- uncertainty-up findings are few but highly consistent (pass all robust filters),
- confidence-up findings are broader, but robust filtering still removes many less stable candidates.

