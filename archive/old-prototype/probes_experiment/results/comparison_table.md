# Comparison of Hallucination Detection Approaches

NQ-Open, Mistral-7B-Instruct-v0.3, 500 train / 500 test

| Approach | AUROC | ACC | Details |
|---|---|---|---|
| Raw raw SE+VU → LR | 0.7770 | 0.7100 | calculated SE/VU |
| Raw raw SE only → LR | 0.7488 | 0.6860 | calculated SE/VU |
| Raw raw VU only → LR | 0.7359 | 0.6780 | calculated SE/VU |
| Probe SE+VU → LR | 0.7337 | 0.6720 | L17 SE+VU |
| Direct LR on hs | 0.7243 | 0.6740 | PCA32+LR_L17 |
| PCA + LR on hs | 0.7243 | 0.6740 | PCA32+LR_L17 |
| MLP_h128_hs_L17 | 0.7086 | 0.6440 | VUF projection |
| MLP_h128_hs_L15 | 0.7061 | 0.6520 | VUF projection |
| MLP_h64_hs_L15 | 0.7028 | 0.6500 | VUF projection |
| MLP_h64_hs_L17 | 0.7021 | 0.6520 | VUF projection |
| MLP_h64_hs_L18 | 0.7018 | 0.6440 | VUF projection |

## Legend

- **Raw SE+VU → LR** (baseline): uses pre-computed semantic entropy and verbal uncertainty scores (requires sampling 10 responses + LLM judge).
- **Probe SE+VU → LR** (exp1): Ridge probes predict SE/VU from hidden states, then LR detects hallucinations. No sampling needed.
- **Direct LR on hs** (exp2): LR directly on hidden states (4096 dims). No intermediate representations.
- **PCA + LR on hs** (exp2): PCA dimensionality reduction then LR.
- **VUF proj → LR/MLP** (exp3): Projects hidden states onto the Verbal Uncertainty Feature directions from the paper. Interpretable: each scalar measures model certainty at a given layer.