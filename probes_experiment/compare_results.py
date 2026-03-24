"""
Compare all experiments and write final comparison table to results/.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import json
import config

RESULTS_DIR = config.RESULTS_DIR

rows = []

for fname, label in [
    ("exp1_probes.json",         "exp1"),
    ("exp2_direct.json",         "exp2"),
    ("exp3_vuf_projection.json", "exp3"),
]:
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        print(f"  [skip] {fname} not found")
        continue
    with open(path) as f:
        d = json.load(f)

    if label == "exp1":
        # Best probe-predicted detector
        det = sorted(d["detection_probe"], key=lambda r: r["auroc"], reverse=True)[0]
        rows.append(dict(approach="Probe SE+VU → LR",         auroc=det["auroc"], acc=det["acc"],
                         details=f"L{det['layer']} {det['features']}", exp="exp1"))
        # Baseline
        for b in d["detection_baseline"]:
            rows.append(dict(approach=f"Raw {b['features']} → LR", auroc=b["auroc"], acc=b["acc"],
                             details="calculated SE/VU", exp="baseline"))
    elif label == "exp2":
        best = sorted(d["results"], key=lambda r: r["auroc"], reverse=True)[0]
        rows.append(dict(approach="Direct LR on hs",          auroc=best["auroc"], acc=best["acc"],
                         details=best["method"], exp="exp2"))
        # also best PCA
        pca = sorted([r for r in d["results"] if "PCA" in r["method"]],
                     key=lambda r: r["auroc"], reverse=True)
        if pca:
            rows.append(dict(approach="PCA + LR on hs",       auroc=pca[0]["auroc"], acc=pca[0]["acc"],
                             details=pca[0]["method"], exp="exp2"))
    elif label == "exp3":
        for r in sorted(d["results"], key=lambda r: r["auroc"], reverse=True)[:5]:
            rows.append(dict(approach=r["method"], auroc=r["auroc"], acc=r["acc"],
                             details="VUF projection", exp="exp3"))

rows.sort(key=lambda r: r["auroc"], reverse=True)

# Print table
header = f"{'Approach':<38} {'AUROC':<8} {'ACC':<8} {'Details'}"
sep    = "-" * 80
print("\n" + "="*80)
print("FINAL COMPARISON: All approaches")
print("="*80)
print(header)
print(sep)
for r in rows:
    print(f"  {r['approach']:<36} {r['auroc']:.4f}   {r['acc']:.4f}   {r['details']}")

# Write markdown
md_lines = [
    "# Comparison of Hallucination Detection Approaches",
    "",
    "NQ-Open, Mistral-7B-Instruct-v0.3, 500 train / 500 test",
    "",
    "| Approach | AUROC | ACC | Details |",
    "|---|---|---|---|",
]
for r in rows:
    md_lines.append(f"| {r['approach']} | {r['auroc']:.4f} | {r['acc']:.4f} | {r['details']} |")

md_lines += [
    "",
    "## Legend",
    "",
    "- **Raw SE+VU → LR** (baseline): uses pre-computed semantic entropy and verbal uncertainty scores (requires sampling 10 responses + LLM judge).",
    "- **Probe SE+VU → LR** (exp1): Ridge probes predict SE/VU from hidden states, then LR detects hallucinations. No sampling needed.",
    "- **Direct LR on hs** (exp2): LR directly on hidden states (4096 dims). No intermediate representations.",
    "- **PCA + LR on hs** (exp2): PCA dimensionality reduction then LR.",
    "- **VUF proj → LR/MLP** (exp3): Projects hidden states onto the Verbal Uncertainty Feature directions from the paper. Interpretable: each scalar measures model certainty at a given layer.",
]

os.makedirs(RESULTS_DIR, exist_ok=True)
md_path = os.path.join(RESULTS_DIR, "comparison_table.md")
with open(md_path, "w") as f:
    f.write("\n".join(md_lines))

json_path = os.path.join(RESULTS_DIR, "all_results.json")
with open(json_path, "w") as f:
    json.dump(rows, f, indent=2)

print(f"\nSaved {md_path}")
print(f"Saved {json_path}")
