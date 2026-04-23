"""
Compute MUC evaluation metrics (Table 3 from the paper).

Metrics computed (before vs after MUC intervention):
  - Overall Hallucination Rate    — incorrect & not refused
  - Confident Hallucination Rate  — incorrect & not refused & low VU (below threshold)
  - Correctness Rate              — answer entailed by golden
  - Refusal Rate                  — model declines to answer
  - VU/SU Disagreement Rate       — VU and SE on opposite sides of threshold
  - Correlation (VU vs SE)        — Pearson r
  - VU for Incorrect answers      — mean VU when wrong
  - VU for Correct answers        — mean VU when right

Notes:
  - Correctness is approximated via substring/exact match (no LLM judge available).
  - Confident hallucination uses pre-intervention VU threshold (VU threshold from Kossen et al.).
  - VU after intervention is NOT available → confident hallu / VU metrics marked as N/A.
  - SE after is estimated from exact-string dedup of 10 MUC responses (approximate).
"""

import ast, json, os, re
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.environ.get("MUC_DATA_ROOT", os.path.join(BASE, "vuf_checkpoint"))

PATHS = dict(
    golden_csv  = os.environ.get("MUC_GOLDEN_CSV",
                    f"{DATA_ROOT}/datasets/nq_open/Mistral-7B-Instruct-v0.3/test.csv"),
    muc_jsonl   = os.environ.get("MUC_MUC_JSONL",
                    f"{DATA_ROOT}/calibration/outputs/nq_open/Mistral-7B-Instruct-v0.3/uncertainty/test/with_vufi_2_range(15,32)_1.0.jsonl"),
    base_jsonl  = os.environ.get("MUC_BASE_JSONL",
                    f"{DATA_ROOT}/sem_uncertainty/test_0.1.jsonl"),
    base_acc    = os.environ.get("MUC_BASE_ACC",
                    f"{DATA_ROOT}/sem_uncertainty/test_most_likely_acc.json"),
    base_refusal= os.environ.get("MUC_BASE_REFUSAL",
                    f"{DATA_ROOT}/sem_uncertainty/test_refusal_rate.json"),
    vu_judge    = os.environ.get("MUC_VU_JUDGE",
                    f"{DATA_ROOT}/verbal_uncertainty/Mistral-7B-Instruct-v0.3_nq_open_test_uncertainty_1.0_vu-llm-judge.json"),
    detection   = os.environ.get("MUC_DETECTION",
                    f"{DATA_ROOT}/detection/LR_outputs/nq_open/Mistral-7B-Instruct-v0.3/test_verbal_uncertainty_sentence_semantic_entropy.json"),
)

# ── Refusal patterns ──────────────────────────────────────────────────────────
REFUSAL_PATTERNS = [
    r"i('m| am) (not sure|unable|not aware|not certain|not confident)",
    r"i (don't|do not|cannot|can't) (know|have|provide|determine|say|confirm|tell)",
    r"i (don't|do not) have (enough|sufficient|the|any)",
    r"(i'm |i am )?(unable|not able) to (provide|answer|confirm|determine|verify)",
    r"i (lack|don't have) (the |)(information|knowledge|data|details)",
    r"(no|not) (specific |enough |)(information|knowledge|data) (is |are |)available",
    r"i('d| would) (need|require) more (information|context|details)",
    r"i apologize",
    r"i'm afraid i (can't|cannot|don't)",
]
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def is_refusal(text: str) -> bool:
    if not text:
        return True
    return bool(_REFUSAL_RE.search(text))


def parse_golden(raw: str) -> list[str]:
    """Parse golden answer field like \"['optic chiasm', 'chiasm']\" → list."""
    try:
        val = ast.literal_eval(raw)
        if isinstance(val, list):
            return [str(v).lower().strip() for v in val]
        return [str(val).lower().strip()]
    except Exception:
        return [str(raw).lower().strip()]


def is_correct(pred: str, golden_list: list[str]) -> bool:
    """Substring match: any golden answer contained in prediction (case-insensitive)."""
    if not pred:
        return False
    pred_l = pred.lower()
    return any(g in pred_l for g in golden_list if g)


def approx_se(responses: list[str]) -> float:
    """
    Approximate semantic entropy from list of responses via exact-string dedup.
    Groups identical strings → entropy of cluster distribution.
    Very rough (no NLI), but captures obvious diversity.
    """
    if not responses:
        return 0.0
    # normalize
    normed = [r.strip().lower()[:120] for r in responses if r]
    n = len(normed)
    if n == 0:
        return 0.0
    counts: dict[str, int] = {}
    for r in normed:
        counts[r] = counts.get(r, 0) + 1
    probs = np.array(list(counts.values())) / n
    # Shannon entropy (nats)
    return float(-np.sum(probs * np.log(probs + 1e-12)))


# ── VU threshold (Kossen et al. 2024 method: minimize sum of squared distances) ─
def find_vu_threshold(vu_scores: np.ndarray) -> float:
    best_t, best_cost = 0.5, float("inf")
    for t in np.linspace(0.0, 1.0, 201):
        low  = vu_scores[vu_scores <= t]
        high = vu_scores[vu_scores >  t]
        cost = (np.sum((low  - t)**2) if len(low)  > 0 else 0) + \
               (np.sum((high - t)**2) if len(high) > 0 else 0)
        if cost < best_cost:
            best_cost, best_t = cost, t
    return best_t


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(PATHS["golden_csv"])
golden = {row["question"].strip(): parse_golden(row["answer"])
          for _, row in df.iterrows()}
vu_csv = df.set_index("question")["verbal_uncertainty"].to_dict()
se_csv = df.set_index("question")["sentence_semantic_entropy"].to_dict()

with open(PATHS["muc_jsonl"]) as f:
    muc_rows = [json.loads(l) for l in f if l.strip()]

with open(PATHS["base_jsonl"]) as f:
    base_rows = [json.loads(l) for l in f if l.strip()]

with open(PATHS["base_acc"]) as f:
    base_acc_dict = json.load(f)

with open(PATHS["vu_judge"]) as f:
    vu_judge = json.load(f)   # list of 10 scores per question

with open(PATHS["detection"]) as f:
    detection = json.load(f)  # y_pred, y_prob

vu_before = np.array([np.mean(s) for s in vu_judge], dtype=np.float32)
se_before = df["sentence_semantic_entropy"].values.astype(np.float32)
y_pred    = np.array(detection["y_pred"])

VU_THRESH = find_vu_threshold(vu_before)
print(f"  VU threshold: {VU_THRESH:.4f}")


# ── BEFORE metrics ────────────────────────────────────────────────────────────
print("\nComputing BEFORE metrics...")

# Correctness from pre-computed json
base_correct = np.array([base_acc_dict[f"test_{i}"] for i in range(len(base_rows))],
                         dtype=np.float32)
base_correct_bin = (base_correct >= 0.5).astype(int)

# Refusal from baseline greedy answers
base_answers = [r["model answers"][0] if r["model answers"] else "" for r in base_rows]
base_refusals = np.array([is_refusal(a) for a in base_answers], dtype=int)

# Hallucination = incorrect AND not refused
base_hal     = ((base_correct_bin == 0) & (base_refusals == 0)).astype(int)
base_conf_hal = ((base_correct_bin == 0) & (base_refusals == 0) &
                 (vu_before < VU_THRESH)).astype(int)

# VU/SE disagreement: one above threshold, other below
SE_THRESH = np.log(10) / 2   # mid-point of [0, ln(10)]
base_vu_high = (vu_before >= VU_THRESH).astype(int)
base_se_high = (se_before >= SE_THRESH).astype(int)
base_disagree = (base_vu_high != base_se_high).astype(int)

# Pearson correlation VU vs SE
r_before, _ = pearsonr(vu_before, se_before)

# VU for incorrect / correct
base_vu_incorrect = vu_before[base_correct_bin == 0].mean()
base_vu_correct   = vu_before[base_correct_bin == 1].mean()

metrics_before = {
    "overall_hallucination_rate":    float(base_hal.mean()),
    "confident_hallucination_rate":  float(base_conf_hal.mean()),
    "correctness_rate":              float(base_correct_bin.mean()),
    "refusal_rate":                  float(base_refusals.mean()),
    "vu_su_disagreement_rate":       float(base_disagree.mean()),
    "correlation_vu_se":             float(r_before),
    "vu_for_incorrect":              float(base_vu_incorrect),
    "vu_for_correct":                float(base_vu_correct),
    "n_total":                       len(base_rows),
}


# ── AFTER metrics ─────────────────────────────────────────────────────────────
print("Computing AFTER metrics...")

# Match MUC rows to golden by question
# MUC has 1000 rows aligned with test.csv (same order)
muc_correct_list, muc_refusal_list, muc_se_list = [], [], []
muc_valid_mask = []   # whether row has a real answer (non-empty)

for i, row in enumerate(muc_rows):
    q = row["question"].strip()
    mla = row.get("most_likely_answer") or ""

    if not mla:
        # alpha=0 or skipped — keep original baseline answer
        mla = base_answers[i]

    gl = golden.get(q, [])
    correct = int(is_correct(mla, gl))
    refusal = int(is_refusal(mla))

    # SE after: from 10 responses if available, else use baseline SE
    responses = row.get("responses") or []
    if responses and any(r for r in responses):
        se_after_val = approx_se(responses)
    else:
        se_after_val = float(se_csv.get(q, se_before[i]))

    muc_correct_list.append(correct)
    muc_refusal_list.append(refusal)
    muc_se_list.append(se_after_val)
    muc_valid_mask.append(bool(row.get("most_likely_answer")))

muc_correct  = np.array(muc_correct_list, dtype=int)
muc_refusal  = np.array(muc_refusal_list, dtype=int)
muc_se_after = np.array(muc_se_list,      dtype=np.float32)
muc_valid    = np.array(muc_valid_mask,   dtype=bool)

muc_hal = ((muc_correct == 0) & (muc_refusal == 0)).astype(int)

# For SE-based metrics after (no VU after → disagreement/correlation = N/A)
SE_THRESH_AFTER = SE_THRESH
muc_se_high = (muc_se_after >= SE_THRESH_AFTER).astype(int)

metrics_after = {
    "overall_hallucination_rate":    float(muc_hal.mean()),
    "confident_hallucination_rate":  "N/A (VU after not available)",
    "correctness_rate":              float(muc_correct.mean()),
    "refusal_rate":                  float(muc_refusal.mean()),
    "vu_su_disagreement_rate":       "N/A (VU after not available)",
    "correlation_vu_se":             "N/A (VU after not available)",
    "vu_for_incorrect":              "N/A (VU after not available)",
    "vu_for_correct":                "N/A (VU after not available)",
    "n_total":                       len(muc_rows),
    "n_intervened":                  int(muc_valid.sum()),
    "n_skipped_alpha0":              int((~muc_valid).sum()),
    "approx_se_after_mean":          float(muc_se_after.mean()),
    "approx_se_before_mean":         float(se_before.mean()),
}

# ── Intervened-only metrics ───────────────────────────────────────────────────
# Compare only on the 612 questions that were actually intervened upon
muc_correct_int  = muc_correct[muc_valid]
muc_refusal_int  = muc_refusal[muc_valid]
muc_hal_int      = muc_hal[muc_valid]
base_correct_int = base_correct_bin[muc_valid]
base_refusal_int = base_refusals[muc_valid]
base_hal_int     = base_hal[muc_valid]

metrics_intervened = {
    "n": int(muc_valid.sum()),
    "before": {
        "overall_hallucination_rate": float(base_hal_int.mean()),
        "correctness_rate":           float(base_correct_int.mean()),
        "refusal_rate":               float(base_refusal_int.mean()),
    },
    "after": {
        "overall_hallucination_rate": float(muc_hal_int.mean()),
        "correctness_rate":           float(muc_correct_int.mean()),
        "refusal_rate":               float(muc_refusal_int.mean()),
    },
}


# ── Print results ─────────────────────────────────────────────────────────────
print()
print("=" * 65)
print(f"{'Metric':<40} {'Before':>10} {'After':>10}")
print("=" * 65)

TABLE = [
    ("Overall Hallucination Rate ↓",   "overall_hallucination_rate",   True),
    ("Confident Hallucination Rate ↓", "confident_hallucination_rate", True),
    ("Correctness Rate ↑",             "correctness_rate",             False),
    ("Refusal Rate",                   "refusal_rate",                 False),
    ("VU/SU Disagreement Rate ↓",      "vu_su_disagreement_rate",      True),
    ("Correlation VU↔SE ↑",            "correlation_vu_se",            False),
    ("VU for Incorrect ↑",             "vu_for_incorrect",             False),
    ("VU for Correct",                 "vu_for_correct",               False),
]

for label, key, lower_better in TABLE:
    b = metrics_before[key]
    a = metrics_after[key]
    b_str = f"{b:.3f}" if isinstance(b, float) else str(b)
    a_str = f"{a:.3f}" if isinstance(a, float) else str(a)
    print(f"  {label:<38} {b_str:>10} {a_str:>10}")

print()
print(f"  Approx SE before: {se_before.mean():.3f}  →  after: {muc_se_after.mean():.3f}")
print()
print("─" * 65)
print(f"  (* after = full 1000 rows, alpha=0 rows keep baseline answer)")
print()
print("  Intervened-only comparison (n=612, alpha>0):")
print(f"{'Metric':<40} {'Before':>10} {'After':>10}")
print("─" * 65)
for key in ["overall_hallucination_rate", "correctness_rate", "refusal_rate"]:
    b = metrics_intervened["before"][key]
    a = metrics_intervened["after"][key]
    print(f"  {key.replace('_',' ').title():<38} {b:.3f}      {a:.3f}")


# ── Save ──────────────────────────────────────────────────────────────────────
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
output = {
    "dataset": "NQ-Open",
    "model": "Mistral-7B-Instruct-v0.3",
    "muc_config": "iti_method=2, range(15,32), max_alpha=1.0, prompt=uncertainty",
    "vu_threshold": float(VU_THRESH),
    "se_threshold": float(SE_THRESH),
    "correctness_method": "substring match (approximate; paper uses LLM-judge)",
    "metrics_before": metrics_before,
    "metrics_after":  metrics_after,
    "metrics_intervened_only": metrics_intervened,
}
with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved → {OUT_DIR}/results.json")
