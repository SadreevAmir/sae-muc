from __future__ import annotations

import math

import pandas as pd
import pytest

from sae_muc.pipeline import evaluate
from sae_muc.pipeline.evaluate import _compute_metrics, _pearson


# ------------- _pearson ----------------


def test_pearson_perfect_positive():
    import numpy as np

    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([2.0, 4.0, 6.0, 8.0])
    assert _pearson(x, y) == pytest.approx(1.0)


def test_pearson_constant_returns_nan():
    import numpy as np

    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 1.0, 1.0])
    assert math.isnan(_pearson(x, y))


# ------------- _compute_metrics ----------------


def _frame(rows):
    return pd.DataFrame(rows)


def test_metrics_on_mixed_set():
    df = _frame(
        [
            # 3 correct, low VU, low SE
            *[{"sample_id": f"c{i}", "vu": 0.1, "se": 0.1, "is_correct": True, "is_refusal": False} for i in range(3)],
            # 3 hallucinated, low VU (= confident hallucination), high SE
            *[{"sample_id": f"h{i}", "vu": 0.2, "se": 1.5, "is_correct": False, "is_refusal": False} for i in range(3)],
            # 1 refusal (is_correct=False but refusal=True → excluded from hallucination count)
            {"sample_id": "r0", "vu": 0.9, "se": 0.0, "is_correct": False, "is_refusal": True},
        ]
    )
    m = _compute_metrics(df, vu_threshold=0.5, su_threshold=1.0)
    assert m["n_total"] == 7
    assert m["n_correct"] == 3
    assert m["n_hallucinated"] == 3
    assert m["n_refusal"] == 1
    assert m["n_confident_hallucinated"] == 3   # VU=0.2 < 0.5 and hallucinated
    assert m["correct_rate"] == pytest.approx(3 / 7)
    assert m["hallucination_rate"] == pytest.approx(3 / 7)
    assert m["refusal_rate"] == pytest.approx(1 / 7)
    assert m["vu_correct_mean"] == pytest.approx(0.1)
    assert m["vu_incorrect_mean"] == pytest.approx(0.2)
    # Disagreement counts per-row:
    #   correct (3x): vu=0.1 low, se=0.1 low → agree
    #   hallucinated (3x): vu=0.2 low, se=1.5 high → disagree
    #   refusal (1x): vu=0.9 high, se=0.0 low → disagree
    # Total: 4 / 7.
    assert m["vu_su_disagreement_rate"] == pytest.approx(4 / 7)


def test_metrics_empty_frame():
    m = _compute_metrics(_frame([]), vu_threshold=0.5, su_threshold=1.0)
    assert m == {"n_total": 0, "empty": True}


def test_metrics_su_threshold_defaults_to_median():
    df = _frame(
        [
            {"sample_id": "a", "vu": 0.1, "se": 0.0, "is_correct": True, "is_refusal": False},
            {"sample_id": "b", "vu": 0.1, "se": 1.0, "is_correct": True, "is_refusal": False},
            {"sample_id": "c", "vu": 0.1, "se": 2.0, "is_correct": True, "is_refusal": False},
        ]
    )
    m = _compute_metrics(df, vu_threshold=0.5, su_threshold=None)
    assert m["thresholds"]["su"] == pytest.approx(1.0)  # median of [0, 1, 2]


# ------------- stage ----------------


def _seed_evaluate_artefacts(fake_ctx):
    """Hand-craft minimal inputs so evaluate computes meaningful metrics."""
    samples_df = pd.DataFrame(
        {
            "sample_id": [f"q{i}" for i in range(6)],
            "question": [f"q{i}" for i in range(6)],
            "gold_answers": [["gold"]] * 6,
        }
    )
    fake_ctx.store.save_parquet("samples.parquet", samples_df)

    gen_rows = [
        {"sample_id": "q0", "kind": "greedy", "gen_idx": 0, "text": "gold", "finish_reason": "stop"},
        {"sample_id": "q1", "kind": "greedy", "gen_idx": 0, "text": "gold", "finish_reason": "stop"},
        {"sample_id": "q2", "kind": "greedy", "gen_idx": 0, "text": "wrong", "finish_reason": "stop"},
        {"sample_id": "q3", "kind": "greedy", "gen_idx": 0, "text": "wrong", "finish_reason": "stop"},
        {"sample_id": "q4", "kind": "greedy", "gen_idx": 0, "text": "I don't know.", "finish_reason": "stop"},
        {"sample_id": "q5", "kind": "greedy", "gen_idx": 0, "text": "wrong", "finish_reason": "stop"},
    ]
    fake_ctx.store.save_parquet("generations.parquet", pd.DataFrame(gen_rows))

    fake_ctx.store.save_parquet(
        "accuracy.parquet",
        pd.DataFrame(
            [
                {"sample_id": "q0", "is_correct": True, "raw": "yes"},
                {"sample_id": "q1", "is_correct": True, "raw": "yes"},
                {"sample_id": "q2", "is_correct": False, "raw": "no"},
                {"sample_id": "q3", "is_correct": False, "raw": "no"},
                {"sample_id": "q4", "is_correct": False, "raw": "no"},
                {"sample_id": "q5", "is_correct": False, "raw": "no"},
            ]
        ),
    )

    # Judge scores: greedy VU selects refusals (q4 is the refusal → greedy_vu 0.95),
    # sampled VU stands in for the model's hedging on the N samples.
    judge_rows = []
    greedy_vus = {"q0": 0.2, "q1": 0.2, "q2": 0.2, "q3": 0.2, "q4": 0.95, "q5": 0.2}
    for sid, vu in [("q0", 0.9), ("q1", 0.8), ("q2", 0.2), ("q3", 0.3), ("q4", 0.9), ("q5", 0.2)]:
        judge_rows.append(
            {"sample_id": sid, "kind": "greedy", "gen_idx": 0,
             "decisiveness": 1.0 - greedy_vus[sid], "vu_score": greedy_vus[sid],
             "raw": str(greedy_vus[sid])}
        )
        for j in range(3):
            judge_rows.append(
                {"sample_id": sid, "kind": "sample", "gen_idx": j,
                 "decisiveness": 1.0 - vu, "vu_score": vu, "raw": str(vu)}
            )
    fake_ctx.store.save_parquet("judge_scores.parquet", pd.DataFrame(judge_rows))

    fake_ctx.store.save_parquet(
        "semantic_entropy.parquet",
        pd.DataFrame(
            [
                {"sample_id": "q0", "semantic_entropy": 0.0, "n_clusters": 1, "n_samples": 3},
                {"sample_id": "q1", "semantic_entropy": 0.1, "n_clusters": 1, "n_samples": 3},
                {"sample_id": "q2", "semantic_entropy": 1.5, "n_clusters": 3, "n_samples": 3},
                {"sample_id": "q3", "semantic_entropy": 1.5, "n_clusters": 3, "n_samples": 3},
                {"sample_id": "q4", "semantic_entropy": 0.0, "n_clusters": 1, "n_samples": 3},
                {"sample_id": "q5", "semantic_entropy": 1.5, "n_clusters": 3, "n_samples": 3},
            ]
        ),
    )


def test_evaluate_stage_writes_metrics_json(fake_ctx):
    _seed_evaluate_artefacts(fake_ctx)
    outputs = evaluate.run(fake_ctx)
    assert outputs == ["metrics.json"]

    m = fake_ctx.store.load_json("metrics.json")
    # 6 questions: 2 correct (q0,q1), 3 hallucinated (q2,q3,q5), 1 refusal (q4).
    assert m["n_total"] == 6
    assert m["n_correct"] == 2
    assert m["n_hallucinated"] == 3
    assert m["n_refusal"] == 1
    # Default vu_threshold=0.5. Hallucinated with VU<0.5: q2 (0.2), q3 (0.3), q5 (0.2) → 3.
    assert m["n_confident_hallucinated"] == 3
    # VU means
    assert m["vu_correct_mean"] == pytest.approx((0.9 + 0.8) / 2)
    assert m["vu_incorrect_mean"] == pytest.approx((0.2 + 0.3 + 0.2) / 3)
