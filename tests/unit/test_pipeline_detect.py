from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sae_muc.pipeline import detect


# ------------- stage with hand-crafted artefacts ----------------


def _seed_detect_artefacts(
    fake_ctx,
    *,
    n: int = 20,
    n_hallucinated: int = 10,
    n_refusal: int = 0,
) -> None:
    """Populate the artefacts detect.run() needs with a well-defined labelling.

    First `n_hallucinated` samples are labelled hallucinated (low sample VU);
    the next `n - n_hallucinated - n_refusal` are correct (high sample VU);
    the last `n_refusal` samples have greedy VU ≥ refusal_vu_threshold (0.85).
    """
    sample_ids = [f"q{i}" for i in range(n)]

    samples = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "question": [f"q{i}" for i in range(n)],
            "gold_answers": [["gold"] for _ in range(n)],
        }
    )
    fake_ctx.store.save_parquet("samples.parquet", samples)

    gen_rows = []
    accuracy_rows = []
    for i, sid in enumerate(sample_ids):
        is_hallucinated = i < n_hallucinated
        is_ref = i >= n - n_refusal
        if is_ref:
            greedy_text = "(refusal)"
            is_correct = False  # arbitrary — refusals are dropped downstream
        elif is_hallucinated:
            greedy_text = "wrong answer"
            is_correct = False
        else:
            greedy_text = "gold"
            is_correct = True
        gen_rows.append(
            {"sample_id": sid, "kind": "greedy", "gen_idx": 0,
             "text": greedy_text, "finish_reason": "stop"}
        )
        accuracy_rows.append({"sample_id": sid, "is_correct": is_correct, "raw": "yes/no"})
    fake_ctx.store.save_parquet("generations.parquet", pd.DataFrame(gen_rows))
    fake_ctx.store.save_parquet("accuracy.parquet", pd.DataFrame(accuracy_rows))

    # Judge scores:
    #  - greedy VU: 0.95 for refusal, 0.2 otherwise (below 0.85 threshold)
    #  - sample VU: 0.1 for hallucinated, 0.9 for correct (separates classes
    #    for the LR detector training)
    judge_rows = []
    for i, sid in enumerate(sample_ids):
        is_ref = i >= n - n_refusal
        greedy_vu = 0.95 if is_ref else 0.2
        judge_rows.append(
            {"sample_id": sid, "kind": "greedy", "gen_idx": 0,
             "decisiveness": 1.0 - greedy_vu, "vu_score": greedy_vu, "raw": str(greedy_vu)}
        )
        sample_vu = 0.1 if i < n_hallucinated else 0.9
        for j in range(3):
            judge_rows.append(
                {"sample_id": sid, "kind": "sample", "gen_idx": j,
                 "decisiveness": 1.0 - sample_vu, "vu_score": sample_vu, "raw": str(sample_vu)}
            )
    fake_ctx.store.save_parquet("judge_scores.parquet", pd.DataFrame(judge_rows))

    # Semantic entropy — hallucinated → high SE; correct → low SE.
    se_rows = []
    for i, sid in enumerate(sample_ids):
        se_val = 1.5 if i < n_hallucinated else 0.1
        se_rows.append(
            {"sample_id": sid, "semantic_entropy": se_val, "n_clusters": 3, "n_samples": 3}
        )
    fake_ctx.store.save_parquet("semantic_entropy.parquet", pd.DataFrame(se_rows))


def test_detect_writes_parquet_and_metrics(fake_ctx):
    _seed_detect_artefacts(fake_ctx, n=20, n_hallucinated=10, n_refusal=0)
    outputs = detect.run(fake_ctx)
    assert outputs == ["detection.parquet", "detection_metrics.json"]

    df = fake_ctx.store.load_parquet("detection.parquet")
    assert len(df) == 20
    # Three prediction columns populated (not NaN) because the fit ran.
    for name in ("verbal", "semantic", "combined"):
        assert df[f"prob_hallucinate_{name}"].notna().all()

    metrics = fake_ctx.store.load_json("detection_metrics.json")
    assert metrics["n_trainable"] == 20
    assert metrics["n_refusal"] == 0
    for name in ("verbal", "semantic", "combined"):
        assert "train" in metrics[name] and "test" in metrics[name]
        assert not np.isnan(metrics[name]["train"]["auroc"])


def test_detect_combined_separates_linearly_separable_classes(fake_ctx):
    _seed_detect_artefacts(fake_ctx, n=20, n_hallucinated=10, n_refusal=0)
    detect.run(fake_ctx)
    metrics = fake_ctx.store.load_json("detection_metrics.json")
    # With a clean linear signal both train and test AUROC should hit 1.
    for name in ("verbal", "semantic", "combined"):
        assert metrics[name]["train"]["auroc"] == pytest.approx(1.0)
        assert metrics[name]["test"]["auroc"] == pytest.approx(1.0)


def test_detect_excludes_refusals_from_trainable(fake_ctx):
    _seed_detect_artefacts(fake_ctx, n=20, n_hallucinated=10, n_refusal=5)
    detect.run(fake_ctx)
    metrics = fake_ctx.store.load_json("detection_metrics.json")
    # 20 total, 5 refusals → 15 trainable.
    assert metrics["n_refusal"] == 5
    assert metrics["n_trainable"] == 15
    df = fake_ctx.store.load_parquet("detection.parquet")
    assert df["is_refusal"].sum() == 5
    # Predictions are filled for refusals too.
    assert df["prob_hallucinate_combined"].notna().all()


def test_detect_handles_insufficient_data(fake_ctx):
    _seed_detect_artefacts(fake_ctx, n=3, n_hallucinated=0, n_refusal=0)
    detect.run(fake_ctx)
    metrics = fake_ctx.store.load_json("detection_metrics.json")
    assert metrics.get("skipped") == "insufficient data"
    # No fit happened → prediction columns are NaN.
    df = fake_ctx.store.load_parquet("detection.parquet")
    for name in ("verbal", "semantic", "combined"):
        assert df[f"prob_hallucinate_{name}"].isna().all()


def test_detect_seed_makes_split_reproducible(fake_ctx):
    _seed_detect_artefacts(fake_ctx, n=20, n_hallucinated=10, n_refusal=0)
    detect.run(fake_ctx)
    m1 = fake_ctx.store.load_json("detection_metrics.json")
    # Re-run with force=True to regenerate the detection artefact.
    from sae_muc.pipeline import run_stage

    run_stage(fake_ctx, "detect", force=True)
    m2 = fake_ctx.store.load_json("detection_metrics.json")
    # Same seed → identical metrics.
    assert m1 == m2
