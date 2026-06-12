"""Concurrent judge pool: output order + per-call error isolation hold.

The judge / accuracy_judge stages run a continuous ThreadPoolExecutor with
`cfg.judge.concurrency` calls in flight. These tests use a deterministic,
thread-safe FAKE judge backend to assert that concurrency changes nothing
observable about the output beyond wall-clock time:
  * results at concurrency=4 are identical and in the same order as at 1;
  * a backend that raises on one specific prompt still isolates that row
    (NaN / ERROR) while the rest score correctly.
"""

from __future__ import annotations

import threading
import time

import pandas as pd

from sae_muc.models import Generation
from sae_muc.pipeline import accuracy_judge, generate, judge, prepare
from sae_muc.pipeline.context import build_context


def _ctx_with_concurrency(fake_cfg, fake_hf_rows, concurrency):
    """Build a fresh pipeline context whose judge.concurrency is overridden."""
    cfg = fake_cfg.model_copy(
        update={"judge": fake_cfg.judge.model_copy(update={"concurrency": concurrency})}
    )
    _rid, ctx = build_context(cfg)
    return ctx


class _DeterministicJudge:
    """Thread-safe fake backend: maps each prompt to a stable decimal in [0,1].

    Counts max in-flight calls so the test can confirm the pool actually
    runs work concurrently (not silently serialised).
    """

    def __init__(self, raise_on: str | None = None):
        self._raise_on = raise_on
        self._lock = threading.Lock()
        self._in_flight = 0
        self.max_in_flight = 0

    def _score(self, prompt: str) -> float:
        # Deterministic per-prompt score in [0, 1], depends on prompt text
        # so a reordering of outputs would be detectable.
        return (hash(prompt) % 1000) / 1000.0

    def generate(self, prompts, **_kw):
        out = []
        for prompt in prompts:
            with self._lock:
                self._in_flight += 1
                self.max_in_flight = max(self.max_in_flight, self._in_flight)
            try:
                # Brief hold so concurrent workers genuinely overlap (makes the
                # max_in_flight > 1 sanity check deterministic, not timing-luck).
                time.sleep(0.02)
                if self._raise_on is not None and self._raise_on in prompt:
                    raise RuntimeError("boom on the marked prompt")
                out.append([Generation(text=f"{self._score(prompt):.3f}", finish_reason="stop")])
            finally:
                with self._lock:
                    self._in_flight -= 1
        return out


def test_judge_concurrency_preserves_order_and_values(fake_cfg, fake_hf_rows):
    # Build the upstream artefacts once on a concurrency=1 context.
    ctx1 = _ctx_with_concurrency(fake_cfg, fake_hf_rows, 1)
    prepare.run(ctx1)
    generate.run(ctx1)
    ctx1.judge = _DeterministicJudge()
    judge.run(ctx1)
    seq = ctx1.store.load_parquet("judge_scores.parquet")

    # Re-run the SAME inputs through a concurrency=4 context. Reuse the run dir
    # by pointing the new context's store at the already-populated one.
    ctx4 = _ctx_with_concurrency(fake_cfg, fake_hf_rows, 4)
    ctx4.store = ctx1.store
    judge_backend = _DeterministicJudge()
    ctx4.judge = judge_backend
    judge.run(ctx4)
    par = ctx4.store.load_parquet("judge_scores.parquet")

    # Identical content AND identical row order.
    pd.testing.assert_frame_equal(seq, par)
    # The pool genuinely overlapped calls (sanity that we tested concurrency).
    assert judge_backend.max_in_flight > 1


def test_judge_error_isolation_under_concurrency(fake_cfg, fake_hf_rows):
    ctx = _ctx_with_concurrency(fake_cfg, fake_hf_rows, 4)
    prepare.run(ctx)
    generate.run(ctx)

    gens = ctx.store.load_parquet("generations.parquet")
    from sae_muc.pipeline._utils import select_vu_judge_rows

    judged = select_vu_judge_rows(gens)
    samples = ctx.store.load_parquet("samples.parquet").set_index("sample_id")
    # Pick one concrete answer text and make the judge blow up only on it.
    target_answer = judged.iloc[0]["text"]
    target_sid = judged.iloc[0]["sample_id"]

    ctx.judge = _DeterministicJudge(raise_on=target_answer)
    judge.run(ctx)
    df = ctx.store.load_parquet("judge_scores.parquet")

    # Exactly the rows whose answer == target_answer error out (NaN + ERROR raw).
    errored_mask = df["raw"].str.startswith("ERROR:")
    assert errored_mask.any()
    assert df.loc[errored_mask, "decisiveness"].isna().all()
    assert df.loc[errored_mask, "vu_score"].isna().all()
    # The target sample is among the errored ones.
    assert target_sid in set(df.loc[errored_mask, "sample_id"])
    # Every non-errored row scored fine (no spurious NaNs from the pool).
    assert df.loc[~errored_mask, "decisiveness"].notna().all()
    # And the stage completed for all rows (isolation, not abort).
    assert len(df) == len(judged)


def test_accuracy_judge_concurrency_preserves_order(fake_cfg, fake_hf_rows):
    ctx1 = _ctx_with_concurrency(fake_cfg, fake_hf_rows, 1)
    prepare.run(ctx1)
    generate.run(ctx1)
    # Accuracy parses yes/no, so return a deterministic yes/no per prompt.

    class _YesNoJudge(_DeterministicJudge):
        def generate(self, prompts, **_kw):
            out = []
            for prompt in prompts:
                with self._lock:
                    self._in_flight += 1
                    self.max_in_flight = max(self.max_in_flight, self._in_flight)
                try:
                    time.sleep(0.02)
                    label = "yes" if hash(prompt) % 2 == 0 else "no"
                    out.append([Generation(text=label, finish_reason="stop")])
                finally:
                    with self._lock:
                        self._in_flight -= 1
            return out

    ctx1.judge = _YesNoJudge()
    accuracy_judge.run(ctx1)
    seq = ctx1.store.load_parquet("accuracy.parquet")

    ctx4 = _ctx_with_concurrency(fake_cfg, fake_hf_rows, 4)
    ctx4.store = ctx1.store
    backend = _YesNoJudge()
    ctx4.judge = backend
    accuracy_judge.run(ctx4)
    par = ctx4.store.load_parquet("accuracy.parquet")

    pd.testing.assert_frame_equal(seq, par)
    assert backend.max_in_flight > 1
