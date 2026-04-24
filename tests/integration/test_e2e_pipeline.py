"""End-to-end pipeline tests on fakes: full run, resume, force, partial re-run.

Everything is in-memory — FakeBackend, FakeNLIBackend, and monkey-patched
HF dataset loader. Full pipeline: prepare → generate → judge →
accuracy_judge → semantic_entropy → hidden_states → vuf → detect →
intervene → evaluate.
"""

from __future__ import annotations

import logging

from sae_muc.artifacts import StageManifest
from sae_muc.pipeline import STAGES, run_all, run_stage


# Artefacts we expect to exist after a full pipeline run.
_EXPECTED_ARTEFACTS = (
    "samples.parquet",
    "generations.parquet",
    "judge_scores.parquet",
    "accuracy.parquet",
    "semantic_entropy.parquet",
    "hidden_states/meta.parquet",
    "vuf/meta.parquet",
    "detection.parquet",
    "detection_metrics.json",
    "intervention/meta.parquet",
    "metrics.json",
)


def test_full_pipeline_end_to_end_writes_all_artefacts(fake_ctx):
    run_all(fake_ctx)
    for rel in _EXPECTED_ARTEFACTS:
        assert fake_ctx.store.exists(rel), f"missing {rel!r}"
    for name in STAGES:
        assert StageManifest(fake_ctx.store.run_dir, name).exists(), f"no manifest for {name!r}"


def test_second_run_skips_every_stage(fake_ctx, caplog):
    caplog.set_level(logging.INFO, logger="sae_muc.pipeline.runner")
    run_all(fake_ctx)

    caplog.clear()
    run_all(fake_ctx)

    skipped = [r for r in caplog.records if r.getMessage().startswith("[skip]")]
    assert len(skipped) == len(STAGES), (
        f"expected every stage ({len(STAGES)}) to report [skip]; got {len(skipped)}: "
        f"{[r.getMessage() for r in skipped]}"
    )


def test_force_all_reruns_every_stage(fake_ctx, caplog):
    caplog.set_level(logging.INFO, logger="sae_muc.pipeline.runner")
    run_all(fake_ctx)

    caplog.clear()
    run_all(fake_ctx, force_all=True)

    skipped = [r for r in caplog.records if r.getMessage().startswith("[skip]")]
    assert skipped == []
    ran = [r for r in caplog.records if r.getMessage().startswith("==>")]
    assert len(ran) == len(STAGES)


def test_deleting_an_output_triggers_that_stage_to_rerun(fake_ctx):
    run_all(fake_ctx)

    # Remove an intermediate artefact; manifest still claims it exists.
    (fake_ctx.store.run_dir / "detection.parquet").unlink()
    m = StageManifest(fake_ctx.store.run_dir, "detect")
    assert m.exists()
    assert not m.should_skip()

    # Re-running just detect restores the file without touching upstream outputs.
    ran = run_stage(fake_ctx, "detect")
    assert ran is True
    assert fake_ctx.store.exists("detection.parquet")


def test_force_single_stage_does_not_invalidate_others(fake_ctx):
    run_all(fake_ctx)

    # Snapshot the generate manifest's timestamp — it should not change when we
    # force-rerun a downstream stage.
    gen_manifest_before = StageManifest(fake_ctx.store.run_dir, "generate").read()

    ran = run_stage(fake_ctx, "evaluate", force=True)
    assert ran is True

    gen_manifest_after = StageManifest(fake_ctx.store.run_dir, "generate").read()
    assert gen_manifest_before["timestamp"] == gen_manifest_after["timestamp"]


def test_partial_pipeline_then_resume_finishes_remaining(fake_ctx):
    # Run only the first two stages.
    assert run_stage(fake_ctx, "prepare") is True
    assert run_stage(fake_ctx, "generate") is True

    # run_all should pick up where prepare + generate left off, skip them, run the rest.
    run_all(fake_ctx)

    # Every artefact now present; every manifest written.
    for rel in _EXPECTED_ARTEFACTS:
        assert fake_ctx.store.exists(rel), f"missing {rel!r}"
