from __future__ import annotations

from sae_muc.artifacts import StageManifest
from sae_muc.pipeline import run_all, run_stage


def test_run_all_runs_both_stages(fake_ctx):
    run_all(fake_ctx)
    assert fake_ctx.store.exists("samples.parquet")
    assert fake_ctx.store.exists("generations.parquet")


def test_run_stage_skips_on_cache_hit(fake_ctx):
    # First time: actually ran.
    ran_once = run_stage(fake_ctx, "prepare")
    assert ran_once is True
    # Second invocation with the same manifest: skipped.
    ran_twice = run_stage(fake_ctx, "prepare")
    assert ran_twice is False


def test_force_flag_ignores_cache(fake_ctx):
    run_stage(fake_ctx, "prepare")
    # Force re-run even though cache exists.
    assert run_stage(fake_ctx, "prepare", force=True) is True


def test_manifest_written_per_stage(fake_ctx):
    run_all(fake_ctx)
    m = StageManifest(fake_ctx.store.run_dir, "generate")
    assert m.exists()
    data = m.read()
    assert data["stage"] == "generate"
    assert data["outputs"] == ["generations.parquet"]


def test_unknown_stage_raises(fake_ctx):
    import pytest

    with pytest.raises(ValueError, match="Unknown stage"):
        run_stage(fake_ctx, "mystery")
