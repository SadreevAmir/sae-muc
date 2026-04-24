"""Orchestration: STAGES registry + stage-level runner with manifest-based skip."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

from sae_muc.artifacts import StageManifest
from sae_muc.pipeline import (
    generate,
    hidden_states,
    judge,
    prepare,
    semantic_entropy,
    vuf,
)
from sae_muc.pipeline.context import PipelineContext

log = logging.getLogger(__name__)

StageFn = Callable[[PipelineContext], list[str]]


STAGES: dict[str, StageFn] = {
    "prepare": prepare.run,
    "generate": generate.run,
    "judge": judge.run,
    "semantic_entropy": semantic_entropy.run,
    "hidden_states": hidden_states.run,
    "vuf": vuf.run,
    # detect / intervene / evaluate land in upcoming commits.
}


def run_stage(
    ctx: PipelineContext,
    stage_name: str,
    *,
    force: bool = False,
) -> bool:
    """Run one named stage. Return True if the stage actually ran, False if skipped."""
    if stage_name not in STAGES:
        raise ValueError(f"Unknown stage: {stage_name!r}. Known: {sorted(STAGES)}")
    stage_fn = STAGES[stage_name]
    manifest = StageManifest(ctx.store.run_dir, stage_name)
    if not force and manifest.should_skip():
        log.info("stage %s: skipped (cached)", stage_name)
        return False
    log.info("stage %s: running", stage_name)
    outputs = stage_fn(ctx)
    manifest.write(outputs=outputs)
    log.info("stage %s: done (%d outputs)", stage_name, len(outputs))
    return True


def run_all(
    ctx: PipelineContext,
    *,
    force: Iterable[str] = (),
    force_all: bool = False,
) -> None:
    """Run every registered stage in order. `force` names stages to ignore cache for."""
    force_set = set(force)
    for stage_name in STAGES:
        run_stage(ctx, stage_name, force=force_all or stage_name in force_set)
