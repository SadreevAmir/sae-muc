"""Orchestration: STAGES registry + stage-level runner with manifest-based skip."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable

from sae_muc.artifacts import StageManifest
from sae_muc.pipeline import (
    accuracy_judge,
    accuracy_judge_post,
    detect,
    diagnostics,
    evaluate,
    evaluate_post,
    generate,
    hidden_states,
    intervene,
    judge,
    judge_post,
    prepare,
    sae_features,
    semantic_entropy,
    semantic_entropy_post,
    vuf,
)
from sae_muc.pipeline.context import PipelineContext

log = logging.getLogger(__name__)

StageFn = Callable[[PipelineContext], list[str]]


STAGES: dict[str, StageFn] = {
    "prepare": prepare.run,
    "generate": generate.run,
    "judge": judge.run,
    "accuracy_judge": accuracy_judge.run,
    "semantic_entropy": semantic_entropy.run,
    "hidden_states": hidden_states.run,
    "vuf": vuf.run,
    "sae_features": sae_features.run,
    "detect": detect.run,
    "intervene": intervene.run,
    "evaluate": evaluate.run,
    # Post-intervention: re-score every intervention variant, then produce
    # the before/after comparison table (paper Tab.3).
    "judge_post": judge_post.run,
    "accuracy_judge_post": accuracy_judge_post.run,
    "semantic_entropy_post": semantic_entropy_post.run,
    "evaluate_post": evaluate_post.run,
    # Sidecar: perplexity / KL drift per intervention variant. Pure additive —
    # safe to skip via `stages.diagnostics.enabled=false` or by name.
    "diagnostics": diagnostics.run,
}


def _fmt_duration(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{int(s):02d}s"


def _fmt_outputs(outputs: list[str]) -> str:
    if not outputs:
        return "no outputs"
    if len(outputs) == 1:
        return outputs[0]
    return f"{len(outputs)} outputs (first: {outputs[0]})"


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
        log.info("[skip] %s (cached)", stage_name)
        return False
    log.info("==> %s", stage_name)
    start = time.monotonic()
    try:
        outputs = stage_fn(ctx)
    except Exception:
        elapsed = time.monotonic() - start
        log.info("[fail] %s after %s", stage_name, _fmt_duration(elapsed))
        raise
    elapsed = time.monotonic() - start
    manifest.write(outputs=outputs)
    log.info("[ok] %s · %s · %s", stage_name, _fmt_duration(elapsed), _fmt_outputs(outputs))
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
