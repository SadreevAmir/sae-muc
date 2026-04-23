"""Shared runtime context passed into every pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass

from sae_muc.artifacts import ArtifactStore, make_run
from sae_muc.config import ExperimentConfig
from sae_muc.models import LLMBackend, build_llm_backend


@dataclass
class PipelineContext:
    cfg: ExperimentConfig
    store: ArtifactStore
    llm: LLMBackend
    judge: LLMBackend


def build_context(cfg: ExperimentConfig, *, run_id: str | None = None) -> tuple[str, PipelineContext]:
    """Resolve run_id, prepare the run directory, and instantiate backends.

    LLM backends are instantiated eagerly. For `fake` and `openrouter` this
    is cheap; for `hf_local` it will eventually load model weights, but that
    backend is a stub until the hidden_states stage lands.
    """
    rid, store = make_run(cfg, run_id=run_id)
    llm = build_llm_backend(cfg.model.provider, cfg.model.name, dtype=cfg.model.dtype)
    judge = build_llm_backend(
        cfg.judge.provider, cfg.judge.model, max_retries=cfg.judge.max_retries
    )
    return rid, PipelineContext(cfg=cfg, store=store, llm=llm, judge=judge)
