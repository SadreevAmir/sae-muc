"""Shared runtime context passed into every pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass

from sae_muc.artifacts import ArtifactStore, make_run
from sae_muc.config import ExperimentConfig
from sae_muc.models import LLMBackend, NLIBackend, build_llm_backend, build_nli_backend
from sae_muc.models.sae import SAEBackend, build_sae_backend


@dataclass
class PipelineContext:
    cfg: ExperimentConfig
    store: ArtifactStore
    llm: LLMBackend
    judge: LLMBackend
    nli: NLIBackend
    sae: SAEBackend


def build_context(cfg: ExperimentConfig, *, run_id: str | None = None) -> tuple[str, PipelineContext]:
    """Resolve run_id, prepare the run directory, and instantiate backends.

    Backends are constructed eagerly. For `fake` and `openrouter` this is
    cheap; `hf_local` LLM and `sae_lens` SAE backends are lazy — the heavy
    model download happens on first use.
    """
    rid, store = make_run(cfg, run_id=run_id)
    llm = build_llm_backend(cfg.model.provider, cfg.model.name, dtype=cfg.model.dtype)
    judge = build_llm_backend(
        cfg.judge.provider, cfg.judge.model, max_retries=cfg.judge.max_retries
    )
    nli = build_nli_backend(cfg.nli.provider, cfg.nli.model)
    sae = build_sae_backend(cfg.sae)
    return rid, PipelineContext(cfg=cfg, store=store, llm=llm, judge=judge, nli=nli, sae=sae)
