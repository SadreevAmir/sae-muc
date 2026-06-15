"""Shared runtime context passed into every pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass

from sae_muc.artifacts import ArtifactStore, make_run
from sae_muc.config import ExperimentConfig, SAEConfig
from sae_muc.models import LLMBackend, NLIBackend, build_llm_backend, build_nli_backend
from sae_muc.models.sae import SAEBackend, build_sae_registry


@dataclass
class PipelineContext:
    cfg: ExperimentConfig
    store: ArtifactStore
    llm: LLMBackend
    judge: LLMBackend
    nli: NLIBackend
    saes: dict[int, SAEBackend]


# Conservative cap covering every model we currently target (Llama-3.1-70B = 80,
# Qwen2.5-72B = 80). Wrappers are cheap; sae-lens weights load lazily.
_MAX_CANDIDATE_LAYERS = 96


def _candidate_sae_layers(cfg: SAEConfig) -> list[int]:
    """Layers for which we pre-build SAE wrappers in `build_context`.

    `target_layers` is only known after the vuf stage (resolved against
    `vuf/meta.parquet`), so we over-cover here. For sae-lens this just means
    extra wrapper objects — nothing is downloaded until a stage calls encode().
    """
    layers: set[int] = set(cfg.sae_id_overrides.keys())
    if cfg.sae_id_template or cfg.sae_id is not None or cfg.provider == "fake":
        layers |= set(range(_MAX_CANDIDATE_LAYERS))
    return sorted(layers)


def build_context(cfg: ExperimentConfig, *, run_id: str | None = None) -> tuple[str, PipelineContext]:
    """Resolve run_id, prepare the run directory, and instantiate backends.

    Backends are constructed eagerly. For `fake` and `openrouter` this is
    cheap; `hf_local` LLM and `sae_lens` SAE backends are lazy — the heavy
    model download happens on first use.
    """
    rid, store = make_run(cfg, run_id=run_id)
    llm = build_llm_backend(cfg.model.provider, cfg.model.name, dtype=cfg.model.dtype)
    judge = build_llm_backend(
        cfg.judge.provider,
        cfg.judge.model,
        max_retries=cfg.judge.max_retries,
        provider_only=cfg.judge.provider_only,
    )
    nli = build_nli_backend(cfg.nli.provider, cfg.nli.model)
    saes = build_sae_registry(cfg.sae, _candidate_sae_layers(cfg.sae))
    return rid, PipelineContext(cfg=cfg, store=store, llm=llm, judge=judge, nli=nli, saes=saes)
