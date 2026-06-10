"""hidden_states: save residual-stream activations for every token.

For each sample we forward a text once and snapshot the hidden state at
every kept token of every layer. Downstream stages (vuf, detect) pool
the sequence — last-token-of-question, last-token-of-answer, mean, etc.
— without re-running the forward pass.

Storage modes (`cfg.stages.hidden_states.storage`):
  - `full` (default): forward `question + greedy_answer`, keep all tokens
    (paper-faithful + supports last_token_a / mean_a pooling).
  - `question_only`: forward just the question. Cheaper; sufficient for
    pooling=last_token_q / mean_q.
  - `last_k_tokens`: forward full text, keep only the last `last_k` tokens.

Artefact layout (relative to the run directory):
  - hidden_states/embedding.safetensors   # token-embedding layer
  - hidden_states/layer_0.safetensors      # 1st transformer block output
  - hidden_states/layer_{n_layers-1}.safetensors
  - hidden_states/meta.parquet             # sample_id, seq_len, question_len, answer_len, storage

Each safetensors file is a map `sample_id -> tensor[seq_len, d_model]` with
ragged sequence lengths (no padding). Sequence lengths per sample are
recorded in `meta.parquet`.
"""

from __future__ import annotations

import logging

import pandas as pd

from sae_muc.data.prompts import format_answer_prompt
from sae_muc.pipeline._utils import PROMPT_ELICITING, select_prompt_kind
from sae_muc.pipeline.context import PipelineContext

log = logging.getLogger(__name__)

OUTPUT_META = "hidden_states/meta.parquet"
OUTPUT_EMBED = "hidden_states/embedding.safetensors"


def _layer_file(layer: int) -> str:
    return f"hidden_states/layer_{layer}.safetensors"


def run(ctx: PipelineContext) -> list[str]:
    stage_cfg = ctx.cfg.stages.hidden_states
    storage = stage_cfg.storage
    last_k = int(stage_cfg.last_k)

    samples = ctx.store.load_parquet("samples.parquet")
    gens = ctx.store.load_parquet("generations.parquet")
    # VUF extraction processes the question under the eliciting prompt (§3.1),
    # so pair the residual extraction with the eliciting most-likely answer
    # (only matters for last_token_a / mean_a pooling + full-text storage).
    greedy = select_prompt_kind(
        gens[gens["kind"] == "greedy"], PROMPT_ELICITING
    ).set_index("sample_id")

    texts: list[str] = []
    sample_ids: list[str] = []
    question_lens: list[int] = []

    for _, row in samples.iterrows():
        question_text = format_answer_prompt(row["question"], eliciting=True)
        if storage == "question_only":
            full_text = question_text
        else:
            greedy_answer = greedy.loc[row["sample_id"], "text"]
            full_text = f"{question_text} {greedy_answer}"

        sample_ids.append(row["sample_id"])
        texts.append(full_text)
        question_lens.append(ctx.llm.tokenize_length(question_text))

    log.info(
        "extracting residual-stream activations for %d samples (storage=%s%s)",
        len(texts), storage,
        f", last_k={last_k}" if storage == "last_k_tokens" else "",
    )
    hidden_list = ctx.llm.hidden_states(texts)
    # Each element shape: [n_layers+1, seq_len_i, d_model]. Layer 0 is
    # the token-embedding output; layers 1..n_layers are transformer blocks.

    if storage == "last_k_tokens":
        sliced: list = []
        new_question_lens: list[int] = []
        for hs, q_len in zip(hidden_list, question_lens, strict=True):
            seq_full = hs.shape[1]
            offset = max(0, seq_full - last_k)
            sliced.append(hs[:, offset:, :])
            # Translate question_len into the kept window: if the question
            # tail is in the window, q_len_stored ≤ last_k. If the question
            # ended before the window, q_len_stored = 0 and last_token_q
            # will degenerate to position 0 (the last token before the kept
            # window's start). Document but don't crash.
            new_question_lens.append(max(0, q_len - offset))
        hidden_list = sliced
        question_lens = new_question_lens

    n_hidden = hidden_list[0].shape[0]
    n_layers = n_hidden - 1
    log.info(
        "saving %d transformer layers (plus embedding) × %d samples",
        n_layers, len(texts),
    )

    outputs: list[str] = []

    embedding = {sid: hs[0] for sid, hs in zip(sample_ids, hidden_list, strict=True)}
    ctx.store.save_safetensors(OUTPUT_EMBED, embedding)
    outputs.append(OUTPUT_EMBED)

    for layer in range(n_layers):
        per_sample = {
            sid: hs[layer + 1]
            for sid, hs in zip(sample_ids, hidden_list, strict=True)
        }
        path = _layer_file(layer)
        ctx.store.save_safetensors(path, per_sample)
        outputs.append(path)

    meta = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "seq_len": [hs.shape[1] for hs in hidden_list],
            "question_len": question_lens,
            "n_layers": n_layers,
            "storage": storage,
        }
    )
    meta["answer_len"] = meta["seq_len"] - meta["question_len"]
    ctx.store.save_parquet(OUTPUT_META, meta)
    outputs.append(OUTPUT_META)

    return outputs
