"""hidden_states: save residual-stream activations for every token.

For each sample we forward the full text (prompt + greedy answer) once
and snapshot the hidden state at every token of every layer. Downstream
stages (vuf, detect) pool the sequence — last-token-of-question,
last-token-of-answer, mean, etc. — without re-running the forward pass.

Artefact layout (relative to the run directory):
  - hidden_states/embedding.safetensors   # token-embedding layer
  - hidden_states/layer_0.safetensors      # 1st transformer block output
  - hidden_states/layer_{n_layers-1}.safetensors
  - hidden_states/meta.parquet             # sample_id, seq_len, question_len, answer_len

Each safetensors file is a map `sample_id -> tensor[seq_len, d_model]` with
ragged sequence lengths (no padding). Sequence lengths per sample are
recorded in `meta.parquet`.
"""

from __future__ import annotations

import pandas as pd

from sae_muc.data.prompts import format_answer_prompt
from sae_muc.pipeline.context import PipelineContext

OUTPUT_META = "hidden_states/meta.parquet"
OUTPUT_EMBED = "hidden_states/embedding.safetensors"


def _layer_file(layer: int) -> str:
    return f"hidden_states/layer_{layer}.safetensors"


def run(ctx: PipelineContext) -> list[str]:
    samples = ctx.store.load_parquet("samples.parquet")
    gens = ctx.store.load_parquet("generations.parquet")
    greedy = gens[gens["kind"] == "greedy"].set_index("sample_id")

    texts: list[str] = []
    sample_ids: list[str] = []
    question_lens: list[int] = []

    for _, row in samples.iterrows():
        question_text = format_answer_prompt(row["question"], eliciting=True)
        greedy_answer = greedy.loc[row["sample_id"], "text"]
        full_text = f"{question_text} {greedy_answer}"

        sample_ids.append(row["sample_id"])
        texts.append(full_text)
        question_lens.append(ctx.llm.tokenize_length(question_text))

    hidden_list = ctx.llm.hidden_states(texts)
    # Each element shape: [n_layers+1, seq_len_i, d_model]. Layer 0 is
    # the token-embedding output; layers 1..n_layers are transformer blocks.
    n_hidden = hidden_list[0].shape[0]
    n_layers = n_hidden - 1

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
        }
    )
    meta["answer_len"] = meta["seq_len"] - meta["question_len"]
    ctx.store.save_parquet(OUTPUT_META, meta)
    outputs.append(OUTPUT_META)

    return outputs
