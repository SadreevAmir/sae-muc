# Copyright (c) Meta Platforms, Inc. and affiliates. (copied from calibration/causal.py)

import jsonlines
import torch
from tqdm.auto import tqdm

_NUM_SAMPLE_RESPONSES = 10


def generate_lines_for_batch(
    model,
    tokenizer,
    batch_questions: list,
    batch_messages: list,
    alpha: float,
) -> list[dict]:
    """Один forward-батч (одинаковый alpha для хуков). Возвращает строки jsonl без записи на диск."""
    inputs = tokenizer.apply_chat_template(
        batch_messages,
        tokenize=True,
        add_generation_prompt=True,
        truncation=True,
        padding=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    n = len(batch_questions)
    n_samples = _NUM_SAMPLE_RESPONSES
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            temperature=0.1,
        )
        outputs_responses = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=1,
            num_return_sequences=n_samples,
        )
    decoded_answers = tokenizer.batch_decode(
        outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    decoded_responses = tokenizer.batch_decode(
        outputs_responses[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    assert len(decoded_answers) == n == len(decoded_responses) // n_samples
    lines = []
    for j, question in enumerate(batch_questions):
        lines.append(
            {
                "alpha": alpha,
                "question": question,
                "most_likely_answer": decoded_answers[j],
                "responses": decoded_responses[j * n_samples : (j + 1) * n_samples],
            }
        )
    return lines


def generate_all_responses(model, tokenizer, all_questions, all_message, alpha, out_file, batch_size):
    for i in tqdm(
        range(0, len(all_message), batch_size),
        total=max(1, (len(all_message) + batch_size - 1) // batch_size),
    ):
        batch_message = all_message[i : i + batch_size]
        batch_question = all_questions[i : i + batch_size]
        for line in generate_lines_for_batch(
            model, tokenizer, batch_question, batch_message, alpha
        ):
            with jsonlines.open(out_file, "a") as writer:
                writer.write(line)
