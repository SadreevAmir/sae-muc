# Copyright (c) Meta Platforms, Inc. and affiliates. (copied from calibration/causal.py)

import jsonlines
import torch
from tqdm.auto import tqdm


def generate_all_responses(model, tokenizer, all_questions, all_message, alpha, out_file, batch_size):
    for i in tqdm(
        range(0, len(all_message), batch_size),
        total=max(1, len(all_message) // batch_size),
    ):
        batch_message = all_message[i : i + batch_size]
        batch_question = all_questions[i : i + batch_size]

        inputs = tokenizer.apply_chat_template(
            batch_message,
            tokenize=True,
            add_generation_prompt=True,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=0.1,
            )
            N = 10
            outputs_responses = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=1,
                num_return_sequences=N,
            )
        decoded_answers = tokenizer.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        decoded_responses = tokenizer.batch_decode(
            outputs_responses[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        assert len(decoded_answers) == len(batch_question) == len(decoded_responses) // N
        for j, (question, answer) in enumerate(zip(batch_question, decoded_answers)):
            line = {
                "alpha": alpha,
                "question": question,
                "most_likely_answer": answer,
                "responses": decoded_responses[j * N : (j + 1) * N],
            }
            with jsonlines.open(out_file, "a") as writer:
                writer.write(line)
