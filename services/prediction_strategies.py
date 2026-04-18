import numpy as np

import torch
import torch.nn.functional as F

from tqdm import tqdm

# from services.utils_embedding_calculation import (
#     get_target_word_embedding,
#     get_context_embedding,
# )

from services.utils_embedding_calculation_v2 import (
    get_target_word_embedding,
    get_context_embedding,
)


class PredictionStrategy:
    @staticmethod
    def all_examples_to_one_embedding(
        lemma,
        examples,
        contexts,
        model,
        tokenizer,
        udpipe_model,
        pooling_strategy,
        device,
    ):
        # TODO: check whether it's more efficient to create numpy here
        max_sim = -1
        correct_context = None

        combined_embedding = [
            get_target_word_embedding(
                model, tokenizer, udpipe_model, pooling_strategy, lemma, example, device
            )
            for example in examples
        ]
        combined_embedding = [emb for emb in combined_embedding if emb is not None]

        if len(combined_embedding) == 0:
            return None

        combined_embedding = torch.tensor(np.array(combined_embedding))
        combined_embedding = torch.mean(combined_embedding, dim=0)

        for context in contexts:
            max_sub_sim = -1

            for sub_context in context:
                sub_context_embedding = get_context_embedding(
                    model, tokenizer, pooling_strategy, sub_context, device
                )
                sub_similarity = F.cosine_similarity(
                    combined_embedding, torch.tensor(sub_context_embedding), dim=0
                ).item()

                if sub_similarity > max_sub_sim:
                    max_sub_sim = sub_similarity

            if max_sub_sim > max_sim:
                max_sim = max_sub_sim
                correct_context = context

        return correct_context

    @staticmethod
    def max_sim_across_all_examples(
        lemma,
        examples,
        contexts,
        model,
        tokenizer,
        udpipe_model,
        pooling_strategy,
        device,
    ):
        max_sim = -1
        correct_context = None

        target_word_embeddings = [
            get_target_word_embedding(
                model, tokenizer, udpipe_model, pooling_strategy, lemma, example, device
            )
            for example in examples
        ]

        target_word_embeddings = [
            emb for emb in target_word_embeddings if emb is not None
        ]

        target_word_embeddings = [
            torch.tensor(np.array(emb)) for emb in target_word_embeddings
        ]

        for context in contexts:
            max_sub_sim = -1

            for sub_context in context:
                sub_context_embedding = get_context_embedding(
                    model, tokenizer, pooling_strategy, sub_context, device
                )
                sub_context_embedding = torch.tensor(sub_context_embedding)

                for embedding in target_word_embeddings:
                    sub_similarity = F.cosine_similarity(
                        embedding, sub_context_embedding, dim=0
                    ).item()

                    max_sub_sim = max(max_sub_sim, sub_similarity)

            if max_sub_sim > max_sim:
                max_sim = max_sub_sim
                correct_context = context

        return correct_context
