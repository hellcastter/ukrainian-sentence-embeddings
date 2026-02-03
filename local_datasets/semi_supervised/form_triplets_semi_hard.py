#!/usr/bin/env python3
import csv
import json
import random
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer, models

INPUT_FILE = "local_datasets/semi_supervised/merged_collected_and_generated_1.json"
OUTPUT_FILE = "local_datasets/semi_supervised/semi_supervised_triplets_hard_mined_1.csv"
MAX_SENTENCES_PER_MEANING = 50

USE_BACK_TRANSLATED = True
BACK_TRANSLATION_PATH = (
    "local_datasets/translation/augmented_sentences_translated_v3.jsonl"
)

TOKENIZER_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
MODEL_NAME = "models/fine-tuned-models/model_-134_0"
BATCH_SIZE = 64
DEVICE = "cuda"

# ===== NEW SAFER MINING PARAMETERS =====
NUM_SEMI_HARD_NEGATIVES = 30
NUM_RANDOM_NEGATIVES = 10
NUM_POSITIVES_TO_SAMPLE = 5
MARGIN = 0.2
MAX_ATTEMPTS_PER_ANCHOR = 200
# ======================================


def get_recommended_number_of_sentences(
    n_sentences_per_meaning: int, index: int
) -> int:
    if n_sentences_per_meaning >= MAX_SENTENCES_PER_MEANING:
        return 1

    base = MAX_SENTENCES_PER_MEANING // n_sentences_per_meaning
    remainder = MAX_SENTENCES_PER_MEANING % n_sentences_per_meaning

    return base + 1 if index < remainder else base


# Load back translations
back_translated_sentences = {}
if USE_BACK_TRANSLATED:
    print("Loading back-translated sentences...")
    with open(BACK_TRANSLATION_PATH, "r", encoding="utf-8") as bt_file:
        for line in bt_file:
            item = json.loads(line)
            back_translated_sentences[item["sentence"]] = item.get("augmented", [])

print("Loading input data...")
with open(INPUT_FILE, "r", encoding="utf-8") as infile:
    data = json.load(infile)

transformer = models.Transformer(MODEL_NAME, tokenizer_name_or_path=TOKENIZER_NAME)
pooling = models.Pooling(
    transformer.get_word_embedding_dimension(), pooling_mode_mean_tokens=True
)

model = SentenceTransformer(modules=[transformer, pooling], device=DEVICE)
print(f"Using model {MODEL_NAME} on device {DEVICE}")


with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["lemma", "anchor", "positive", "negative"])

    for lemma, meanings in tqdm(data.items(), desc="Lemmas", total=len(data)):
        lemma_meanings = [
            m for m in meanings.values() if len(m.get("sentences", [])) >= 1
        ]
        if len(lemma_meanings) < 2:
            continue

        items = []
        orig_anchor_to_item_indices = {}

        for meaning_idx, meaning in enumerate(lemma_meanings):
            base_sentences = meaning["sentences"][:MAX_SENTENCES_PER_MEANING]
            n_sentences = len(base_sentences)

            for orig_index, s in enumerate(base_sentences):
                orig_text = s["sentence"]
                orig_anchor_id = (meaning_idx, orig_index, n_sentences)

                idx = len(items)
                items.append(
                    {
                        "text": orig_text,
                        "meaning_idx": meaning_idx,
                        "orig_anchor_id": orig_anchor_id,
                    }
                )
                orig_anchor_to_item_indices.setdefault(orig_anchor_id, []).append(idx)

                if USE_BACK_TRANSLATED:
                    for aug in back_translated_sentences.get(orig_text, []):
                        j = len(items)
                        items.append(
                            {
                                "text": aug,
                                "meaning_idx": meaning_idx,
                                "orig_anchor_id": orig_anchor_id,
                            }
                        )
                        orig_anchor_to_item_indices.setdefault(
                            orig_anchor_id, []
                        ).append(j)

        if len(items) < 3:
            continue

        texts = [it["text"] for it in items]
        embeddings = model.encode(
            texts, batch_size=BATCH_SIZE, convert_to_numpy=True, show_progress_bar=False
        )

        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        sim = embeddings @ embeddings.T

        meaning_to_indices = {}
        for idx, it in enumerate(items):
            meaning_to_indices.setdefault(it["meaning_idx"], []).append(idx)

        for orig_anchor_id, anchor_indices in orig_anchor_to_item_indices.items():
            meaning_idx, orig_pos, n_sentences = orig_anchor_id
            recommended = get_recommended_number_of_sentences(n_sentences, orig_pos)

            same_meaning = meaning_to_indices[meaning_idx]
            other_meaning = [
                i
                for m, idxs in meaning_to_indices.items()
                if m != meaning_idx
                for i in idxs
            ]

            if len(same_meaning) < 2 or not other_meaning:
                continue

            produced = 0
            used = set()

            for a_idx in anchor_indices:
                attempts = 0

                pos_candidates = [i for i in same_meaning if i != a_idx]

                if not pos_candidates:
                    continue

                pos_sims = sim[a_idx, pos_candidates]
                sorted_pos = np.argsort(pos_sims)

                # avoid only the absolute hardest positives
                valid_pos = [
                    pos_candidates[i] for i in sorted_pos[:NUM_POSITIVES_TO_SAMPLE]
                ]

                neg_sims = sim[a_idx, other_meaning]
                sorted_neg = np.argsort(-neg_sims)

                # Semi-hard: take from a band instead of absolute top
                semi_hard = [
                    other_meaning[i]
                    for i in sorted_neg[5 : 5 + NUM_SEMI_HARD_NEGATIVES]
                ]

                random_negs = random.sample(
                    other_meaning, min(NUM_RANDOM_NEGATIVES, len(other_meaning))
                )

                candidate_negs = list(set(semi_hard + random_negs))

                while produced < recommended and attempts < MAX_ATTEMPTS_PER_ANCHOR:
                    attempts += 1

                    p = random.choice(valid_pos)
                    n = random.choice(candidate_negs)

                    # margin-based sanity check
                    if sim[a_idx, p] + MARGIN < sim[a_idx, n]:
                        continue

                    key = (a_idx, p, n)
                    if key in used:
                        continue

                    used.add(key)
                    writer.writerow(
                        [
                            lemma,
                            items[a_idx]["text"],
                            items[p]["text"],
                            items[n]["text"],
                        ]
                    )
                    produced += 1

                if produced >= recommended:
                    break

print(f"Triplet mining finished — output written to {OUTPUT_FILE}")
