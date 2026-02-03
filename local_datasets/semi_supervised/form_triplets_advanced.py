#!/usr/bin/env python3
# form_hard_triplets.py
import csv
import json
import random
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer, models
import math
import os

INPUT_FILE = "local_datasets/semi_supervised/merged_collected_and_generated_2.json"
OUTPUT_FILE = "local_datasets/semi_supervised/semi_supervised_triplets_hard_mined_2.csv"
MAX_SENTENCES_PER_MEANING = 100

USE_BACK_TRANSLATED = True
BACK_TRANSLATION_PATH = (
    "local_datasets/translation/augmented_sentences_translated_v3.jsonl"
)

TOKENIZER_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
MODEL_NAME = "models/fine-tuned-models/model_-134_0"
BATCH_SIZE = 64
NUM_HARD_NEGATIVES = 20  # how many hardest negatives to consider per anchor
NUM_HARD_POSITIVES = (
    10  # how many hardest positives to consider per anchor (choose lowest-similarities)
)
DEVICE = "cuda"


def get_recommended_number_of_sentences(
    n_sentences_per_meaning: int, index: int
) -> int:
    if n_sentences_per_meaning >= MAX_SENTENCES_PER_MEANING:
        return 1

    base = MAX_SENTENCES_PER_MEANING // n_sentences_per_meaning
    remainder = MAX_SENTENCES_PER_MEANING % n_sentences_per_meaning

    return base + 1 if index < remainder else base


# load back-translations mapping: original_sentence -> list_of_augments
back_translated_sentences = {}
if USE_BACK_TRANSLATED:
    print("Loading back-translated sentences...")
    with open(BACK_TRANSLATION_PATH, "r", encoding="utf-8") as bt_file:
        for line in bt_file:
            item = json.loads(line)
            # expect item['sentence'] -> item['augmented'] (list)
            back_translated_sentences[item["sentence"]] = item.get("augmented", [])

print("Loading input data...")
with open(INPUT_FILE, "r", encoding="utf-8") as infile:
    data = json.load(infile)


transformer = models.Transformer(MODEL_NAME, tokenizer_name_or_path=TOKENIZER_NAME)
pooling = models.Pooling(
    transformer.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False,
)

model = SentenceTransformer(modules=[transformer, pooling], device=DEVICE)
print(f"Using model {MODEL_NAME} on device {DEVICE}")

# We'll write while processing lemma-by-lemma to avoid storing all triplets in memory
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["lemma", "anchor", "positive", "negative"])

    # iterate lemmas
    for lemma, meanings in tqdm(data.items(), desc="Lemmas", total=len(data)):
        # collect per-meaning sentence lists (filter at least 1)
        lemma_meanings = [
            m for m in meanings.values() if len(m.get("sentences", [])) >= 1
        ]
        if len(lemma_meanings) < 2:
            # need at least two meanings to form negatives
            continue

        # Build items: list of dicts {text, meaning_idx, orig_anchor (meaning_idx, idx_in_base_list)}
        items = []
        # We'll also keep mapping from orig_anchor_id -> list of item indices (to treat augmented versions as anchors for same orig)
        orig_anchor_to_item_indices = {}

        for meaning_idx, meaning in enumerate(lemma_meanings):
            base_sentences = meaning["sentences"][:]
            random.shuffle(base_sentences)
            base_sentences = base_sentences[:MAX_SENTENCES_PER_MEANING]
            n_sentences = len(base_sentences)

            for orig_index, s in enumerate(base_sentences):
                orig_text = s["sentence"]
                # Each original sentence gets an orig_anchor_id used for recommended count
                orig_anchor_id = (
                    meaning_idx,
                    orig_index,
                    n_sentences,
                )  # include n_sentences for easier retrieval later

                # add original text as an item
                idx = len(items)
                items.append(
                    {
                        "text": orig_text,
                        "meaning_idx": meaning_idx,
                        "orig_anchor_id": orig_anchor_id,
                    }
                )
                orig_anchor_to_item_indices.setdefault(orig_anchor_id, []).append(idx)

                # add back-translations as separate items (if present)
                if USE_BACK_TRANSLATED:
                    aug_list = back_translated_sentences.get(orig_text)
                    if aug_list:
                        for aug in aug_list:
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
            continue  # nothing to do for this lemma

        # encode all items' texts in batches
        texts = [it["text"] for it in items]
        embeddings = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i : i + BATCH_SIZE]
            batch_emb = model.encode(
                batch_texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                device=DEVICE,
            )
            embeddings.append(batch_emb)
        embeddings = np.vstack(embeddings)

        # normalize embeddings for cosine similarity with dot product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        embeddings = embeddings / norms

        # compute similarity matrix (cosine) — small lemmas should fit in memory
        sim = embeddings @ embeddings.T  # shape (N, N)

        N = len(items)

        # precompute indices by meaning for quick lookup
        meaning_to_indices = {}
        for idx, it in enumerate(items):
            meaning_to_indices.setdefault(it["meaning_idx"], []).append(idx)

        # Now iterate over each original anchor group and produce triplets
        for orig_anchor_id, anchor_item_indices in orig_anchor_to_item_indices.items():
            meaning_idx, orig_pos, n_sentences = orig_anchor_id
            recommended = get_recommended_number_of_sentences(n_sentences, orig_pos)

            # gather all candidate indices for positives and negatives
            # positives: indices sharing same meaning (exclude anchor index)
            same_meaning_indices = meaning_to_indices.get(meaning_idx, [])
            # if less than 2 items in meaning (after augmentation) we can't form positive; skip
            if len(same_meaning_indices) <= 1:
                continue

            other_meaning_indices = [
                ii
                for mi, idxs in meaning_to_indices.items()
                for ii in idxs
                if mi != meaning_idx
            ]
            if len(other_meaning_indices) == 0:
                continue

            # For counting triplets produced for this original anchor
            produced = 0
            used_triplet_keys = set()

            # Precompute for each anchor candidate its sorted lists of hard positives and hard negatives
            # To avoid recomputing repeatedly, compute for each anchor index in anchor_item_indices:
            precomputed = {}
            for a_idx in anchor_item_indices:
                # positives: same meaning excluding self
                pos_idxs = [i for i in same_meaning_indices if i != a_idx]
                # compute similarities and sort ascending -> hardest positives are smallest similarity
                pos_sims = sim[a_idx, pos_idxs]
                # sort pos_idxs by sim ascending
                pos_order = np.argsort(pos_sims)  # smallest first -> hardest positive
                hard_pos_idxs = (
                    [pos_idxs[k] for k in pos_order[:NUM_HARD_POSITIVES]]
                    if len(pos_order) > 0
                    else []
                )

                # negatives: other meanings; sort descending -> hardest negatives (highest sim)
                neg_idxs = other_meaning_indices
                neg_sims = sim[a_idx, neg_idxs]
                neg_order = np.argsort(-neg_sims)  # largest first
                hard_neg_idxs = (
                    [neg_idxs[k] for k in neg_order[:NUM_HARD_NEGATIVES]]
                    if len(neg_order) > 0
                    else []
                )

                precomputed[a_idx] = {
                    "hard_pos": hard_pos_idxs,
                    "hard_neg": hard_neg_idxs,
                }

            # If no anchors have hard negatives/positives, skip
            any_valid = any(
                len(v["hard_pos"]) > 0 and len(v["hard_neg"]) > 0
                for v in precomputed.values()
            )
            if not any_valid:
                continue

            # Form triplets by cycling through anchors, hard positives, hard negatives
            # We'll attempt to create up to `recommended` unique triplets for this original anchor
            anchor_cycle = list(anchor_item_indices)
            pos_cycle_map = {a: precomputed[a]["hard_pos"] for a in anchor_cycle}
            neg_cycle_map = {a: precomputed[a]["hard_neg"] for a in anchor_cycle}

            # indices to iterate in round-robin fashion
            a_ptr = 0
            pos_ptrs = {a: 0 for a in anchor_cycle}
            neg_ptrs = {a: 0 for a in anchor_cycle}
            attempts = 0
            max_attempts = max(1000, recommended * 50)  # safety stop

            while produced < recommended and attempts < max_attempts:
                attempts += 1
                a = anchor_cycle[a_ptr % len(anchor_cycle)]
                a_ptr += 1

                pos_list = pos_cycle_map.get(a, [])
                neg_list = neg_cycle_map.get(a, [])
                if not pos_list or not neg_list:
                    continue

                # select the next positive (hardest first, then cycle)
                p_idx = pos_list[pos_ptrs[a] % len(pos_list)]
                pos_ptrs[a] += 1
                # select the next negative (hardest first, then cycle)
                n_idx = neg_list[neg_ptrs[a] % len(neg_list)]
                neg_ptrs[a] += 1

                anchor_text = items[a]["text"]
                positive_text = items[p_idx]["text"]
                negative_text = items[n_idx]["text"]

                key = (anchor_text, positive_text, negative_text)
                if key in used_triplet_keys:
                    continue

                used_triplet_keys.add(key)
                writer.writerow([lemma, anchor_text, positive_text, negative_text])
                produced += 1

            if attempts >= max_attempts and produced < recommended:
                # best-effort: you can log this if you want
                pass

# done
print(f"Triplet mining finished — output written to {OUTPUT_FILE}")
