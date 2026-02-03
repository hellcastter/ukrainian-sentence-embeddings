import csv
import json
import random

from tqdm import tqdm

INPUT_FILE = "local_datasets/semi_supervised/merged_collected_and_generated.json"
OUTPUT_FILE = "local_datasets/semi_supervised/semi_supervised_triplets_with_2.csv"
MAX_SENTENCES_PER_MEANING = 50

USE_BACK_TRANSLATED = True
BACK_TRANSLATION_PATH = (
    "local_datasets/translation/augmented_sentences_translated_v3.jsonl"
)


def get_recommended_number_of_sentences(
    n_sentences_per_meaning: int, index: int
) -> int:
    if n_sentences_per_meaning >= MAX_SENTENCES_PER_MEANING:
        return 1

    base = MAX_SENTENCES_PER_MEANING // n_sentences_per_meaning
    remainder = MAX_SENTENCES_PER_MEANING % n_sentences_per_meaning

    return base + 1 if index < remainder else base


if USE_BACK_TRANSLATED:
    back_translated_sentences = {}
    print("Loading back-translated sentences...")
    with open(BACK_TRANSLATION_PATH, "r", encoding="utf-8") as bt_file:
        for line in bt_file:
            item = json.loads(line)
            back_translated_sentences[item["sentence"]] = item["augmented"]

print("Loading input data...")
with open(INPUT_FILE, "r", encoding="utf-8") as infile:
    data = json.load(infile)

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["lemma", "anchor", "positive", "negative"])

    for lemma, meanings in tqdm(data.items(), desc="Forming triplets", total=len(data)):
        lemma_meanings = list(meanings.values())
        lemma_meanings = [
            m for m in lemma_meanings if len(m["sentences"]) >= 1
        ]  # at least 1 sentence per meaning

        # there will be no negative samples
        if len(lemma_meanings) < 2:
            continue

        for i, meaning in enumerate(lemma_meanings):
            # there will be no positive samples
            if len(meaning["sentences"]) <= 1:
                continue

            sentences = meaning["sentences"][:]
            random.shuffle(sentences)
            sentences = sentences[:MAX_SENTENCES_PER_MEANING]
            n_sentences = len(sentences)
            for i, sentence in enumerate(sentences):
                # find all anchor variations
                anchor = sentence["sentence"]
                anchors = [anchor]

                recommended_sentences_with_anchor = get_recommended_number_of_sentences(
                    n_sentences, i
                )

                if USE_BACK_TRANSLATED:
                    anchors_augmented = back_translated_sentences.get(anchor)
                    if anchors_augmented is not None:
                        anchors.extend(anchors_augmented)

                # find all positives
                positives = set(
                    [
                        s["sentence"]
                        for s in meaning["sentences"]
                        if s["sentence"] != anchor
                    ]
                )
                if USE_BACK_TRANSLATED:
                    positives_augmented = [
                        back_translated_sentences.get(s) for s in positives
                    ]
                    positives_augmented = [
                        i for aug in positives_augmented if aug is not None for i in aug
                    ]
                    positives.update(positives_augmented)

                # find all negatives
                negative_meanings = [m for j, m in enumerate(lemma_meanings) if j != i]
                negatives = set()
                for negative_meaning in negative_meanings:
                    negatives.update(
                        [s["sentence"] for s in negative_meaning["sentences"]]
                    )

                if USE_BACK_TRANSLATED:
                    negatives_augmented = [
                        back_translated_sentences.get(s) for s in negatives
                    ]
                    negatives_augmented = [
                        i for aug in negatives_augmented if aug is not None for i in aug
                    ]
                    negatives.update(negatives_augmented)

                # form triplets
                used_pairs = set()

                number_of_possible_triplets = (
                    len(anchors) * len(positives) * len(negatives)
                )
                if recommended_sentences_with_anchor > number_of_possible_triplets:
                    print(
                        f"Warning: Not enough unique triplets can be formed for lemma '{lemma}', meaning index {i}."
                        f"Requested {recommended_sentences_with_anchor}, can form only {number_of_possible_triplets}."
                    )

                # print(
                #     f"Lemma '{lemma}', meaning index {i}: "
                #     f"Using {len(anchors)} anchors, {len(positives)} positives, {len(negatives)} negatives. "
                #     f"Forming up to {min(recommended_sentences_with_anchor, number_of_possible_triplets)} triplets."
                # )
                for _ in range(
                    min(recommended_sentences_with_anchor, number_of_possible_triplets)
                ):
                    while True:
                        anchor = random.choice(anchors)
                        positive = random.choice(list(positives))
                        negative = random.choice(list(negatives))
                        key = (anchor, positive, negative)

                        if key not in used_pairs:
                            used_pairs.add(key)
                            break

                    writer.writerow([lemma, anchor, positive, negative])
