import csv
from local_datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
import random
import json

random.seed(42)

load_dotenv()

SRC_PATH = "local_datasets/ubertext_triplets_200K.csv"
DST_PATH = (
    "local_datasets/ubertext_triplets_200K_back_translation_changed_triples_3.csv"
)

LEMMA_IDX = None
ANCHOR_IDX = None
POSITIVE_IDX = None
NEGATIVE_IDX = None
N_AUGMENTED = 2

# read back-translation dataset
back_translation_dataset = load_dataset(
    "hellcaster/wsd-sentences", split="back_translation"
)
back_translation_dict = {
    item["sentence"]: item["augmented"]
    for item in tqdm(
        back_translation_dataset, desc="Building back-translation dictionary"
    )
}

# read lemma sentences dataset
lemma_sentences = {}
with open("local_datasets/lemma_sentences.jsonl", "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Reading lemma sentences"):
        entry = json.loads(line)
        lemma_sentences[entry["lemma"]] = entry["sentences"]

# read original dataset
original_dataset = []
with open(SRC_PATH, "r") as f:
    reader = csv.reader(f)
    header = next(reader)  # Skip header

    LEMMA_IDX = header.index("lemma")
    POSITIVE_IDX = header.index("positive")
    NEGATIVE_IDX = header.index("negative")
    ANCHOR_IDX = header.index("anchor")

    for row in reader:
        original_dataset.append(row)

# augment and write new dataset
n_rows = 0
with open(DST_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)  # Write header

    for row in tqdm(original_dataset, desc="Augmenting dataset with back-translation"):
        lemma = row[LEMMA_IDX]
        anchor = row[ANCHOR_IDX]

        back_translated_anchors = back_translation_dict.get(anchor, [])
        if not back_translated_anchors:
            continue

        for _ in range(N_AUGMENTED):
            negative = None
            while not negative or negative == anchor:
                negative = random.choice(lemma_sentences[lemma])

            back_translation_negatives = back_translation_dict.get(negative, [])
            back_translation_negatives.append(negative)

            row[ANCHOR_IDX] = anchor
            row[POSITIVE_IDX] = random.choice(back_translated_anchors)
            row[NEGATIVE_IDX] = random.choice(back_translation_negatives)

            writer.writerow(row)
            n_rows += 1

print(f"Total rows written: {n_rows}")
