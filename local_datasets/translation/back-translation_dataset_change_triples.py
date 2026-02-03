import csv
from local_datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
import random

random.seed(42)

load_dotenv()

SRC_PATH = "local_datasets/ubertext_triplets_200K.csv"
DST_PATH = (
    "local_datasets/ubertext_triplets_200K_back_translation_changed_triples_2.csv"
)

ANCHOR_IDX = None
POSITIVE_IDX = None
NEGATIVE_IDX = None
N_AUGMENTED = 2

back_translation_dataset = load_dataset(
    "hellcaster/wsd-sentences", split="back_translation"
)
back_translation_dict = {
    item["sentence"]: item["augmented"]
    for item in tqdm(
        back_translation_dataset, desc="Building back-translation dictionary"
    )
}

original_dataset = []
with open(SRC_PATH, "r") as f:
    reader = csv.reader(f)
    header = next(reader)  # Skip header

    POSITIVE_IDX = header.index("positive")
    NEGATIVE_IDX = header.index("negative")
    ANCHOR_IDX = header.index("anchor")

    for row in reader:
        original_dataset.append(row)

n_rows = 0
with open(DST_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)  # Write header

    for row in tqdm(original_dataset, desc="Augmenting dataset with back-translation"):
        anchor = row[ANCHOR_IDX]
        positive = row[POSITIVE_IDX]
        negative = row[NEGATIVE_IDX]

        back_translated_anchors = back_translation_dict.get(anchor, [])

        # back_translated_positives = back_translation_dict.get(positive, [])
        # back_translated_positives.append(positive)

        back_translation_negatives = back_translation_dict.get(row[NEGATIVE_IDX], [])
        back_translation_negatives.append(negative)

        for _ in range(
            min(
                N_AUGMENTED,
                len(back_translated_anchors) * len(back_translation_negatives),
            )
        ):
            row[ANCHOR_IDX] = anchor
            row[POSITIVE_IDX] = random.choice(back_translated_anchors)
            row[NEGATIVE_IDX] = random.choice(back_translation_negatives)

            writer.writerow(row)
            n_rows += 1

print(f"Total rows written: {n_rows}")
