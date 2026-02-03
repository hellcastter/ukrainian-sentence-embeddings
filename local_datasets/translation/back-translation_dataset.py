import csv
from local_datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
import random

random.seed(42)

load_dotenv()

SRC_PATH = "local_datasets/ubertext_triplets_200K.csv"
DST_PATH = "local_datasets/ubertext_triplets_200K_back_translation.csv"

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

with open(DST_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)  # Write header

    for row in tqdm(original_dataset, desc="Augmenting dataset with back-translation"):
        # write original row
        writer.writerow(row)

        # write back-translated row
        anchor = row[ANCHOR_IDX]
        positive = row[POSITIVE_IDX]
        negative = row[NEGATIVE_IDX]

        back_translated_anchors = back_translation_dict.get(anchor, [])
        back_translated_anchors.append(anchor)

        back_translated_positives = back_translation_dict.get(positive, [])
        back_translated_positives.append(positive)

        back_translated_negatives = back_translation_dict.get(negative, [])
        back_translated_negatives.append(negative)

        for _ in range(N_AUGMENTED):
            row[ANCHOR_IDX] = random.choice(back_translated_anchors)
            row[POSITIVE_IDX] = random.choice(back_translated_positives)
            row[NEGATIVE_IDX] = random.choice(back_translated_negatives)

            writer.writerow(row)
