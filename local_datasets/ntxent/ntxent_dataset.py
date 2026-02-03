import csv
import random
from local_datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
import random
import json

random.seed(42)

load_dotenv()

DST_PATH = "local_datasets/ntxent_dataset_1.csv"
N_AUGMENTED = 1
MAX_SENTENCES = -1  # set to -1 to use all sentences

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

# read original dataset
original_sentences = []
with open("local_datasets/unique_sentences.jsonl", "r") as f:
    original_sentences = [json.loads(line)["sentence"] for line in f]

random.shuffle(original_sentences)
original_sentences = original_sentences[:MAX_SENTENCES]

# augment and write new dataset
n_rows = 0
with open(DST_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["view1", "view2"])  # Write header

    for view1 in tqdm(
        original_sentences, desc="Augmenting dataset with back-translation"
    ):
        augmented_sentences: list[str] = back_translation_dict.get(view1, [])
        if not augmented_sentences:
            continue

        for _ in range(min(N_AUGMENTED, len(augmented_sentences))):
            view2 = random.choice(augmented_sentences)
            augmented_sentences.remove(view2)

            writer.writerow([view1, view2])
            n_rows += 1

print(f"Total rows written: {n_rows}")
