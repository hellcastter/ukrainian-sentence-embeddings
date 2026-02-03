from local_datasets import load_dataset, DatasetDict
from local_datasets import Features, Value, Sequence
from dotenv import load_dotenv
load_dotenv()

OUTPUT_TEXTS_PATH = "data/augmented_sentences_translated.jsonl"  # new back_translation file
REPO_ID = "hellcaster/wsd-sentences"

# 1) load dataset dict from the Hub
dataset_dict = load_dataset(REPO_ID)  # loads existing splits

# 2) detect the "original" split name (robust to either "origin" or "original")
orig_split = None
for candidate in ("origin", "original"):
    if candidate in dataset_dict:
        orig_split = candidate
        break
if orig_split is None:
    # fallback: pick the first split if neither is present
    orig_split = list(dataset_dict.keys())[0]

print(f"Detected original split named: '{orig_split}'")

# 3) add 'augmented' column (empty list) if missing
if "augmented" not in dataset_dict[orig_split].column_names:
    def add_empty_augmented(examples):
        # number of rows in this batch
        n = len(next(iter(examples.values())))
        # return a list-of-lists: one empty list per example
        return {"augmented": [[] for _ in range(n)]}

    print("Adding 'augmented' (empty list) column to original split...")
    dataset_dict[orig_split] = dataset_dict[orig_split].map(
        add_empty_augmented,
        batched=True,
        batch_size=5000,   # tune this if you run into memory/CPU issues
        remove_columns=None,  # keep existing columns
    )
    print("Done adding 'augmented'.")
else:
    print("'augmented' column already present in original split — skipping add.")

# 4) load new back_translation dataset from your jsonl (ensure augmented is a sequence of strings)
features = Features({
    "sentence": Value("string"),
    "augmented": Sequence(Value("string")),
})

print("Loading back_translation from jsonl...")
back_translation = load_dataset(
    "json",
    data_files=OUTPUT_TEXTS_PATH,
    features=features,
    split="train"
)

# 5) attach as back_translation split (overwrite if present)
dataset_dict["back_translation"] = back_translation
print("Attached 'back_translation' split with", len(back_translation), "examples.")

target_features = Features({
    "sentence": Value("string"),
    "augmented": Sequence(Value("string")),
})

# Cast original split to correct schema
dataset_dict[orig_split] = dataset_dict[orig_split].cast(target_features)

# back_translation already has correct features
dataset_dict["back_translation"] = dataset_dict["back_translation"].cast(target_features)

# Push
dataset_dict.push_to_hub(REPO_ID)
