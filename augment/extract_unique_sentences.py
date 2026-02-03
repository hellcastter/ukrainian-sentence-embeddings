import csv
import json

from tqdm import tqdm

TRIPLETS_PATH = "local_datasets/ubertext_triplets.csv"
OUTPUT_PATH = "local_datasets/unique_sentences.jsonl"


def main():
    print(f"Extracting unique sentences from {TRIPLETS_PATH}...")
    with open(TRIPLETS_PATH, "r", encoding="utf-8") as infile:
        unique_sentences = set()

        reader = csv.reader(infile)
        header = next(reader)

        anchor_index = header.index("anchor")
        positive_index = header.index("positive")
        negative_index = header.index("negative")

        for row in tqdm(reader):
            unique_sentences.add(row[anchor_index])
            unique_sentences.add(row[positive_index])
            unique_sentences.add(row[negative_index])

    # save unique sentences to output file
    print(f"Saving {len(unique_sentences)} unique sentences to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as outfile:
        for sentence in tqdm(unique_sentences):
            json_line = json.dumps({"sentence": sentence}, ensure_ascii=False)
            outfile.write(json_line + "\n")


if __name__ == "__main__":
    main()
