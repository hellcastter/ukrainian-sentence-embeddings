import csv
import json
from collections import defaultdict

from tqdm import tqdm

INPUT_CSV_PATH = "./datasets/ubertext_triplets.csv"
OUTPUT_JSONL_PATH = "./datasets/ubertext_unique_sentences.jsonl"

def main(): 
    unique_sentences = defaultdict(set)

    with open(INPUT_CSV_PATH) as f:
        reader = csv.DictReader(f)

        for row in tqdm(reader):
            lemma = row["lemma"]
            anchor = row["anchor"]
            positive = row["positive"]
            negative = row["negative"]
            
            unique_sentences[lemma].add(anchor)
            unique_sentences[lemma].add(positive)
            unique_sentences[lemma].add(negative)


    with open(OUTPUT_JSONL_PATH, "w", newline="") as out_f:
        for lemma, sentences in tqdm(unique_sentences.items()):
            row = {
                "lemma": lemma,
                "sentences": list(sentences)
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
