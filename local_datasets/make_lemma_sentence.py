import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import json

df = pd.read_csv("local_datasets/ubertext_triplets.csv")

# group by 'lemma' and aggregate anchor, positive, negative into 1 list
grouped = (
    df.groupby("lemma")
    .agg(
        {
            "anchor": lambda x: list(x),
            "positive": lambda x: list(x),
            "negative": lambda x: list(x),
        }
    )
    .reset_index()
)

# create new dataframe with sentences
sentences = defaultdict(set)
for _, row in tqdm(grouped.iterrows(), total=grouped.shape[0]):
    lemma = row["lemma"]
    anchors = row["anchor"]
    positives = row["positive"]
    negatives = row["negative"]

    sentences[lemma].update(anchors)
    sentences[lemma].update(positives)
    sentences[lemma].update(negatives)

with open("local_datasets/lemma_sentences.jsonl", "w", encoding="utf-8") as f:
    for lemma, sents in tqdm(sentences.items(), total=len(sentences)):
        entry = {"lemma": lemma, "sentences": list(sents)}
        f.write(json.dumps(entry) + "\n")
