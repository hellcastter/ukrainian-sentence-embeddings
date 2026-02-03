"""
Run: python3 -m local_datasets.semi_supervised_2.form_triplets
"""

import csv
import json
import random

import spacy
from tqdm import tqdm
from transformers import AutoTokenizer

from services.udpipe_model import UDPipeModel
from services.config import PATH_TO_SOURCE_UDPIPE

DATASET_PATH = "local_datasets/semi_supervised/merged_collected_and_generated.json"
OUTPUT_CSV = "local_datasets/semi_supervised_2/triplets_semi_supervised.csv"
TOKENIZER = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
BACK_TRANSLATION_PATH = (
    "local_datasets/translation/augmented_sentences_translated_v3.jsonl"
)
DEFINITIONS_BACK_TRANSLATION_PATH = (
    "local_datasets/translation/augmented_sentences_translated_definitions.jsonl"
)

MAX_SENTENCES_PER_MEANING = 50
USE_BACK_TRANSLATED = True
USE_DEFINITIONS_BACK_TRANSLATED = True

SCHEMA = [
    "lemma",
    "anchor",
    "positive",
    "negative",
    # "positives",
    "anchor_target_word_ids",
    "meaning_idx",
]


def get_recommended_number_of_sentences(
    n_sentences_per_meaning: int, index: int
) -> int:
    if n_sentences_per_meaning >= MAX_SENTENCES_PER_MEANING:
        return 1

    base = MAX_SENTENCES_PER_MEANING // n_sentences_per_meaning
    remainder = MAX_SENTENCES_PER_MEANING % n_sentences_per_meaning

    return base + 1 if index < remainder else base


def ngrams(s: str, n: int = 3) -> set[str]:
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def char_dice(a: str, b: str, n: int = 3) -> float:
    """
    Character n-gram Sørensen–Dice coefficient.
    Designed for fuzzy lemma matching (e.g., Ukrainian morphology noise).
    """

    # normalize
    a = a.lower().strip()
    b = b.lower().strip()

    # exact match shortcut
    if a == b:
        return 1.0

    # guard against garbage
    if not a or not b:
        return 0.0

    # if too short, don't lie
    if min(len(a), len(b)) < n:
        return 0.0

    na = ngrams(a, n)
    nb = ngrams(b, n)

    if not na or not nb:
        return 0.0

    return 2 * len(na & nb) / (len(na) + len(nb))


def same_lemma(a: str, b: str) -> bool:
    if a in b or b in a:
        return True

    # length guard
    if min(len(a), len(b)) < 4:
        return False

    # character 3-gram Jaccard
    if char_dice(a, b) >= 0.5:
        return True

    return False


def normalize_word(word: str) -> str:
    return (
        word.replace("«", "")
        .replace("»", "")
        .replace('"', "")
        .replace("“", "")
        .replace("”", "")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace(".", "")
        .strip()
    )


def _find_target_word_in_tokenized_text_new(tokenizer, tokenized_input_text, word: str):
    target_words_with_indexes = []

    word = word.strip().lower()
    current_word = ""
    start_index = 0

    zipped = zip(tokenized_input_text["input_ids"][0], tokenized_input_text.word_ids())
    for index, (input_id, word_id) in enumerate(zipped):
        token = tokenizer.decode([input_id]).strip()

        # Remove subword prefix if present
        if token.startswith("##"):
            token = token.replace("##", "")

        token = token.replace("▁", "").lower()

        current_word += token
        current_word_normalized = normalize_word(current_word)

        if word.startswith(current_word_normalized):
            if word == current_word_normalized:
                end_index = index
                target_words_with_indexes.append(
                    (word, list(range(start_index, end_index + 1)))
                )

                current_word = ""
                start_index = index + 1

        else:
            current_word = token
            start_index = index
            current_word_normalized = normalize_word(current_word)

            if word.startswith(current_word_normalized):
                if word == current_word_normalized:
                    end_index = index
                    target_words_with_indexes.append(
                        (word, list(range(start_index, end_index + 1)))
                    )

                    current_word = ""
                    start_index = index + 1
            else:
                current_word = ""
                start_index = index + 1

    return target_words_with_indexes


def _find_target_word_in_sentence(
    udpipe_model: UDPipeModel, input_text: str, target_word: str
):
    target_word = target_word.strip().lower()

    tokenized = udpipe_model.tokenize(target_word)
    udpipe_model.tag(tokenized[0])
    target_word_lemma = "".join([i.lemma.lower() for i in tokenized[0].words[1:]])

    tokenized = udpipe_model.tokenize(input_text)
    for tok_sent in tokenized:
        udpipe_model.tag(tok_sent)

        for word_index, w in enumerate(tok_sent.words[1:]):  # under 0 index is root
            token_lemma = w.lemma.lower()

            if same_lemma(target_word, token_lemma) or same_lemma(
                target_word_lemma, token_lemma
            ):
                return tok_sent.words[word_index + 1].form

    # use spacy as fallback
    target_word_lemma = "".join([i.lemma_.lower() for i in spacy_nlp(target_word)])

    doc = spacy_nlp(input_text)
    for token in doc:
        token_lemma = token.lemma_.lower()

        if same_lemma(target_word, token_lemma) or same_lemma(
            target_word_lemma, token_lemma
        ):
            return token.text

    return None


def get_target_word_embedding_idx(udpipe_model, tokenizer, sentence: str, lemma: str):
    word_in_sentence = _find_target_word_in_sentence(udpipe_model, sentence, lemma)

    if not word_in_sentence:
        return None, None

    inputs = tokenizer(sentence, return_tensors="pt")
    word_positions = _find_target_word_in_tokenized_text_new(
        tokenizer, inputs, word_in_sentence
    )

    if not word_positions:
        return word_in_sentence, None

    return word_positions[0]


def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=SCHEMA)
        writer.writeheader()

        for lemma, meanings in tqdm(data.items(), total=len(data)):
            for meaning_idx, meaning_data in enumerate(meanings.values()):
                positives = set(meaning_data["meaning"]["gloss"])

                negatives_dicts = [
                    p for i, p in enumerate(meanings.values()) if meaning_idx != i
                ]
                negatives = set()
                for neg_dict in negatives_dicts:
                    negatives.update(neg_dict["meaning"]["gloss"])

                if USE_DEFINITIONS_BACK_TRANSLATED:
                    for positive in list(positives):
                        positives.update(
                            definitions_back_translated_sentences.get(positive, [])
                        )

                    for negative in list(negatives):
                        negatives.update(
                            definitions_back_translated_sentences.get(negative, [])
                        )

                positives = list(positives)
                negatives = list(negatives)

                sentences = meaning_data["sentences"]
                sentences = random.sample(
                    sentences,
                    min(len(sentences), MAX_SENTENCES_PER_MEANING),
                )

                for index, sentence in enumerate(sentences):
                    anchors = set([sentence["sentence"]])
                    probability = sentence.get("probability")  # currently unused

                    recommended_sentences_with_anchor = (
                        get_recommended_number_of_sentences(len(sentences), index)
                    )

                    if USE_BACK_TRANSLATED:
                        anchors_augmented = back_translated_sentences.get(
                            sentence["sentence"]
                        )
                        if anchors_augmented is not None:
                            anchors.update(anchors_augmented)

                    anchors = list(anchors)

                    max_sentences = min(
                        recommended_sentences_with_anchor,
                        len(positives) * len(anchors) * len(negatives),
                    )

                    used = set()

                    for _ in range(max_sentences):
                        positive = random.choice(positives)
                        anchor = random.choice(anchors)
                        negative = random.choice(negatives)

                        while (anchor, positive, negative) in used:
                            positive = random.choice(positives)
                            anchor = random.choice(anchors)
                            negative = random.choice(negatives)

                        used.add((anchor, positive, negative))
                        _, anchor_target_ids = get_target_word_embedding_idx(
                            udpipe_model, tokenizer, anchor, lemma
                        )

                        if anchor_target_ids is None:
                            continue

                        row = {
                            "lemma": lemma,
                            "anchor": anchor,
                            "positive": positive,
                            "negative": negative,
                            # "positives": json.dumps(positives, ensure_ascii=False),
                            "anchor_target_word_ids": json.dumps(
                                anchor_target_ids, ensure_ascii=False
                            ),
                            "meaning_idx": meaning_idx,
                        }

                        writer.writerow(row)


if __name__ == "__main__":
    if USE_BACK_TRANSLATED:
        back_translated_sentences = {}
        print("Loading back-translated sentences...")
        with open(BACK_TRANSLATION_PATH, "r", encoding="utf-8") as bt_file:
            for line in bt_file:
                item = json.loads(line)
                back_translated_sentences[item["sentence"]] = item["augmented"]

    if USE_DEFINITIONS_BACK_TRANSLATED:
        definitions_back_translated_sentences = {}
        print("Loading definitions back-translated sentences...")
        with open(DEFINITIONS_BACK_TRANSLATION_PATH, "r", encoding="utf-8") as dbt_file:
            for line in dbt_file:
                item = json.loads(line)
                definitions_back_translated_sentences[item["sentence"]] = item[
                    "augmented"
                ]

    print("Loading NLP models...")
    spacy_nlp = spacy.load("uk_core_news_sm", enable=["lemmatizer"])
    udpipe_model = UDPipeModel(PATH_TO_SOURCE_UDPIPE)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    main()
