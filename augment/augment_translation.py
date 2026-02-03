"""
Run: python3 -m augment.augment_translation
"""

import json
import os
import queue
import threading

import spacy
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from augment.back_translator import (
    BackTranslator,
    HelsinkiCTranslateTranslator,
    NLLB200CTranslateTranslator,
    NLLB200TransformersTranslator,
)

from services.udpipe_model import UDPipeModel

UDPIPE_MODEL = "models/20180506.uk.mova-institute.udpipe"

udpipe_model = UDPipeModel(UDPIPE_MODEL)
spacy_nlp = spacy.load("uk_core_news_sm", enable=["lemmatizer"])

INPUT_TEXTS_PATH = "local_datasets/semi_supervised/merged_collected_and_generated.json"
OUTPUT_TEXTS_PATH = "local_datasets/translation/augmented_sentences_translated_v3.jsonl"

BATCH_SIZE = 256
NUM_WORKERS = 2
NUM_AUGMENTATIONS = 4

# generating BATCH_SIZE x NUM_AUGMENTATIONS augmented per batch


class TextDataset(Dataset):
    def __init__(self, texts_file: str, last_sentence: str = None):
        with open(texts_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.texts = []

        for lemma, meanings in data.items():
            for meaning in meanings.values():
                self.texts.extend(
                    [
                        {"lemma": lemma, "sentence": sentence["sentence"]}
                        for sentence in meaning["sentences"]
                    ]
                )

        if last_sentence is not None:
            try:
                for index, item in enumerate(self.texts):
                    if item["sentence"] == last_sentence:
                        last_sentence = index
                        self.texts = self.texts[last_sentence + 1 :]
                        break
            except ValueError:
                pass  # last_sentence not found, keep all texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


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

            if target_word in token_lemma or target_word_lemma in token_lemma:
                return tok_sent.words[word_index + 1].form

    # use spacy as fallback
    target_word_lemma = "".join([i.lemma_.lower() for i in spacy_nlp(target_word)])

    doc = spacy_nlp(input_text)
    for token in doc:
        token_lemma = token.lemma_.lower()

        if target_word in token_lemma or target_word_lemma in token_lemma:
            return token.text

    return None


class ThreadedWriter:
    def __init__(self, filepath):
        self.f = open(filepath, "a", encoding="utf-8")
        self.queue = queue.Queue()
        self.finished = False
        self.thread = threading.Thread(target=self._write_loop, daemon=True)
        self.thread.start()

    def write(self, original_texts, batch_texts):
        self.queue.put((original_texts, batch_texts))

    def _write_loop(self):
        while not self.finished or not self.queue.empty():
            try:
                batch, augmented = self.queue.get(timeout=1)
                # Write chunk at once to minimize disk syscalls
                write_strs = []
                augmented_per_original = len(augmented) // len(batch["sentence"])

                for i, (lemma, sentence) in enumerate(
                    zip(batch["lemma"], batch["sentence"])
                ):
                    augmented_texts = set(
                        augmented[
                            i
                            * augmented_per_original : (i + 1)
                            * augmented_per_original
                        ]
                    )
                    augmented_texts = [
                        i
                        for i in augmented_texts
                        if i.strip()
                        != sentence.strip()  # not the same as not-augmented sentence
                        and i.strip() != ""  # not empty
                        and "<unk>" not in i  # not containing unknown tokens
                    ]

                    augmented_texts = [
                        augmented_sentence
                        for augmented_sentence in augmented_texts
                        if _find_target_word_in_sentence(
                            udpipe_model, augmented_sentence, lemma
                        )
                        is not None
                    ]

                    write_strs.append(
                        json.dumps(
                            {"sentence": sentence, "augmented": augmented_texts},
                            ensure_ascii=False,
                        )
                    )

                self.f.write("\n".join(write_strs) + "\n")
                self.queue.task_done()
            except queue.Empty:
                continue

        self.f.close()

    def close(self):
        self.finished = True
        self.thread.join()


def main():
    # pivot = NLLB200CTranslateTranslator(
    #     "models/translators/nllb-200-3.3B",
    #     ["uk", "en"],
    #     device="cuda",
    #     device_index=[0, 1],
    # )

    pivot1 = HelsinkiCTranslateTranslator(
        "models/translators/opus-mt-zle-en-ct2",
        "Helsinki-NLP/opus-mt-tc-big-zle-en",
        device="cuda",
        device_index=[0],
    )

    pivot2 = HelsinkiCTranslateTranslator(
        "models/translators/opus-mt-en-zle-ct2",
        "Helsinki-NLP/opus-mt-tc-big-en-zle",
        device="cuda",
        device_index=[0],
    )

    translator = BackTranslator(
        pivot_models=[pivot1, pivot2], languages=[("uk", "en"), ("en", "uk")]
    )

    last_processed_sentence = None
    if os.path.exists(OUTPUT_TEXTS_PATH):
        with open(OUTPUT_TEXTS_PATH, "r", encoding="utf-8") as f:
            for line in reversed(list(f)):
                data = json.loads(line)
                if "original" in data:
                    last_processed_sentence = data["original"]
                    print(f"Resuming from sentence: {last_processed_sentence}")
                    break
    else:
        print("No existing output file found, starting fresh.")

    texts_dataset = TextDataset(INPUT_TEXTS_PATH, last_sentence=last_processed_sentence)
    dataloader = DataLoader(
        texts_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Open file for writing results immediately
    writer = ThreadedWriter(OUTPUT_TEXTS_PATH)

    try:
        for batch in tqdm(dataloader, desc="Augmenting"):
            augmented_texts = translator.augment(batch["sentence"], n=NUM_AUGMENTATIONS)
            writer.write(batch, augmented_texts)
    finally:
        writer.close()


if __name__ == "__main__":
    main()

# ct2-transformers-converter --model facebook/nllb-200-3.3B --output_dir ./models/translators/nllb-200-3.3B
