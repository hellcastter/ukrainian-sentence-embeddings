"""
Run: python3 -m augment.augment_translation_definitions
"""

import json
import os
import queue
import threading

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from augment.back_translator import (
    BackTranslator,
    HelsinkiCTranslateTranslator,
    NLLB200CTranslateTranslator,
    NLLB200TransformersTranslator,
)

INPUT_TEXTS_PATH = "local_datasets/semi_supervised/merged_collected_and_generated.json"
OUTPUT_TEXTS_PATH = (
    "local_datasets/translation/augmented_sentences_translated_definitions.jsonl"
)

BATCH_SIZE = 256
NUM_WORKERS = 2
NUM_AUGMENTATIONS = 4

# generating BATCH_SIZE x NUM_AUGMENTATIONS augmented per batch


class TextDataset(Dataset):
    def __init__(self, texts_file: str):
        with open(texts_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.texts = []

        for lemma, meanings in data.items():
            for meaning in meanings.values():
                self.texts.extend(meaning["meaning"]["gloss"])

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


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
                augmented_per_original = len(augmented) // len(batch)

                for i, sentence in enumerate(batch):
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

    texts_dataset = TextDataset(INPUT_TEXTS_PATH)
    dataloader = DataLoader(
        texts_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Open file for writing results immediately
    writer = ThreadedWriter(OUTPUT_TEXTS_PATH)

    try:
        for batch in tqdm(dataloader, desc="Augmenting"):
            augmented_texts = translator.augment(batch, n=NUM_AUGMENTATIONS)
            writer.write(batch, augmented_texts)
    finally:
        writer.close()


if __name__ == "__main__":
    main()

# ct2-transformers-converter --model facebook/nllb-200-3.3B --output_dir ./models/translators/nllb-200-3.3B
