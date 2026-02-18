import json
import os
import queue
import threading

from torch.utils.data import Dataset

from services.udpipe_model import UDPipeModel
from services.utils_embedding_calculation_v2 import _find_target_word_in_sentence
from services.config import PATH_TO_SOURCE_UDPIPE


class TextDataset(Dataset):
    def __init__(self, texts_file: str, last_sentence: str = None, load_definitions=False):
        with open(texts_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.texts = []
        
        if load_definitions:
            for lemma, meanings in data.items():
                for meaning in meanings.values():
                    self.texts.extend([{"lemma": lemma, "sentence": gl} for gl in meaning["meaning"]["gloss"]])
        else:
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


class ThreadedWriter:
    def __init__(self, filepath, include_target_word_check=True):
        filepath_dir = os.path.dirname(filepath)
        if not os.path.exists(filepath_dir):
            os.makedirs(filepath_dir)

        self.f = open(filepath, "w", encoding="utf-8")
        self.queue = queue.Queue()
        self.finished = False
        self.thread = threading.Thread(target=self._write_loop, daemon=True)
        self.include_target_word_check = include_target_word_check
        
        if include_target_word_check:
            self.udpipe_model = UDPipeModel(PATH_TO_SOURCE_UDPIPE)

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

                    if self.include_target_word_check:
                        augmented_texts = [
                            augmented_sentence
                            for augmented_sentence in augmented_texts
                            if _find_target_word_in_sentence(
                                self.udpipe_model, augmented_sentence, lemma
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
