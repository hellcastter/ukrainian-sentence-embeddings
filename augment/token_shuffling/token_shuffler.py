import math
import random

from services.udpipe_model import UDPipeModel
from augment.common import Augmenter


class TokenShuffler(Augmenter):
    def __init__(self, udpipe_model: UDPipeModel, window_size=3):
        self.udpipe_model = udpipe_model
        self.window_size = window_size

    def _local_shuffle(self, tokens, window=3):
        tokens = tokens.copy()

        for i in range(0, len(tokens), math.ceil(window / 2)):
            chunk = tokens[i : i + window]
            random.shuffle(chunk)
            tokens[i : i + window] = chunk

        return tokens

    def _augment_sentence(self, sentence: str, n: int = 1):
        try:
            tokenized = self.udpipe_model.tokenize(sentence)
            words = []
            for tok in tokenized:
                self.udpipe_model.tag(tok)
                words.extend(tok.words[1:])
        except Exception as e:
            print(f"UDPipe failed to process sentence '{sentence}'")
            raise e

        augmented_sentences = []

        for _ in range(n):
            # shuffle only between punctuation to preserve sentence structure
            augmented_sentence = []
            current_sentence = []

            for word in words:
                if word.upostag not in ["PUNCT"]:
                    current_sentence.append(word.form)
                    continue

                augm_part = " ".join(
                    self._local_shuffle(current_sentence, self.window_size)
                )
                augm_part += word.form
                augmented_sentence.append(augm_part)
                current_sentence = []

            # if there are remaining tokens after the last punctuation
            if current_sentence:
                augm_part = " ".join(
                    self._local_shuffle(current_sentence, self.window_size)
                )
                augmented_sentence.append(augm_part)

            augmented_sentence = " ".join(augmented_sentence)
            augmented_sentences.append(augmented_sentence)

        return augmented_sentences

    def __call__(self, sentences: list[str], n: int = 1) -> dict[str, list[str]]:
        """Augment sentences by shuffling tokens within a window.

        Args:
            sentences (list[str]): batch of sentences to augment
            n (int, optional): number of augmentations per sentence. Defaults to 1.
        """

        augmented_sentences = {}
        for sentence in sentences:
            augmented_sentences[sentence] = self._augment_sentence(sentence, n)

        return augmented_sentences


if __name__ == "__main__":
    from services.config import PATH_TO_SOURCE_UDPIPE

    udpipe_model = UDPipeModel(PATH_TO_SOURCE_UDPIPE)
    shuffler = TokenShuffler(udpipe_model)

    # Example usage
    sentence = "Він був дуже щасливий, коли отримав цю новину."
    augmented = shuffler([sentence], n=5)
    print(augmented)
