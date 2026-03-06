import random

from services.udpipe_model import UDPipeModel
from augment.common import Augmenter


class Dropouter(Augmenter):
    def __init__(self, udpipe_model: UDPipeModel, rate=0.15):
        self.udpipe_model = udpipe_model
        self.rate = rate

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
            augmented_sentence = []

            for i, word in enumerate(words):
                if random.random() <= self.rate:
                    continue

                augmented_sentence.append(
                    words[i - 1].getSpacesAfter() if i > 0 else ""
                )
                augmented_sentence.append(word.form)

            augmented_sentence = "".join(augmented_sentence)
            augmented_sentences.append(augmented_sentence)

        return augmented_sentences

    def __call__(self, sentences: list[str], n: int = 1) -> dict[str, list[str]]:
        """Augment sentences by dropping tokens.

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
    dropouter = Dropouter(udpipe_model, rate=0.15)

    # Example usage
    sentences = [
        "Він був дуже щасливий, коли отримав цю новину.",
        "Але іноді життя підкидає несподіванки, які змінюють наші плани.",
    ]
    augmented = dropouter(sentences, n=5)

    for original, augs in augmented.items():
        print(f"Original: {original}")
        for i, aug in enumerate(augs, 1):
            print(f"\tAugmentation {i}: {aug}")
        print()

# python3 -m augment.dropout.dropouter
