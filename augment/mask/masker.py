import random


class Masker:
    def __init__(self, rate=0.15):
        self.rate = rate

    def _augment_sentence(self, sentence: str, n: int = 1):
        try:
            tokenized = self.udpipe_model.tokenize(sentence)[0]
            self.udpipe_model.tag(tokenized)
        except Exception as e:
            print(f"UDPipe failed to process sentence '{sentence}'")
            raise e

        words = tokenized.words[1:]  # under 0 index is root
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

    def augment(self, sentences: list[str], n: int = 1):
        """Augment sentences by dropping tokens.

        Args:
            sentences (list[str]): batch of sentences to augment
            n (int, optional): number of augmentations per sentence. Defaults to 1.
        """

        augmented_sentences = []
        for sentence in sentences:
            augmented_sentences.extend(self._augment_sentence(sentence, n))

        return augmented_sentences


if __name__ == "__main__":
    from services.config import PATH_TO_SOURCE_UDPIPE

    shuffler = Masker(rate=0.15)

    # Example usage
    sentence = "Він був дуже щасливий, коли отримав цю новину."
    augmented = shuffler.augment([sentence], n=5)
    print(augmented)
