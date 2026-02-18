from datasets import load_dataset
from sentence_transformers import SentenceTransformer, evaluation


model_name_or_path = "models/fine-tuned-models/model_mvqvd4tm_best"
benchmark_hf = "sentence-transformers/stsb"


def main():
    ## load benchmark
    # there is not test split for this benchmark, so we use train split for evaluation
    eval_dataset = load_dataset(benchmark_hf, split="train")

    sentences1 = eval_dataset["sentence1"]
    sentences2 = eval_dataset["sentence2"]

    scores = eval_dataset["score"]
    # fix 1.0 scores for identical sentences
    scores = [
        1.0 if s1 == s2 else s for s1, s2, s in zip(sentences1, sentences2, scores)
    ]

    ## load model
    model = SentenceTransformer(model_name_or_path)

    ## evaluate model
    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        sentences1, sentences2, scores, show_progress_bar=True
    )

    print(f"Evaluating model {model_name_or_path} on STS benchmark...")
    print(evaluator(model))

    # sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    # {'pearson_cosine': 0.8630022521158784, 'spearman_cosine': 0.8592535781144339}
    # pool_targets {'pearson_cosine': 0.8530037387492666, 'spearman_cosine': 0.8548085691234922}
    # don't pool targets {'pearson_cosine': 0.8555752686341339, 'spearman_cosine': 0.8549837409156329}

    # intfloat/multilingual-e5-large
    # {'pearson_cosine': 0.8545772316485915, 'spearman_cosine': 0.8525929803955488}

    # intfloat/multilingual-e5-large-instruct
    # {'pearson_cosine': 0.8455628969019475, 'spearman_cosine': 0.8568462550313934}


if __name__ == "__main__":
    main()
