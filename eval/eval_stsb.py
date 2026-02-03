from datasets import load_dataset, load_from_disk
from sentence_transformers import SentenceTransformer, evaluation, models

eval_dataset = load_dataset("sentence-transformers/stsb")

sentences1 = eval_dataset["train"]["sentence1"]
sentences2 = eval_dataset["train"]["sentence2"]

scores = eval_dataset["train"]["score"]
# fix 1.0 scores for identical sentences
scores = [1.0 if s1 == s2 else s for s1, s2, s in zip(sentences1, sentences2, scores)]


path_to_save_model = "models/fine-tuned-models/model_-166_0"
base_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

transformer = models.Transformer(path_to_save_model, tokenizer_name_or_path=base_model)
pooling = models.Pooling(
    transformer.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True,
    pooling_mode_cls_token=False,
    pooling_mode_max_tokens=False,
)

model = SentenceTransformer(modules=[transformer, pooling])

evaluator = evaluation.EmbeddingSimilarityEvaluator(
    sentences1, sentences2, scores, name="sts-dev"
)

print(evaluator(model))

# # sentence-transformers/paraphrase-multilingual-mpnet-base-v2
# # {'sts-dev_pearson_cosine': 0.8314385469524757, 'sts-dev_spearman_cosine': 0.8214346284594765}
