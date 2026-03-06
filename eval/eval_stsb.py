import pandas as pd

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, evaluation


model_name_or_path = "lang-uk/ukr-paraphrase-multilingual-mpnet-base"
benchmark_hf = "anikol12/STSB-UK"

# pool targets: false
# models = [
#     "lang-uk/ukr-paraphrase-multilingual-mpnet-base",
#     "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    # "models/fine-tuned-models/model_7wd97f4o_final",
    # "models/fine-tuned-models/model_u9l23623_final",
    # "models/fine-tuned-models/model_pt0axf82_final",
    # "models/fine-tuned-models/model_1ezktszs_final",
    # "models/fine-tuned-models/model_ok0ia00j_final",
    # "models/fine-tuned-models/model_rpwv6n2t_final",
    # "models/fine-tuned-models/model_a3eh99hl_final",
    # "models/fine-tuned-models/model_xwzpoedx_final",
    # "models/fine-tuned-models/model_8099d7r8_final",
# ]

# pool targets: true
models = [
    "lang-uk/ukr-paraphrase-multilingual-mpnet-base",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "models/fine-tuned-models/model_p1l04h2q_final",
    "models/fine-tuned-models/model_ksqah5x7_final",
    "models/fine-tuned-models/model_5yjgpx88_final",
    "models/fine-tuned-models/model_4gi9x91e_final",
    "models/fine-tuned-models/model_16mhe55g_final",
    "models/fine-tuned-models/model_69wvrfad_final",
    "models/fine-tuned-models/model_rwy4jgup_final",
    "models/fine-tuned-models/model_jrntf6jg_final",
    "models/fine-tuned-models/model_r4z3fy8z_final",
    "models/fine-tuned-models/model_vbngi6nk_final",
]

def main():
    # load dataset
    eval_dataset = load_dataset(benchmark_hf, split="train")

    sentences1 = eval_dataset["sentence1"]
    sentences2 = eval_dataset["sentence2"]
    scores = [
        1.0 if s1 == s2 else s
        for s1, s2, s in zip(sentences1, sentences2, eval_dataset["score"])
    ]

    evaluator = evaluation.EmbeddingSimilarityEvaluator(
        sentences1, sentences2, scores, show_progress_bar=True
    )

    rows = []

    for model_name_or_path in models:
        print(f"Evaluating {model_name_or_path}")

        model = SentenceTransformer(model_name_or_path)
        results = evaluator(model)

        rows.append({
            "model": model_name_or_path,
            "pearson_cosine": results["pearson_cosine"] * 100,
            "spearman_cosine": results["spearman_cosine"] * 100,
        })

    # ✅ build table
    df = pd.DataFrame(rows).set_index("model").sort_values(
        "spearman_cosine", ascending=False
    )

    print("\nFinal table:")
    print(df.round(4))

    # optional save
    df.to_csv("sts_results.csv")


if __name__ == "__main__":
    main()

    # sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    # {'pearson_cosine': 0.8630022521158784, 'spearman_cosine': 0.8592535781144339}
    # pool_targets {'pearson_cosine': 0.8530037387492666, 'spearman_cosine': 0.8548085691234922}
    # don't pool targets {'pearson_cosine': 0.8555752686341339, 'spearman_cosine': 0.8549837409156329}

    # intfloat/multilingual-e5-large
    # {'pearson_cosine': 0.8545772316485915, 'spearman_cosine': 0.8525929803955488}

    # intfloat/multilingual-e5-large-instruct
    # {'pearson_cosine': 0.8455628969019475, 'spearman_cosine': 0.8568462550313934}

