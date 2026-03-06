import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from scipy.stats import wilcoxon

# --- Configuration ---
# Your original general-purpose model
MODEL_BASE_PATH = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# Your model tuned on the 3.5M WSD sentences
MODEL_WSD_PATH = "models/fine-tuned-models/model_xwzpoedx_final"
BENCHMARK_HF = "anikol12/STSB-UK"
N_BOOTSTRAPS = 5_000  # Number of iterations for the stress test


def get_cosine_similarities(model, sentences1, sentences2):
    """Generates embeddings and computes pairwise cosine similarity."""
    print(f"Generating embeddings for {model.model_card_data.model_name or 'model'}...")

    # Use torch for fast, batched cosine similarity calculation
    emb1 = model.encode(sentences1, convert_to_tensor=True, show_progress_bar=True)
    emb2 = model.encode(sentences2, convert_to_tensor=True, show_progress_bar=True)

    sims = F.cosine_similarity(emb1, emb2).cpu().numpy()
    return sims


def paired_bootstrap_test(preds_base, preds_wsd, labels, n_iterations=10_000):
    """Runs a paired bootstrap test to compare two dependent correlations."""
    print(f"\nRunning {n_iterations} bootstrap iterations. This might take a minute...")
    n_samples = len(labels)
    differences = []

    # Calculate base metrics
    corr_base, _ = spearmanr(preds_base, labels)
    corr_wsd, _ = spearmanr(preds_wsd, labels)
    actual_diff = corr_base - corr_wsd

    for _ in range(n_iterations):
        # Sample with replacement
        indices = np.random.randint(0, n_samples, n_samples)

        b_labels = labels[indices]
        b_preds_base = preds_base[indices]
        b_preds_wsd = preds_wsd[indices]

        b_corr_base, _ = spearmanr(b_preds_base, b_labels)
        b_corr_wsd, _ = spearmanr(b_preds_wsd, b_labels)

        # Track the difference in correlation for this sample
        differences.append(b_corr_base - b_corr_wsd)

    differences = np.array(differences)

    # Calculate the two-sided p-value
    # How often is the difference <= 0 (meaning WSD model beat or tied the base model?)
    # p_value_1sided  = np.sum(differences <= 0) / n_iterations

    stat, p_value = wilcoxon(
        differences,
        alternative="greater",  # H1: mean difference > 0 (base model is better than WSD model)
    )

    # Calculate 95% Confidence Interval
    ci_lower = np.percentile(differences, 2.5)
    ci_upper = np.percentile(differences, 97.5)

    return actual_diff, p_value, ci_lower, ci_upper


def main():
    print("Loading STS-B dataset...")
    eval_dataset = load_dataset(BENCHMARK_HF, split="train")

    sentences1 = eval_dataset["sentence1"]
    sentences2 = eval_dataset["sentence2"]

    # Normalize scores between 0 and 1 if they aren't already,
    # though Spearman is rank-based so scaling won't actually change the correlation
    raw_scores = eval_dataset["score"]
    labels = np.array(
        [
            1.0 if s1 == s2 else s
            for s1, s2, s in zip(sentences1, sentences2, raw_scores)
        ]
    )

    # 1. Load both models
    print("\nLoading Base Model...")
    model_base = SentenceTransformer(MODEL_BASE_PATH)

    print("\nLoading WSD-Tuned Model...")
    model_wsd = SentenceTransformer(MODEL_WSD_PATH)

    # 2. Extract raw predictions
    preds_base = get_cosine_similarities(model_base, sentences1, sentences2)
    preds_wsd = get_cosine_similarities(model_wsd, sentences1, sentences2)

    # 3. Stress-test the difference
    diff, p_value, ci_lower, ci_upper = paired_bootstrap_test(
        preds_base, preds_wsd, labels, N_BOOTSTRAPS
    )

    # 4. The Verdict
    print("\n" + "=" * 40)
    print("THE VERDICT")
    print("=" * 40)
    print(f"Actual difference in Spearman: {diff:.5f}")
    print(f"95% Confidence Interval of diff: [{ci_lower:.5f}, {ci_upper:.5f}]")
    print(f"p-value: {p_value}")
    print("-" * 40)

    if p_value < 0.05:
        print("Result: SIGNIFICANT.")
        print(
            "The drop is real. The WSD tuning statistically warped your semantic space."
        )
    else:
        print("Result: INSIGNIFICANT.")
        print("The drop is statistical noise. Your model did not downgrade.")


if __name__ == "__main__":
    main()
