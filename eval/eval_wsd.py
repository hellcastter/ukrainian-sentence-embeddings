"""
Run: python3 -m eval.eval_wsd
"""

from services.poolings import PoolingStrategy
from services.udpipe_model import UDPipeModel
from services.utils_results import results_reports
from services.utils_data import read_and_transform_data
from services.word_sense_detector import WordSenseDetector
from services.prediction_strategies import PredictionStrategy
from services.config import PATH_TO_SOURCE_UDPIPE, SUM_14_PATH, SUM_12_PATH
from services.utils_results import prediction_accuracy

from transformers import AutoTokenizer, AutoModel

MODEL_NAME_OR_PATH = "models/fine-tuned-models/model_u9l23623_best"
MODEL_NAME_OR_PATH = "models/fine-tuned-models/model_u9l23623_final"

DEVICE = "cuda"  # or "cpu"


def evaluate_wsd(
    model_path: str,
    model_tokenizer_path: str | None = None,
    verbose: bool = True,
    sum_path: str = SUM_14_PATH,
):
    if model_tokenizer_path is None:
        model_tokenizer_path = model_path

    print("Loading evaluation dataset...")
    data = read_and_transform_data(sum_path, homonym=True)

    print("Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_tokenizer_path, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        model_path, output_hidden_states=True, trust_remote_code=True
    )
    model = model.to(DEVICE).eval()

    print("Loading UDPipe model...")
    udpipe_model = UDPipeModel(PATH_TO_SOURCE_UDPIPE)

    print("Running Word Sense Detection...")
    word_sense_detector = WordSenseDetector(
        pretrained_model=model,
        tokenizer=tokenizer,
        udpipe_model=udpipe_model,
        evaluation_dataset=data,
        pooling_strategy=PoolingStrategy.mean_pooling,
        prediction_strategy=PredictionStrategy.max_sim_across_all_examples,
    )
    evaluation_dataset_pd = word_sense_detector.run()

    if verbose:
        results_reports(evaluation_dataset_pd, udpipe_model)

    return prediction_accuracy(evaluation_dataset_pd)


if __name__ == "__main__":
    evaluate_wsd(MODEL_NAME_OR_PATH, MODEL_NAME_OR_PATH, verbose=True)


# original model 0.701503
# -28_0 0.724572 (звичайний тюн)
# -29_0 0.734359 (анкор аугментований + позитив аугментований + негатив аугментований)
# -30_0 пішло вниз (анкор + анкор аугментований + позитив аугментований)
# -33_0 0.736805 (анкор + анкор аугментований + негатив аугментований)
# -36_0 0.723523 (анкор + анкор аугментований + рандомне речення з такою самою лемою аугментоване)
# -79_0 0.726319 (звичайний датасет, але зближуємо таргет ворд ембеддінги)

# -87_0 0.742747 (тріплети через спосіб, який я рекомедував. максимум 200 тріплетів на лему, по 5 позитиви на анкор)
# -88_0 0.742747 (тріплети через спосіб, який я рекомедував. по 5 позитиви на анкор)
# -94_0 0.743446 (SENTENCES_PER_MEANING = 50, SENTENCES_WITH_ANCHOR = 4, USE_BACK_TRANSLATED = True)
# -95_0 0.747641 (SENTENCES_PER_MEANING = 75, SENTENCES_WITH_ANCHOR = 4, USE_BACK_TRANSLATED = True)
# -9x_0 0.734 (SENTENCES_PER_MEANING = 75, SENTENCES_WITH_ANCHOR = 4, USE_BACK_TRANSLATED = False)
# -103_0 0.775952 (SENTENCES_PER_MEANING = 75, SENTENCES_WITH_ANCHOR = 4, USE_BACK_TRANSLATED = False, intfloat/multilingual-e5-large)

# -106_0 0.737854 (SENTENCES_PER_MEANING = 20, SENTENCES_WITH_ANCHOR = 4, USE_BACK_TRANSLATED = False, base model)
# -107_0 0.742048 (SENTENCES_PER_MEANING = 75, SENTENCES_WITH_ANCHOR = 4, USE_BACK_TRANSLATED = False, base model)
# -108_0 0.742747 (SENTENCES_PER_MEANING = 75, SENTENCES_WITH_ANCHOR = 4, USE_BACK_TRANSLATED = True, base model)

# -110_0 0.737854 (SENTENCES_PER_MEANING = 50, SENTENCES_WITH_ANCHOR = 2, USE_BACK_TRANSLATED = False, base model, new dataset formation)
# -110_final 0.737854 (SENTENCES_PER_MEANING = 50, SENTENCES_WITH_ANCHOR = 2, USE_BACK_TRANSLATED = False, base model, new dataset formation)

# -113_0 0.79972 (SENTENCES_PER_MEANING = 75, SENTENCES_WITH_ANCHOR = 2, USE_BACK_TRANSLATED = True, intfloat/multilingual-e5-large, new dataset formation)

# -163_0 0.789934
# -166_0 0.787836 (0.803842) (0.819805 on SUM12)
