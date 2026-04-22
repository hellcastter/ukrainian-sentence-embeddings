"""
Run: python3 -m eval.eval_wsd
"""

import logging

from services.poolings import PoolingStrategy
from services.udpipe_model import UDPipeModel
from services.utils_results import results_reports
from services.utils_data import read_and_transform_data
from services.word_sense_detector import WordSenseDetector
from services.prediction_strategies import PredictionStrategy
from services.config import PATH_TO_SOURCE_UDPIPE, SUM_PATH
from services.utils_results import prediction_accuracy

import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME_OR_PATH = "models/fine-tuned-models/model_xwzpoedx_best"
MODEL_NAME_OR_PATH = "models/fine-tuned-models/model_sor6k453_final"

DEVICE = "cuda"  # or "cpu"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler("eval_wsd.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def evaluate_wsd(
    model_path: str,
    model_tokenizer_path: str | None = None,
    verbose: bool = True,
    sum_path: str = SUM_PATH,
    device: str = DEVICE,
):
    if model_tokenizer_path is None:
        model_tokenizer_path = model_path

    logger.info("Loading evaluation dataset...")
    data = read_and_transform_data(sum_path, homonym=True)

    logger.info("Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_tokenizer_path, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        model_path, output_hidden_states=True, trust_remote_code=True
    )
    model = model.to(device).eval()

    logger.info("Loading UDPipe model...")
    udpipe_model = UDPipeModel(PATH_TO_SOURCE_UDPIPE)

    logger.info("Running Word Sense Detection...")
    word_sense_detector = WordSenseDetector(
        pretrained_model=model,
        tokenizer=tokenizer,
        udpipe_model=udpipe_model,
        evaluation_dataset=data,
        pooling_strategy=PoolingStrategy.mean_pooling,
        prediction_strategy=PredictionStrategy.max_sim_across_all_examples,
        device = torch.device(device),
    )
    evaluation_dataset_pd = word_sense_detector.run()

    if verbose:
        results_reports(evaluation_dataset_pd, udpipe_model)

    return prediction_accuracy(evaluation_dataset_pd)


if __name__ == "__main__":
    evaluate_wsd(MODEL_NAME_OR_PATH, MODEL_NAME_OR_PATH, verbose=True)
