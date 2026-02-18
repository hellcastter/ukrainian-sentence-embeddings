"""
Run: python3 -m eval.eval_mteb
"""

import mteb
from mteb.cache import ResultCache

from sentence_transformers import SentenceTransformer

import os
import warnings
import simplejson as json

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PATH_TO_SAVED_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
ALLOWED_MODALITIES = ["text"]
ALLOWED_LANGUAGES = ["ukr"]
NUM_PROC = 8


def main():
    model = SentenceTransformer(PATH_TO_SAVED_MODEL)

    # get tasks that support ukrainian language
    ukrainian_tasks = mteb.get_tasks(
        languages=ALLOWED_LANGUAGES, modalities=ALLOWED_MODALITIES
    )
    # get only tasks that support text modality (we don't want to evaluate on tasks that require image or audio inputs)
    ukrainian_tasks = [
        task
        for task in ukrainian_tasks
        if task.metadata.modalities == ALLOWED_MODALITIES
    ]

    print(f"Found {len(ukrainian_tasks)} tasks that support Ukrainian language.")

    # evaluate model on ukrainian tasks with caching
    cache = ResultCache(f"./cache/mteb_{PATH_TO_SAVED_MODEL.replace('/', '_')}")
    results = mteb.evaluate(
        model,
        tasks=ukrainian_tasks,
        cache=cache,
        num_proc=NUM_PROC,
        prediction_folder=f"./eval/mteb_prediction/{PATH_TO_SAVED_MODEL.replace('/', '_')}",
    )

    # save results to a file
    results_dir = f"./eval/mteb_results/{PATH_TO_SAVED_MODEL.replace('/', '_')}"
    os.makedirs(results_dir, exist_ok=True)

    for result in results.task_results:
        result_file = os.path.join(results_dir, f"{result.task_name}_results.json")

        with open(result_file, "w") as f:
            json.dump(result.scores, f, indent=2, ignore_nan=True)


if __name__ == "__main__":
    main()
