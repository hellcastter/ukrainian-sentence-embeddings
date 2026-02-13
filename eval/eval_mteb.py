"""
Run: python3 -m eval.eval_mteb
"""

import mteb
from mteb.cache import ResultCache

from sentence_transformers import SentenceTransformer

import os
import json
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PATH_TO_SAVED_MODEL = "models/fine-tuned-models/model_7o3pod93_final"
ALLOWED_MODALITIES = ["text"]


def main():
    model = SentenceTransformer(PATH_TO_SAVED_MODEL)

    # get tasks that support ukrainian language
    ukrainian_tasks = mteb.get_tasks(languages=["ukr"], modalities=ALLOWED_MODALITIES)
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
        num_proc=8,
        prediction_folder=f"./eval/mteb_{PATH_TO_SAVED_MODEL.replace('/', '_')}",
    )

    # # save results to json file
    # with open(f"mteb_{PATH_TO_SAVED_MODEL.replace('/', '_')}.json", "w") as file:
    #     json.dump(
    #         [
    #             {
    #                 "task": result.task_name,
    #                 "scores": result.scores,
    #             }
    #             for result in results.task_results
    #         ],
    #         file,
    #         indent=2,
    # )


if __name__ == "__main__":
    main()
