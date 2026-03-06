"""
Run: python3 -m eval.eval_mteb
"""

import mteb
from mteb.cache import ResultCache

from sentence_transformers import SentenceTransformer

import logging
import os
import warnings
import simplejson as json
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

ALLOWED_MODALITIES = ["text"]
ALLOWED_LANGUAGES = ["ukr"]
NUM_PROC = 8

CASE = 1

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


models = [
    # base models
    "lang-uk/ukr-paraphrase-multilingual-mpnet-base",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
]
DEVICE = None
 
if CASE == 0:   
    # pool targets: true
    models.extend([
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
        "models/fine-tuned-models/model_eo7hmwqv_final"
        
    ])
    DEVICE = "cuda:0"
elif CASE == 1:
    # pool targets: false
    models.extend([
        "models/fine-tuned-models/model_xwzpoedx_final",
        "models/fine-tuned-models/model_7wd97f4o_final",
        "models/fine-tuned-models/model_u9l23623_final",
        "models/fine-tuned-models/model_pt0axf82_final",
        "models/fine-tuned-models/model_1ezktszs_final",
        "models/fine-tuned-models/model_ok0ia00j_final",
        "models/fine-tuned-models/model_rpwv6n2t_final",
        "models/fine-tuned-models/model_a3eh99hl_final",
        "models/fine-tuned-models/model_8099d7r8_final",
        "models/fine-tuned-models/model_etkxukg9_final"
    ])
    DEVICE = "cuda:1"


def main():
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

    for model_name_or_path in tqdm(models, desc="Evaluating models"):
        model = SentenceTransformer(model_name_or_path, device=DEVICE)

        # evaluate model on ukrainian tasks with caching
        cache = ResultCache(f"./cache/mteb_{model_name_or_path.replace('/', '_')}")
        results = mteb.evaluate(
            model,
            tasks=ukrainian_tasks,
            cache=cache,
            num_proc=NUM_PROC,
            prediction_folder=f"./eval/mteb_prediction/{model_name_or_path.replace('/', '_')}",
        )

        # save results to a file
        results_dir = f"./eval/mteb_results/{model_name_or_path.replace('/', '_')}"
        os.makedirs(results_dir, exist_ok=True)

        for result in results.task_results:
            result_file = os.path.join(results_dir, f"{result.task_name}_results.json")

            with open(result_file, "w") as f:
                json.dump(result.scores, f, indent=2, ignore_nan=True)


if __name__ == "__main__":
    main()
