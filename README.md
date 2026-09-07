# Improving Ukrainian word sense disambiguation with sense-aware sentence embeddings

Research code accompanying the manuscript by Victor Muryn and Yurii Laba, submitted to **PeerJ Computer Science** as an **AI Application** article. The approach adapts multilingual embeddings by pairing Ukrainian sentences with dictionary definitions of the meanings expressed by their target words.

**Reproduction status:** the repository contains corpus extraction, pseudo-labeling, benchmark-overlap filtering, augmentation, triplet construction, training, and embedding-based evaluation code. Hugging Face training inputs and a launcher for **144 jobs** (eight configurations × three dataset seeds × three training seeds × two pooling settings) are now implemented. It does **not yet provide an exact reproduction of all paper results**: the original environment, artifact-to-result manifest, complete dataset-construction provenance, and result aggregation remain missing; several implementation/manuscript discrepancies remain. Commands below describe the current implementation and state their prerequisites.

Quick navigation: [datasets](#dataset-information) · [code](#code-structure) · [installation](#installation-and-requirements) · [hardware](#hardware-requirements) · [usage](#usage-instructions) · [method](#methodology) · [reproduction steps](#reproducing-the-paper) · [configurations](#reproducing-experimental-configurations) · [results](#reproducing-paper-tables-and-results) · [seeds](#randomness-and-seeds) · [outputs](#expected-outputs) · [models](#pretrained-models-and-external-ai-models)

## Project Description

Word sense disambiguation (WSD) selects the meaning of an ambiguous word in context: for example, whether Ukrainian *коса* refers to a braid or a geographical feature. A dictionary supplies the possible meanings; contextual examples supply evidence for choosing among them.

This project retrieves sentences containing ambiguous lemmas from UberText 2.0, assigns provisional meanings by comparing sentence and definition embeddings, and retains confident assignments. It removes natural contexts that closely match dictionary evaluation examples, supplements underrepresented meanings with generated sentences, and creates transformed versions of sentences and definitions. A shared transformer encoder is then fine-tuned on sentence–definition triplets, bringing a sentence closer to its assigned definition and farther from another meaning of the same lemma.

The code supports investigating corpus/meaning coverage, training sense-aware embeddings, and evaluating WSD, Ukrainian STS-B, and Ukrainian MTEB tasks. Exploratory notebooks support coverage analysis. The zero-shot LLM comparison, human pseudo-label quality audit, sense-availability result tables, and nine-run statistical aggregation do not have complete executable reproduction workflows here. Generated training sentences use an LLM, but that script is **not** an LLM WSD evaluator.

## Dataset Information

Paths in tables below are **required external inputs or generated destinations**, not claims that those artifacts are included in a clone. Most data, checkpoints, and metric files are excluded by `.gitignore`. Python modules, notebooks, and configuration files linked elsewhere in this README are repository source files.

### External third-party resources

| Resource | Purpose and source | Acquisition and expected location/format | Preprocessing and redistribution status |
| --- | --- | --- | --- |
| Ukrainian dictionary/WSD source data | Definitions, ambiguous lemma inventory, and evaluation examples; prior benchmark described by [Laba et al. (2023)](https://aclanthology.org/2023.unlp-1.2/) | Manual. Current `services.config.SUM_PATH` is `datasets_pre_defined/sum_final.jsonlines`. Raw dictionary JSON Lines, schema below. **TODO: publish the exact download URL, snapshot date, and provenance.** The older README referred to `sum_fixed.jsonlines`; the active code overrides that path with `sum_final.jsonlines`. | Processed by `read_and_transform_data(..., homonym=True)` in `services/utils_data.py`. Underlying dictionary/benchmark license and redistribution permission are not recorded here. Do not infer permission from code availability. |
| Official expanded WSD benchmark | [yuriilaba/ukrainian-homonym-dict](https://huggingface.co/datasets/yuriilaba/ukrainian-homonym-dict), designated snapshot for the PeerJ article: **revision `07d2cd2`**. Verified: **1,464 lemmas, 3,071 meanings, 15,961 examples**. | Load through Hugging Face `datasets` with full revision `07d2cd250f1e17333a6fa233a6b9b0cf8c9789e2`, split `train`. Processed Parquet columns: `lemma`, `gloss`, `examples`; one row per meaning. No relocation or raw-file upload to GitHub is needed. | Contextual examples are reserved for evaluation; the lemma inventory and definitions also support training-data preparation/adaptation. The `train` split name is a storage convention. No dataset license is declared. The legacy raw-dictionary loader does not directly accept this processed schema. |
| UberText 2.0, sentence-split news, Wikipedia, and fiction | Naturally occurring unlabeled contexts; [project download page](https://lang.org.ua/en/ubertext/) and [corpus paper](https://aclanthology.org/2023.unlp-1.1/) | Manual download to `datasets_pre_defined/`; UTF-8 text, one sentence per line, compressed as `.txt.bz2`. The download commands below preserve URLs from the previous README. | Extraction normalizes whitespace, filters sentences, and matches lemmas with UDPipe or spaCy. Both analyzers' outputs are merged. License/redistribution terms are not preserved here; consult the corpus provider before redistributing extracted text. |
| Ukrainian STS-B, `anikol12/STSB-UK` | Sentence-level similarity evaluation; identifier in `eval/eval_stsb.py` | Downloaded through `datasets.load_dataset`, split `train`, into the library-managed cache. Expected columns: `sentence1`, `sentence2`, `score`. No dataset revision is specified. | Script changes the score of exactly identical sentence pairs to `1.0`; all other scores are used as supplied. The manuscript reports 5,749 pairs. Translation provenance, exact snapshot, and license need confirmation from the dataset provider. |
| Ukrainian text tasks selected by MTEB | Classification, clustering, retrieval, and bitext mining; task definitions come from the installed `mteb` package | Automatic task-specific downloads/caching. `mteb.get_tasks(languages=['ukr'], modalities=['text'])`, followed by an exact modality filter. No task/dataset revision or frozen task list is supplied. | Each underlying dataset has its own terms. The paper lists 11 tasks, but the script selects tasks dynamically, so a different MTEB installation can produce a different suite. |
| Ukrainian morphology resources | Corpus matching, target extraction, dictionary deduplication, and optional POS reports | Manual UDPipe weights expected at `models/20180506.uk.mova-institute.udpipe`; spaCy package `uk_core_news_sm` installed separately. Stanza Ukrainian resources may be downloaded when POS reports create a pipeline. | These are external model assets, not corpus files. Exact weight revisions and model-specific licenses are not recorded. The UDPipe model download URL/checksum is missing; installing the Python binding does not install this model. |

The 11 MTEB task names reported in the manuscript are SIB200, UkrFormal, SIB200ClusteringS2S, WebFAQQAs, WebFAQQuestions, NTREX, Bible-NLP, Flores, Tatoeba, Belebele, and WebFAQ. These are **manuscript labels**, not a verified list of runnable task identifiers for an unspecified MTEB version. Task-specific URLs, subsets, revisions, and license records must be recovered from the original evaluation environment.

### Official benchmark revision and loading

**Benchmark version referenced for the PeerJ Computer Science article: revision `07d2cd2`.** Its full immutable commit is `07d2cd250f1e17333a6fa233a6b9b0cf8c9789e2`. Use the revision-pinned dataset rather than an evolving `main` branch. See the [Hugging Face dataset card](https://huggingface.co/datasets/yuriilaba/ukrainian-homonym-dict) for the maintained dataset description.

```python
from datasets import load_dataset

benchmark = load_dataset(
    "yuriilaba/ukrainian-homonym-dict",
    revision="07d2cd250f1e17333a6fa233a6b9b0cf8c9789e2",
    split="train",
)
assert len(benchmark) == 3071
assert len(set(benchmark["lemma"])) == 1464
assert sum(len(examples) for examples in benchmark["examples"]) == 15961
```

The pinned `data/train-00000-of-00001.parquet` has SHA-256 `25f792dc3aad4f3349f643bbc1da1cb9889b35d0a18d2bb77ab71d9e7365a6f2`. These counts were measured from that file and are the corrected counts used for the manuscript. Dataset availability and version identity are established; original run records are still needed to verify the exact snapshot/subset behind each result.

The contextual evaluation sentences must not be treated as training anchors merely because the split is called `train`. Candidate definitions and the lemma inventory are used by the method during corpus selection, pseudo-labeling, and adaptation, so the evaluation-only statement applies to `examples`, not to every field.

### Legacy raw source schema and local-file identity

`read_and_transform_data` expects dictionary records with `lemma`, `prime`, `suffixes`, `tags`, `synsets`, `phrases`, `word_id`, and `url`. Each synset has `sense_id`, a list of `gloss` strings, and an `examples` list whose entries contain `ex_text`. It is not a loader for a flat sentence-label CSV. The processed in-memory table contains `lemma`, lists of `gloss` strings, and lists of `examples`, with one row per retained dictionary meaning.

The inspected raw records contain source links to the SUM-20 dictionary, for example [entry `wordid=1`](https://sum20ua.com/Entry/index?wordid=1). This identifies the underlying dictionary source. The official processed benchmark is the pinned Hugging Face dataset above; its upstream extraction date and complete construction manifest are still unrecorded.

The **local, untracked** raw file examined during README preparation had:

- Filename: `sum_final.jsonlines` at the repository root, rather than the configured data directory.
- Size: 456,658,460 bytes; 138,044 raw dictionary records.
- SHA-256: `6e4ecd7c9fde0a486826f6d033f14a7c020f9b503208abc4262cf0855401da6c`.

These are raw-file measurements, not counts or an identifier of the official expanded WSD benchmark. This raw file is not required to download the Hugging Face snapshot and does not need to be uploaded to GitHub. The existing raw-dictionary preparation/evaluation entry points still expect it; their output has not been checked for row-by-row equivalence to the published benchmark. Do not pass the processed three-column dataset through `read_and_transform_data`, which expects the nested raw schema.

### Project-generated data and intermediate artifacts

| Artifact | Producer and purpose | Format and destination |
| --- | --- | --- |
| Target lemma list | Derived from the processed dictionary; used for corpus extraction | One lemma per line at `datasets_pre_defined/unique_lemmas_homonyms.txt`; the reproduction guide provides the derivation command |
| Extracted corpus batches | `collect_sentences/collect_ubertext_sentences.py` | JSON Lines despite the `.json` suffix: each line maps lemmas to sentence lists; output chosen with `--save_dataset` |
| Deduplicated corpus contexts | `local_datasets/raw_sentences/process_raw_sentences.py` | `local_datasets/raw_sentences/unique_lemma_sentences.jsonl`; each record has `lemma` and `sentences` |
| Confident pseudo-label assignments | `local_datasets/semi_supervised_2/assign_meaning_to_sentence.py` | `assigned_meanings_mpnet.jsonl` in the same directory; `lemma`, `sentence`, `similarity`, `probability`, `assigned_meaning` |
| Meaning-organized natural pool | Same pseudo-label script | `lemmas_with_meanings_and_sentences_mpnet.json` in the same directory; lemma → first gloss → `{meaning: {gloss, examples}, sentences: [...]}`. Empty sentence lists preserve uncovered meanings. The `examples` metadata comes from the dictionary; it is not the anchor pool. |
| Natural pool after benchmark-overlap filtering | `local_datasets/semi_supervised_2/delete_similar_sentences.py`; removes natural contexts with cosine similarity ≥0.95 to any dictionary example in the input | `local_datasets/semi_supervised_2/lemmas_with_meanings_and_sentences_mpnet_filtered.json`; same nested schema, preserving meaning metadata. This is the input for generation and merging. |
| Generated sentences | `collect_sentences/generate_sentences_4_absent_meanings.py` | `local_datasets/semi_supervised_2/generated_sentences.jsonl`; nested lemma/meaning objects with generated sentence strings |
| Merged natural/generated pool | `local_datasets/semi_supervised_2/merge_collected_and_generated.py` | `local_datasets/semi_supervised_2/merged_collected_and_generated_mpnet.json`; generated entries carry `source='generated'` and null similarity/probability |
| Transformation outputs | Scripts under `augment/` | JSON Lines with `sentence` and `augmented` (list of strings), under `local_datasets/augmented/`; exact filenames appear under Expected Outputs |
| Training triplets | `local_datasets/semi_supervised_2/form_triplets.py` | CSV with `lemma,anchor,positive,negative,anchor_target_word_ids,meaning_idx`; target indices are a JSON-encoded list inside the CSV field |
| Fine-tuned checkpoints | `services/trainer/trainer.py` | Hugging Face encoder/tokenizer and mean-pooling SentenceTransformer exports under `models/fine-tuned-models/` |

The training launcher uses the project dataset [victormuryn/wsd-training-dataset](https://huggingface.co/datasets/victormuryn/wsd-training-dataset). `services/trainer/trainer.py` downloads a selected configuration's `train` split through `datasets.load_dataset`; files reside in the Hugging Face cache, with no required manual copy into this repository. The eight configuration prefixes are `raw`, `generated`, `mask`, `dropout`, `translation`, `token_shuffling`, `markov`, and `all_augs`, each with suffix `_seed42`, `_seed123`, or `_seed456`. These are prepared triplet datasets, separate from the evaluation benchmark. The active triplet trainer requires `anchor`, `positive`, and `negative` text columns and, for target pooling, compatible `anchor_target_word_ids`; the local builder's full schema is shown above.

Loading these prepared datasets allows training without repeating corpus extraction and augmentation. It does not rebuild their three construction seeds: no dataset-generation/upload driver or checksum manifest links the local builder to all 24 published configurations. The training loader does not pin a dataset revision. Record the resolved dataset identity and hashes with each run, and recover the paper's exact revision before claiming historical reproduction.

Project artifacts contain or derive from third-party text/model outputs. No separate generated-data license or complete artifact checksum manifest is recorded in this code repository. Their redistribution terms must be established separately from the code license.

Older triplet-rewriting scripts in `local_datasets/augmented/translation/` and the archived NT-Xent dataset builder automatically load the Hugging Face identifier `hellcaster/wsd-sentences`, split `back_translation`, expecting `sentence` and `augmented` columns. This is an additional historical project-data dependency, not an input to the active pipeline. No revision, license record, or confirmed mapping to the paper's datasets is supplied.

## Code Structure

```text
.
├── collect_sentences/             # Corpus extraction and LLM sentence generation
├── local_datasets/
│   ├── raw_sentences/             # Merge/deduplicate extracted batches
│   ├── semi_supervised_2/         # Active pseudo-labeling and triplet pipeline; EDA notebooks
│   │   ├── assign_meaning_to_sentence.py # Contexts + dictionary → confident natural pool
│   │   ├── delete_similar_sentences.py   # Natural pool → pool without benchmark overlaps
│   │   ├── merge_collected_and_generated.py # Filtered pool + generation JSONL → merged pool
│   │   └── form_triplets.py      # Selected pool + optional transformations → triplet CSV
│   ├── augmented/translation/     # Older triplet rewriting experiments
│   ├── archive/                   # Earlier pseudo-labeling, mining, and NT-Xent experiments
│   └── sum_and_ubertext_eda.ipynb  # Dictionary/corpus coverage exploration
├── augment/
│   ├── dropout/                   # Word deletion; sentences and definitions
│   ├── mask/                      # MLM replacement; sentences and definitions
│   ├── token_shuffling/           # Local word shuffling
│   ├── translation/               # OPUS back-translation and alternative translator classes
│   ├── common.py                 # Dataset loader, filtering writer, stochastic selector
│   └── augment_all_together*.py   # Stochastic sequential transformations
├── services/
│   ├── config.py                 # Dictionary/corpus/UDPipe paths and preprocessing constants
│   ├── utils_data.py             # Dictionary preparation and reporting features
│   ├── utils_embedding_calculation_v2.py  # Morphological matching and target-token alignment
│   ├── word_sense_detector.py     # Dictionary-sense evaluation loop
│   ├── prediction_strategies.py   # Example/definition similarity aggregation
│   ├── poolings.py               # Inference pooling implementations
│   └── trainer/
│       ├── trainer.py            # Optimizer, training, validation, checkpoint export
│       ├── training_config.py    # INI parser and fallback defaults
│       ├── fine_tuning_config.ini # Original experiment-oriented defaults
│       ├── reviewer_config.ini   # Current-code example with W&B disabled
│       ├── datasets.py           # Tokenization and dataset objects
│       ├── data_factory.py       # DataLoaders and collators
│       └── losses.py             # Triplet, MNR, and NT-Xent implementations
├── eval/                         # eval_wsd.py, eval_stsb.py, eval_mteb.py
├── train_all.sh                  # 144 HF-dataset training jobs across seeds/pooling settings
├── scripts/reproduce/environment_report.py # Local environment/asset report; no downloads
├── datasets_pre_defined/         # External inputs; normally only .gitkeep is committed
├── models/                       # External weights/checkpoints; normally only .gitkeep
├── demo.py                       # Qualitative target-word/definition comparison
├── requirements.txt              # Unpinned dependency inventory, added for this README
├── requirements-notebooks.txt    # Optional notebook tools
└── .env.example                  # Optional W&B credential template
```

The active pipeline is `semi_supervised_2`, not `archive/semi_supervised`. Archived scripts use different thresholds and hard/semi-hard negative mining, including missing historical checkpoints. They are retained for provenance and are not substitutes for the paper's random-negative triplet construction. The clustering notebooks contain exploratory settings that differ from the active pseudo-label script; clustering is not a required pipeline stage.

## Installation and Requirements

### Version evidence

**Python 3.10.14 is recorded in all five committed notebooks.** The previous README stated Python 3.10+, and source annotations require Python 3.10 syntax. This establishes a recorded notebook environment, not a verified training environment or a guarantee of compatibility with every newer Python release.

| Software | Repository evidence | Exact experiment version |
| --- | --- | --- |
| Python | Notebook `metadata.language_info.version` and one kernel display name | 3.10.14 for notebooks; training interpreter unrecorded |
| PyTorch | `torch.amp.GradScaler`, `autocast`, AdamW, DataLoader | Not pinned |
| Transformers / SentenceTransformers | `AutoModel`, tokenizers, pipelines, `SentenceTransformer` | Neither pinned |
| datasets / tokenizers / accelerate | Dataset loading; tokenizer functionality; low-memory model-loading support in an alternative translator | None pinned; `accelerate` is a support dependency rather than a direct import |
| scikit-learn / SciPy | Cosine similarity, metrics, softmax; notebook PCA/KMeans | Neither pinned |
| NumPy / pandas | Data preparation, sampling, metrics | Neither pinned |
| MTEB | `mteb.evaluate`, `ResultCache`, dynamic task discovery | Not pinned; API and task registry compatibility remain to be validated |
| spaCy / `uk_core_news_sm` | Lemmatization and dictionary deduplication | Neither library nor model pinned |
| UDPipe bindings / Ukrainian weights | `ufal.udpipe`; dated model filename | No committed package pin. Local untracked archive `ufal.udpipe-1.2.0.1.tar.gz` has package metadata version **1.2.0.1**, but does not establish the experiment version. Weight checksum absent. |
| Stanza / pymorphy2 | Imported by shared data utilities; POS-reporting alternatives | Neither pinned |
| CTranslate2 / SentencePiece | Translation implementations | Neither pinned; converted OPUS weights absent |
| OpenAI Python client / generation server | Local OpenAI-compatible chat completion request | Client, server, model revision, quantization, and decoding defaults unrecorded |
| Other runtime/notebook packages | `smart-open`, `langdetect`, `tqdm`, `wandb`, `python-dotenv`, `simplejson`, plotting/notebook tools | Not pinned |

No original requirements/lockfile, Conda environment, Dockerfile, Slurm job, committed W&B run export, or CUDA/driver version record was found. `train_all.sh` now supplies an experiment driver, but no environment lock. The requirements files list dependencies inferred from imports; **they are not a recovered or validated paper environment**. Installing currently resolved versions can encounter API incompatibilities. Replace them with a tested, fully pinned environment once the authors recover the original records; do not label a new environment as the historical one.

### Environment setup

Run commands from the repository root. Prepare Python 3.10.14 separately if matching the notebook interpreter, then check which interpreter `python3` selects:

```bash
python3 --version
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python -m spacy download uk_core_news_sm
```

The package installation command is a **bootstrap attempt**, not a validated lockfile installation. For GPU work, the installed PyTorch and CTranslate2 builds must be compatible with the machine's NVIDIA driver/CUDA runtime; no exact CUDA installation command can be recovered from this repository. Building `ufal.udpipe` from source, if no suitable wheel is available, requires a working C++ build toolchain. The local source archive is not required by the requirements file and does not contain the Ukrainian model weights.

For notebooks:

```bash
python -m pip install -r requirements-notebooks.txt
python -m jupyterlab
```

Open each notebook with its own directory as the kernel working directory: relative paths and `sys.path` adjustments assume this. The two coverage notebooks are `local_datasets/sum_and_ubertext_eda.ipynb` and `local_datasets/semi_supervised_2/eda_lemmas_with_meanings.ipynb`.

Inspect the current machine without importing NLP models or downloading anything:

```bash
python scripts/reproduce/environment_report.py
```

After an environment has actually passed validation, record it for that new run:

```bash
mkdir -p logs
python scripts/reproduce/environment_report.py > logs/environment-report.json
python -m pip freeze > logs/requirements-resolved.txt
```

These are newly created local records, not original paper metadata. They are ignored by Git and need to be included deliberately in an eventual experiment archive. `pip freeze` does not capture model revisions, system libraries, CUDA drivers, or data hashes.

### Assets, authentication, and external services

- Load the official processed benchmark from the pinned Hugging Face revision above. The current legacy raw-data scripts additionally need their raw dictionary input; obtain that input and the UDPipe weights separately, as described in Dataset Information. spaCy is loaded at import time by shared embedding utilities, so it is needed even for several nominally UDPipe-based entry points.
- Hugging Face model loaders normally download weights/tokenizers when missing from the cache. Active model loaders and the training dataset loader have no revision pin; the official evaluation benchmark is pinned in the loading example above. The trainer reads optional `HF_TOKEN` from the environment and passes it to `load_dataset`. No gated-access requirement is recorded; supply a token if your chosen resource requires access. `.env.example` currently documents W&B only.
- The matrix launcher requires Bash, `flock`, `mktemp`, `sed`, and `python3` from the activated environment. Check `command -v flock` before launching; provide a system package supplying `flock` if absent. The launcher is intended for a CUDA/Linux environment and has no CPU mode.
- Back-translation requires **already converted CTranslate2 OPUS models** at `models/translators/opus-mt-zle-en-ct2` and `models/translators/opus-mt-en-zle-ct2`. Tokenizers download automatically; converted weights do not. The repository has an NLLB converter command in comments, but no validated OPUS conversion recipe, source revisions, or conversion metadata. Obtain the original artifacts or recover and validate that recipe before running translation.
- Training loads `.env`. The original INI enables W&B under an author-specific entity. Use `reviewer_config.ini` to disable W&B, or set `wandb_entity`, `wandb_project_name`, and optionally `wandb_run_name` in your INI and supply `WANDB_API_KEY` using `.env.example` as a template. No author-account access is needed when logging is disabled.
- Generation uses `BASE_URL='http://localhost:8000/v1'`, `API_KEY='EMPTY'`, and `MODEL_NAME='Qwen/Qwen3-VL-8B-Instruct'` in its Python script. These are constants, **not environment variables**. An OpenAI-compatible server must already be serving that identifier. The earlier README identifies llama.cpp as the original server, but no launch command, version, quantization, or server configuration is supplied. The generation script does not launch the server or download Qwen weights.

## Hardware Requirements

The manuscript reports training on **one NVIDIA RTX 3090**. The trainer uses one selected device, defaults to CUDA when available, and accepts `--device cuda:0` or `--device cpu`. Although `enable_gpu_parallel=True` appears in the configuration, the current trainer does not implement that flag. It is not evidence of multi-GPU training.

`train_all.sh` defaults to two independent workers on `cuda:0` and `cuda:1`, with one training run per worker. Set `--num-gpus 1` to run its 144 jobs sequentially on one GPU. This is job scheduling, not distributed training of one model. The overlap filter lets SentenceTransformer select its device; it stores embeddings for all natural contexts and dictionary examples in memory and computes similarities in chunks of 2,048 contexts, so chunking does not bound total embedding storage.

CUDA training uses FP16 autocast and gradient scaling; CPU training disables AMP. Input sequences are padded/truncated to 128 subword tokens. The configured batch size is 104, and all encoder layers are trainable by default. No measured peak VRAM, RAM minimum, driver version, or minimum GPU specification is committed; the reported GPU is a reference machine rather than a validated minimum.

CPU training and embedding evaluation have code paths, but have not been validated as a complete reproduction workflow. Pseudo-labeling hard-codes `model_device='cuda'`; change that constant for CPU. MLM selects CUDA when available but requests FP16 even on CPU, so CPU compatibility is not established. OPUS translation uses CUDA and float16 explicitly. Corpus multiprocessing assumes worker access to initialized global NLP models; use a Linux setup with compatible process-start behavior, since spawn-based execution can fail. The training collators also contain nested functions that are problematic with spawn workers.

MTEB's original `CASE=1` default selects `cuda:1`; the commands below explicitly override it to `cuda:0`. Exploratory clustering notebooks also refer to two GPUs. These settings do not establish a two-GPU requirement for the paper's training.

The manuscript reports 4,607 MB of compressed UberText input across the three selected domains. Additional disk space is required for extracted JSON, augmentation variants, CSV triplets, model caches, and checkpoints; no complete storage measurement is supplied. Deduplication holds sentence sets in RAM, pseudo-labeling materializes per-lemma embeddings, and evaluation repeatedly embeds definitions. MLM, back-translation, local Qwen generation, and the full MTEB suite can be expensive. No runtime estimate is validated by this README audit.

## Usage Instructions

The sequential data/training commands are given under [Reproducing the Paper](#reproducing-the-paper). Individual evaluators can run independently once their dataset, morphology assets, dependencies, and chosen model are available.

Evaluate the pretrained MPNet WSD baseline, skipping optional POS reports:

```bash
python -m eval.eval_wsd \
  --model-path sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \
  --sum-path datasets_pre_defined/sum_final.jsonlines \
  --device cuda:0 --no-reports
```

This reads the dictionary snapshot, downloads/loads the specified encoder and tokenizer, and prints accuracy over **retained dictionary-sense rows**. It also creates/appends `eval_wsd.log`. Omitting `--no-reports` runs POS/gloss reports and writes `badly_predicted.csv`; Stanza resources may be needed. No complete per-context prediction export is implemented. `--tokenizer-path` can select a separate tokenizer; by default it follows `--model-path`.

Evaluate sentence similarity for the two baselines listed in the STS script:

```bash
python -m eval.eval_stsb \
  --models sentence-transformers/paraphrase-multilingual-mpnet-base-v2 lang-uk/ukr-paraphrase-multilingual-mpnet-base \
  --device cuda:0
```

This loads `anikol12/STSB-UK`, evaluates each selected model, prints a table, and overwrites `sts_results.csv`. Pearson and Spearman cosine correlations are multiplied by 100 in the CSV.

Evaluate the installed MTEB registry's Ukrainian text tasks:

```bash
python -m eval.eval_mteb \
  --models sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \
  --device cuda:0
```

This downloads the selected tasks' inputs and produces per-task JSON scores, predictions, and caches described under Expected Outputs. This is not guaranteed to select the paper's 11-task snapshot. The script uses `NUM_PROC=8`. Supplying `--models` avoids its hard-coded historical checkpoint list.

Run the qualitative demo after preparing the WSD/morphology resources:

```bash
python demo.py
```

`demo.main()` selects `victormuryn/mpnet-use-markov-pt`, target lemma `коса`, two example sentences, and `cuda:0`. Edit those constants to inspect other examples or use CPU. Output is printed definitions and cosine similarities; no result file is saved. The demo's model identifier is evidence of an available integration, not a manifest identifying which of the paper's nine runs it represents.

## Methodology

### 1. Corpus extraction

`CollectUberTextSentences._process_ubertext_line` removes line breaks, replaces non-breaking spaces, and collapses whitespace. It requires **8–15 whitespace-separated tokens after ASCII punctuation removal**, fewer than four `*` characters, fewer than five em dashes, fewer than ten digits, and `langdetect.detect(...) == 'uk'`. Exact predicted-lemma intersections identify candidates. The collector uses half the CPU count, multiprocessing chunk size **128**, and flushes after **50,000 retained sentences**. `--num_examples -1` is converted to infinity by `main` and scans the source to completion.

Extraction is run separately with UDPipe and spaCy. The merger takes the union by lemma and exact sentence text, using sets. It does not perform near-duplicate or benchmark-overlap filtering.

### 2. Morphological matching and dictionary preparation

`read_and_transform_data` filters raw dictionary records to lemma length **greater than 3**, groups homonymous entries, removes stress marks, excludes missing definitions/examples, uses the first gloss of each synset by default, filters glosses occurring more than **4** times, removes configured function words and reference definitions, normalizes wording, and deduplicates using spaCy. Definitions/examples are then grouped by dictionary entry, and only lemmas with at least two retained meanings survive. See the function and `services/config.py` for the full exclusion list.

For triplet indices and WSD inference, `_find_target_word_in_sentence` uses a more permissive matcher than extraction: substring containment or character-trigram Dice similarity of at least **0.5** can count as a lemma match. It returns the first matching surface token. `_find_target_word_in_tokenized_text` reconstructs candidate words from decoded subwords; it reads tokenizer word IDs but does not use them to enforce boundaries. Triplet generation takes the first recovered occurrence, whereas inference rejects zero or multiple recovered occurrences. These choices can affect coverage and correctness.

### 3–5. Embeddings, pseudo-labels, and confidence filtering

`assign_meaning_to_sentence.process_lemma` uses `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` through `SentenceTransformer.encode`, full-sentence embeddings, and normalized outputs. Each meaning's representation is the arithmetic mean of its separately encoded definition strings. Cosine similarities are computed against all meanings of the same lemma, then softmax is applied across meanings:

```text
p(meaning | sentence, lemma) = softmax(cosine_similarity / 0.05)
```

The highest-probability meaning is retained only when **probability ≥ 0.9 AND cosine similarity ≥ 0.6**. Exact equality between the sentence and the meaning's first gloss is excluded. Batch size is **2048**. This confidence filtering is inside the assignment script; no artifact of all rejected assignments is saved. Dictionary examples remain metadata and are not encoded to produce these pseudo-labels.

The subsequent `delete_similar_sentences.py` stage embeds those dictionary examples and retained natural contexts with the same MPNet identifier and normalized SentenceTransformer embeddings. It removes a context if its maximum cosine similarity to **any example across all lemmas** is **≥0.95**. Encoding batch size is **512**; similarity chunks contain **2,048** contexts. It preserves definitions, examples, and meanings whose natural context list becomes empty. It reads examples from the input JSON metadata, so isolation against the official Hugging Face snapshot still requires verification that those examples match that snapshot. This filter runs before generation/transformations and does not audit the later generated or transformed outputs.

### 6. Generation-based augmentation

For meanings with fewer than **5** retained sentences, the generation prompt requests `max(0, 5 - current_count)` new examples from Qwen. It contains the lemma, definitions, and requested count. Although the script constructs an `existing_sentences` variable from dictionary examples, the prompt template has no corresponding placeholder, so those examples are not sent.

Only response lines beginning with `-` are parsed as generated sentences. They inherit the prompted meaning and are not pseudo-labeled again. Temperature, top-p, maximum response length, and seed are not set by the client; server defaults apply. The script does not guarantee the requested count, enforce uniqueness/target presence, retry malformed output, or skip previously generated meanings on a rerun. Output is appended; use a fresh destination for an independent run.

### 7. Transformation-based augmentation

Transformations are applied to the merged natural/generated pool, with separate scripts for contextual sentences and dictionary definitions. `ThreadedWriter` removes duplicate, empty, unchanged, and `<unk>`-containing variants. Sentence variants must retain a morphologically matching target; definitions deliberately skip that target check. Presence does not establish preservation of the original sense.

| Transformation | Implemented settings |
| --- | --- |
| Dropout | Independent deletion probability **0.15** over UDPipe tokens; **4** requested variants. Punctuation is not explicitly exempted. |
| MLM replacement | `Goader/modern-liberta-large`; mask non-punctuation words with probability **0.15**. `int(sqrt(n))` maskings, repeated that many times; **n=4** gives four candidates. Replacements are sampled with `torch.multinomial` after softmax over returned fill-mask scores. Default candidate count comes from the Transformers pipeline, not an explicit project constant. |
| Shuffling | Shuffle words inside punctuation-bounded segments in overlapping windows of **3**, advancing by **ceil(3/2)=2** words; **4** variants. This is not implemented as arbitrary swaps within ±3 positions. |
| Back-translation | Ukrainian → English → Ukrainian through the two OPUS models. **4** requested final variants, two hypotheses per stage; temperature **0.8**, top-k **50**, top-p **0.95**, maximum decoding length **512**, beam size defaults to **1** in the wrapper. CTranslate2 float16. |
| Stochastic combination | One augmenter selected uniformly per batch at each step; at least one step. `MARKOV_P=0.75` is the **stopping** probability after that minimum in `markov_process`. First step requests **9** variants; the **second and later** steps request **1** per input. The current origin-mapping code overwrites some branches; see issues. |

Standalone augmentation DataLoaders use batch size **256** and **2** workers. MLM inference batches are **128** in the standalone scripts. Stochastic orchestrators use DataLoader batch size **128**, **2** workers, and MLM inference batch size **1024**. All candidate counts are before output filtering and may decrease.

### 8. Definition-anchored triplets

The active `form_triplets.py` samples a contextual anchor, a positive definition from its assigned meaning, and a negative definition from a different meaning **of the same lemma**. Augmented definitions are added to the appropriate pools. There is no hard-negative mining or in-batch negative objective in this default path.

`MAX_SENTENCES_PER_MEANING=100` caps sampled source sentences and distributes a **target of 100 candidate triplet draws** across each nonempty meaning. Commit `2b1f205` changed this constant from 300 to 100. The script tries unused combinations within each source sentence, then permits repeats once combinations are exhausted. The computed unique-combination cap is not used by the final loop. Failed target-word matches are skipped, so output can be below 100 rows per meaning and is not guaranteed unique. Meanings with zero contexts produce no rows. The numeric cap now matches the manuscript, but the manuscript's uniqueness claim remains unresolved.

Target indices are computed with the MPNet tokenizer before training. Only anchors have target indices in the default CSV; definitions use whole-input mean pooling. Changing the tokenizer requires rebuilding the indices.

### 9. Training

The trainer loads a Hugging Face `AutoModel` and shares it across all three triplet inputs. It trains the encoder directly, then exports a SentenceTransformer with mean pooling. This distinction matters for token-level anchor pooling during training versus full-sentence use after export.

| Parameter | Current configured value / implementation |
| --- | --- |
| Backbone/tokenizer | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` |
| Objective | `triplet_loss_cosine`; `TripletMarginWithDistanceLoss` with cosine distance |
| Margin | **0.2** for cosine triplet loss; ordinary `triplet_loss` uses **1.0** |
| Pooling | `pool_targets=True`, `use_both_poolings=False`; target anchor and full-definition means |
| Sequence length | **128**, fixed padding and truncation |
| Batch size / epochs | **104 / 2** |
| Optimizer | AdamW; learning rate **2e-6**, weight decay **0.01** |
| Gradient clipping | Maximum norm **1.0** |
| Schedule | Linear warm-up/decay; **0.1** warm-up ratio; `apply_warmup=True` |
| Train/validation split | Selected HF `train` split or local CSV shuffled with pandas `random_state=42`; **99% / 1% by triplet row**, independently of training seed |
| Validation / early stopping | Before each epoch and every **200** minibatches after batch index 0; patience **10** in INI, **15** in dataclass fallback |
| Trainable layers | `layers_to_unfreeze=0` means all layers trainable |
| AMP | FP16 plus GradScaler on CUDA; disabled on CPU |
| Data loading | **4** workers, prefetch factor **2**, persistent workers, pinned memory; both train and validation shuffled |
| Random initialization | Pretrained weights used; reinitialization settings are not wired into the trainer |

The CLI now accepts `--hf-dataset`, `--hf-subset`, `--train-data`, `--seed` (default 42), `--pool-targets`, `--run-name`, and `--batch-size`, alongside config/device selection. Providing both HF arguments selects the prepared dataset and bypasses the configured CSV. The same training seed is applied to Python, NumPy, and PyTorch before trainer initialization.

The original INI points to a dropout CSV; `reviewer_config.ini` disables W&B but still points to the former `..._300.csv` output. Neither path automatically follows the current builder. There is also a local-input regression: `TrainingConfig` lacks `hf_dataset`/`hf_subset` defaults, yet `_setup_data` accesses them directly. The documented HF command supplies both attributes through the CLI; local CSV execution requires that config defect to be fixed and the input path to be updated. The reviewer INI remains a **current-code example, not a recovered paper-run configuration**.

Validation drives checkpoint selection using loss, not WSD accuracy. `_best` is saved only at an improving within-epoch check with `batch_count > 0`; early-stopped and final exports also exist. Ordinary per-epoch saving is currently commented out. Final WSD evaluation loads `_final`, not `_best`. MNR and NT-Xent branches are experimental and have known defects; they are not required for the manuscript's cosine-triplet experiments.

### 10. WSD inference/evaluation

`WordSenseDetector` groups candidate gloss lists by lemma and predicts once for each processed **dictionary-meaning row**, whose `examples` field can contain multiple sentences. The evaluator's `max_sim_across_all_examples` strategy chooses the candidate meaning with the greatest cosine similarity across its individual definitions and all usable target-word example embeddings. This is a maximum, not the averaged meaning representation used during pseudo-labeling.

Inference uses mean-pooled target-word hidden states from the last layer and whole-definition mean embeddings. `prediction_accuracy` drops rows with null values, compares the first gold/predicted gloss, and computes accuracy over remaining rows. It does **not** produce one independent prediction for each of the manuscript's 15,961 contexts. Report the evaluation unit and dropped coverage explicitly; the current metric cannot be assumed to reproduce a context-level WSD table. Training `pool_targets=False` does not switch this WSD evaluator to full-sentence inference.

### 11–12. STS-B and MTEB

STS-B uses `EmbeddingSimilarityEvaluator` on full-sentence embeddings and the dataset's `train` split as evaluation data. Exactly identical strings receive target score 1.0; other values are not rescaled by the script. Pearson and Spearman cosine correlations are exported as percentages.

MTEB loads SentenceTransformer models, dynamically selects Ukrainian text tasks, invokes `mteb.evaluate`, caches results, saves task predictions, and serializes each task's `scores`. It does not freeze the paper's task suite, map checkpoints to configurations, or aggregate multiple runs into paper tables.

## Reproducing the Paper

**Read this as a dependency-ordered execution guide for the current code.** Steps with unavailable external assets or missing paper procedures are explicitly marked. Completing the executable stages alone does not resolve the manuscript discrepancies. Record every run's source revision, environment, input hashes, configuration, seeds, and output identity before claiming paper reproduction.

There are two entry points: rebuild natural/generated/augmented data through Steps 0–6, or load the prepared Hugging Face triplets in Step 7. No script uploads Step 6's output to Hugging Face or establishes that it reproduces all published subsets. Both training routes still require the trainer's morphology assets, and final WSD evaluation requires the raw dictionary input.

### Step 0 — Prepare the benchmark, legacy raw input, morphology assets, and corpus

The official expanded benchmark is already hosted on Hugging Face: use the revision-pinned loading example under Dataset Information. Do not move it to another archive or upload the large raw dictionary to GitHub. **Integration gap:** the existing pipeline/evaluator still calls the nested raw-dictionary loader; directly consuming the processed Hugging Face rows requires an explicit loader path and validation that preserves their meanings/examples. No implicit reconstruction or extra filtering is applied here.

To execute the existing legacy pipeline commands below, obtain the author-approved raw dictionary input and place it at the configured location. If using the locally supplied root-level file described above, the following copies it without overwriting an existing destination:

```bash
cp -n sum_final.jsonlines datasets_pre_defined/sum_final.jsonlines
```

This command requires that local file; it is not a download method. Obtain `models/20180506.uk.mova-institute.udpipe` separately and install the spaCy model as described above. **The legacy raw-data execution path remains blocked for a fresh clone until the raw input and UDPipe weights are available; the official processed benchmark itself is publicly available.**

Derive the lemma list through the same preprocessing function used for pseudo-labeling/evaluation:

```bash
python - <<'PY'
from pathlib import Path
from services.config import SUM_PATH, PATH_TO_LEMMAS_OF_INTEREST
from services.utils_data import read_and_transform_data
frame = read_and_transform_data(SUM_PATH, homonym=True)
lemmas = sorted(frame['lemma'].unique())
Path(PATH_TO_LEMMAS_OF_INTEREST).write_text('\n'.join(lemmas) + '\n', encoding='utf-8')
print('lemmas:', len(lemmas), 'meanings:', len(frame),
      'examples:', frame['examples'].map(len).sum())
PY
```

This convenience command invokes legacy raw preprocessing; it does not recover the upstream construction process or establish equivalence to the official snapshot. Compare its rows/counts against the pinned Hugging Face data, including the verified 1,464-lemma count, rather than assuming they match. The output lemma file is overwritten.

Download the sentence-split source files using URLs already documented by this project (availability and checksums have not been revalidated in this audit):

```bash
curl -fL https://lang.org.ua/static/downloads/ubertext2.0/news/sentenced/ubertext.news.filter_rus_gcld+short.text_only.txt.bz2 -o datasets_pre_defined/ubertext.news.filter_rus_gcld+short.text_only.txt.bz2
curl -fL https://lang.org.ua/static/downloads/ubertext2.0/wikipedia/sentenced/ubertext.wikipedia.filter_rus_gcld+short.text_only.txt.bz2 -o datasets_pre_defined/ubertext.wikipedia.filter_rus_gcld+short.text_only.txt.bz2
curl -fL https://lang.org.ua/static/downloads/ubertext2.0/fiction/sentenced/ubertext.fiction.filter_rus_gcld+short.text_only.txt.bz2 -o datasets_pre_defined/ubertext.fiction.filter_rus_gcld+short.text_only.txt.bz2
```

### Step 1 — Extract corpus contexts and merge both analyzers

With all Step 0 inputs present, run the six source/analyzer combinations:

```bash
for corpus in news wikipedia fiction; do
  for analyzer in udpipe spacy; do
    python -m collect_sentences.collect_ubertext_sentences \
      --source_dataset "datasets_pre_defined/ubertext.${corpus}.filter_rus_gcld+short.text_only.txt.bz2" \
      --save_dataset "local_datasets/raw_sentences/lemma_examples_samples_${analyzer}_${corpus}.json" \
      --lemmas_file datasets_pre_defined/unique_lemmas_homonyms.txt \
      --tokenizer "$analyzer" --num_examples -1 || exit 1
  done
done
python -m local_datasets.raw_sentences.process_raw_sentences
```

The collector appends JSONL batches to the six chosen `.json` destinations. Start with fresh outputs for a new extraction. The merger reads **every `.json` file** in `local_datasets/raw_sentences/` and overwrites `unique_lemma_sentences.jsonl`; keep unrelated experiments out of that directory. Multiprocessing ordering and unseeded language detection can vary. No dataset-construction seed is exposed here.

### Step 2 — Assign pseudo-labels

```bash
python -m local_datasets.semi_supervised_2.assign_meaning_to_sentence \
  --embedder_model sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \
  --batch_size 2048
```

Inputs: dictionary at `SUM_PATH` and deduplicated contexts. Outputs: `assigned_meanings_mpnet.jsonl` and `lemmas_with_meanings_and_sentences_mpnet.json`, both under `local_datasets/semi_supervised_2/`. Device is the script's `model_device` constant. Output names remain `mpnet` even if `--embedder_model` changes; use separate destinations when comparing teachers. No explicit seed is set.

### Step 3 — Confidence filtering and benchmark isolation

The **0.9 probability / 0.6 cosine** confidence filter already ran in Step 2. Now apply the separate **0.95 cosine** benchmark-overlap filter:

```bash
python -m local_datasets.semi_supervised_2.delete_similar_sentences
```

Input: `local_datasets/semi_supervised_2/lemmas_with_meanings_and_sentences_mpnet.json`. Output: `local_datasets/semi_supervised_2/lemmas_with_meanings_and_sentences_mpnet_filtered.json`, overwritten on each successful run. The filter uses MPNet and all dictionary `meaning.examples` stored in the input. It prints total/retained/removed counts; it does not save a complete excluded-record manifest or set an explicit RNG seed. Model, threshold, batch sizes, and input/output paths are constants in the script.

For optional inspection without writing the output, use:

```bash
python -m local_datasets.semi_supervised_2.delete_similar_sentences --dry-run --show 20
```

The dry run still performs full embedding/similarity computation and prints the first matching exclusions; it is not a cheap smoke test. Run the normal command to create the file required by Step 4. The implementation now exists, but historical filter outputs, input identity against the official benchmark, and proof that this procedure produced the published training datasets still need to be archived.

### Step 4 — Generate examples and merge

Prerequisites: resolved Step 3, a local server serving the configured Qwen identifier, and fresh generation output. Review `INPUT_FILE`, `OUTPUT_FILE`, `MIN_SENTENCES`, `BASE_URL`, `API_KEY`, and `MODEL_NAME` in the generation script.

```bash
python -m collect_sentences.generate_sentences_4_absent_meanings
python -m local_datasets.semi_supervised_2.merge_collected_and_generated
```

Generation and merging both read `lemmas_with_meanings_and_sentences_mpnet_filtered.json` from Step 3. Generation checks the remaining natural counts and appends `generated_sentences.jsonl`; merge reads those generated records plus the filtered natural pool and overwrites `merged_collected_and_generated_mpnet.json`, all under `local_datasets/semi_supervised_2/`. The server's sampling state is not recorded by these commands. Merge warns about missing meanings or fewer than five examples, but does not remedy them. For the Natural configuration, skip generation/merge and point the triplet builder directly at the **filtered** natural pool as described below. Existing generated, merged, transformed, and triplet files are not retroactively corrected by the path fix; retain their provenance and rebuild downstream artifacts when using newly filtered inputs.

### Step 5 — Transform sentences and definitions

For the individual transformation pools used by the active default triplet builder:

```bash
python -m augment.dropout.dropout
python -m augment.dropout.dropout_definitions
python -m augment.mask.mask
python -m augment.mask.mask_definitions
python -m augment.token_shuffling.token_shuffling
python -m augment.token_shuffling.token_shuffling_definitions
python -m augment.translation.augment_translation
python -m augment.translation.augment_translation_definitions
```

Each reads `merged_collected_and_generated_mpnet.json` and overwrites its own JSONL output under `local_datasets/augmented/`. Translation remains blocked until converted OPUS weights are supplied. MLM requires its downloaded model. For a single-method experiment, run only its sentence/definition pair and select that pair in the triplet builder. Seeds are not consistently exposed; see Randomness and Seeds.

For the separate stochastic experiment:

```bash
python -m augment.augment_all_together
python -m augment.augment_all_together_definitions
```

These load all four augmenters, including local OPUS weights, and write the `all_together/` outputs plus selection logs. They do not simply concatenate the four individual pools, and the active default triplet builder does not consume their outputs. Resolve the stochastic-method discrepancies before using them to reproduce that paper configuration.

### Step 6 — Construct triplets

Select `DATASET_PATH`, `OUTPUT_CSV`, `USE_AUGMENTED`, `USE_DEFINITIONS_AUGMENTED`, and the two augmentation-path tuples in the source as described in the configuration table below. There is no experiment-name or seed CLI.

```bash
python -m local_datasets.semi_supervised_2.form_triplets
```

Current defaults read the merged pool and all four individual transformation pairs. Output: `local_datasets/semi_supervised_2/triplets_semi_supervised_all_augs_mixed_100.csv`. The script sets Python's seed to 42, loads both spaCy and UDPipe, and computes MPNet target indices. It attempts 100 draws per nonempty meaning, with skipped target matches and possible repeated triplets. The numeric change alone does not establish the manuscript's 100-unique-triplet procedure or recreate the published HF configurations.

### Step 7 — Train from a prepared Hugging Face dataset

After installing dependencies and preparing the morphology assets and raw dictionary for final evaluation, run one All combined, dataset-seed-42, training-seed-42, target-pooling example:

```bash
python -m services.trainer.trainer \
  --config services/trainer/reviewer_config.ini \
  --device cuda:0 \
  --hf-dataset victormuryn/wsd-training-dataset \
  --hf-subset all_augs_seed42 \
  --seed 42 \
  --pool-targets true \
  --run-name all_augs_seed42_trainseed42_pooltrue
```

Inputs: the selected HF configuration's `train` split, pretrained encoder/tokenizer, and morphology assets. The trainer creates its 99/1 split internally with shuffle seed 42. Outputs: optional best/early-stopped and final model directories under `models/fine-tuned-models/`, followed by WSD evaluation of the final export. The reviewer INI disables W&B; its old CSV path is bypassed by the two HF arguments. With W&B disabled, checkpoint IDs are timestamps and `--run-name` does not replace them; record the command and checkpoint path together. With W&B enabled, the name labels the W&B run and checkpoint IDs use the W&B run ID. Final WSD evaluation also requires the dictionary snapshot and POS-report resources.

Select another HF subset, pooling setting, or seed using the CLI. `--batch-size` overrides the INI batch size. For freshly constructed local CSVs, first add `hf_dataset` and `hf_subset` defaults to `TrainingConfig`, then select the new `_100.csv` using `--train-data` or an updated INI. The old local-only README command cannot currently complete `_setup_data`. Missing config files also silently fall back to defaults; verify the chosen INI exists.

### Steps 8–10 — Evaluate WSD, STS-B, and MTEB

Set a shell variable to an **actual exported model directory**. This command prompts for the path so no nonexistent run identifier is presented as a runnable checkpoint:

```bash
printf 'Path to the exported model directory: '
read -r MODEL_PATH
python -m eval.eval_wsd --model-path "$MODEL_PATH" --device cuda:0 --no-reports
python -m eval.eval_stsb --models "$MODEL_PATH" --device cuda:0
python -m eval.eval_mteb --models "$MODEL_PATH" --device cuda:0
```

WSD inputs include the dictionary and morphology resources. STS-B/MTEB load their external datasets. Outputs follow the Usage and Expected Outputs sections. Explicitly record whether the directory is `_best`, `_early_stopped`, or `_final`: the trainer itself evaluates `_final`, and no manifest maps historical checkpoints to the paper's result rows. Each STS invocation overwrites the same CSV, so preserve it before the next invocation or pass multiple models together. MTEB uses model-path-based caches; reusing a path with changed weights can reuse stale results.

There is no executable Step 11 for the paper's zero-shot LLM comparison or a final command that regenerates all manuscript tables.

## Reproducing Experimental Configurations

For training on prepared triplets, `train_all.sh` defines the following HF configuration prefixes. The mapping reflects repository naming and augmentation pools; the original artifact-to-result manifest is still required to verify each published result.

| Paper configuration | HF configuration prefix | Example subset for dataset seed 42 |
| --- | --- | --- |
| Natural | `raw` | `raw_seed42` |
| Generation | `generated` | `generated_seed42` |
| Generation + MLM | `mask` | `mask_seed42` |
| Generation + Dropout | `dropout` | `dropout_seed42` |
| Generation + Back-translation | `translation` | `translation_seed42` |
| Generation + Shuffling | `token_shuffling` | `token_shuffling_seed42` |
| Stochastic combination | `markov` | `markov_seed42` |
| All combined | `all_augs` | `all_augs_seed42` |

Use the same prefixes with `_seed123` and `_seed456` for the other dataset construction seeds. The dataset is `victormuryn/wsd-training-dataset`; `combined` is not the prefix used by the current launcher.

For rebuilding the triplets locally, selection still requires editing constants in `local_datasets/semi_supervised_2/form_triplets.py` and preserving that source/config with each output:

| Paper configuration | `DATASET_PATH` choice | Augmentation selection in the triplet builder | Reproduction limit |
| --- | --- | --- | --- |
| Natural | `local_datasets/semi_supervised_2/lemmas_with_meanings_and_sentences_mpnet_filtered.json` | Set both `USE_AUGMENTED=False` and `USE_DEFINITIONS_AUGMENTED=False` | Requires Step 3; unique sampling and original artifact provenance remain unresolved |
| Generation | `local_datasets/semi_supervised_2/merged_collected_and_generated_mpnet.json` | Both flags false | Requires original generation settings/artifacts |
| Generation + MLM | Same merged pool | Both flags true; retain only the `mask/` entry in each path tuple | Original augmentation seed settings unavailable |
| Generation + Dropout | Same merged pool | Both flags true; retain only the `dropout/` entry in each tuple | Original INI names a dropout CSV, but no matching saved builder configuration is supplied |
| Generation + Back-translation | Same merged pool | Both flags true; retain only the `translation/` entry in each tuple | Converted weights/decoding provenance unavailable |
| Generation + Shuffling | Same merged pool | Both flags true; retain only the `token_shuffling/` entry in each tuple | Paper's verbal window description differs from implementation |
| Stochastic combination | Same merged pool | Both flags true; select only the currently commented `all_together/` entry in each tuple | Stop/continue probability and branch retention require resolution |
| All combined | Same merged pool | Both flags true; four individual methods, the active defaults | This mapping is supported by the code's pool union, but no original per-run manifest confirms the precise published composition |

Change `OUTPUT_CSV` to a distinct destination for every locally rebuilt dataset/configuration. The default filename does not change automatically when flags change, and the builder still hard-codes seed 42. The launcher selects already constructed seed-specific HF datasets; it does not generate these CSV variants. Local CSV training has the config-default blocker described in Step 7.

For **full-sentence anchor pooling**, set:

```ini
pool_targets = False
use_both_poolings = False
```

For **target-token anchor pooling**, set:

```ini
pool_targets = True
use_both_poolings = False
```

Both settings retain full-definition mean pooling in the active triplet path. `use_both_poolings=True` averages full-input and target-anchor objectives; it is not one of these two paper settings.

The equivalent CLI overrides are `--pool-targets false` for full-sentence anchors and `--pool-targets true` for target-token anchors, with `use_both_poolings=False` in the INI.

The launcher implements **dataset seeds {42, 123, 456} × training seeds {42, 123, 456} = nine runs per configuration/pooling setting**. Across eight configurations and two poolings, this is **144 unique jobs using 24 HF subsets**. The dataset seed chooses the subset; `--seed` controls Python, NumPy, and PyTorch within training. The pandas split seed remains 42. This matches the manuscript's run-count design, while the exact construction process and historical result/checkpoint mapping still need provenance.

To launch this entire matrix on one GPU with W&B disabled:

```bash
bash train_all.sh --config services/trainer/reviewer_config.ini --num-gpus 1
```

This schedules actual training; there is no dry-run flag. The script also accepts `--batch-size N` and defaults to two GPUs if `--num-gpus` is omitted. Without `--config`, it uses `fine_tuning_config.ini`, which enables the author-specific W&B entity. Run names follow `<prefix>_seed<dataset_seed>_trainseed<training_seed>_pool<true|false>`.

Despite its `MISSING_JOBS` variable and completion message, the launcher always queues its full hard-coded list; it does not detect finished experiments or resume training. Errors are printed and processing continues, and trainer exceptions can be swallowed before reaching the shell. The final “complete” message and shell exit status do not establish 144 successful runs. Check each run's logs/evaluation and archive a success/failure manifest; no automatic mean/SD or manuscript-table aggregation follows this command. Shared `eval_wsd.log` and `badly_predicted.csv` outputs are not separated by run and can collide with multiple workers.

## Reproducing Paper Tables and Results

Paper result labels below refer to groups identifiable in the supplied manuscript sources. The numeric manuscript tables are not treated as executable result artifacts.

| Paper result | Implemented script/notebook | Input | Output / missing link |
| --- | --- | --- | --- |
| Dictionary/corpus coverage | `local_datasets/sum_and_ubertext_eda.ipynb` | Dictionary snapshot; deduplicated contexts | Interactive counts, descriptive statistics, plots; no consolidated table exporter |
| Meaning coverage after confidence filtering | `local_datasets/semi_supervised_2/eda_lemmas_with_meanings.ipynb` | Meaning-organized natural pool | Interactive counts/distributions, including uncovered meanings/lemmas; no publication-table generator |
| Pseudo-label human QA | Assignment script supplies candidate records only | Retained assignments plus human labels | The manuscript's 400-record sample, annotations, sampling seed, and agreement/CI computation are not provided |
| WSD baselines | `python -m eval.eval_wsd` with explicit model arguments as above | Chosen model, dictionary, morphology assets | Printed dictionary-sense-row accuracy, optional error CSV; full paper baseline roster and context-level protocol not reproduced |
| Augmentation/pooling comparison | `bash train_all.sh --config services/trainer/reviewer_config.ini --num-gpus 1` → trainer's final WSD evaluation | 24 HF subsets, INI, pretrained model and morphology/raw dictionary assets | 144 scheduled runs across nine seed combinations per configuration/pooling; checkpoints and WSD return values. No historical result manifest, mean/SD aggregation, or table script; success must be checked per run. |
| Sense-availability analysis | No dedicated implementation found | Would require natural counts joined to gold meanings and aligned predictions | Cannot regenerate the paper's availability-group accuracy/gain tables from a supplied command |
| STS-B results | `python -m eval.eval_stsb` with explicit models | `anikol12/STSB-UK`, selected models | `sts_results.csv`; no configuration grouping or mean/SD across nine runs |
| Ukrainian MTEB results | `python -m eval.eval_mteb` with explicit models | Installed registry's tasks, selected models | Per-task score JSON; original task/revision manifest and paper aggregation missing |
| Zero-shot LLM comparison | No evaluation script found | Manuscript/supplement contain prompts and reported results | Generation script serves another purpose; request exact evaluator, parsing rules, server settings, and predictions |

The stored notebooks are exploratory evidence, not a frozen full-paper workflow. Their clustering parameters (including temperature 0.2 and thresholds 0.7) must not replace the active pseudo-label settings of 0.05/0.9/0.6.

## Randomness and Seeds

| Source of randomness | Current implementation |
| --- | --- |
| Corpus collection | No explicit language-detector seed; unordered multiprocessing results; analyzer/model versions can affect retention |
| Deduplication / ordering | Sets converted to lists in raw merging and augmentation/triplet pools; no fixed `PYTHONHASHSEED` or stable ordering |
| Pseudo-label assignment | No explicit RNG seed; fixed thresholds, but model/device/library differences remain |
| Benchmark-overlap filter | No explicit RNG seed; fixed threshold 0.95 and first-match dry-run samples; input identity and model/device/library versions determine retained contexts |
| Generation | No seed, temperature, or top-p in the API request; server defaults and model/quantization determine sampling |
| Standalone dropout/shuffling | Python `random` used without an entry-point seed |
| MLM | `Masker(seed=42)` resets the global Python RNG; replacement sampling uses PyTorch without an explicit seed in that stage |
| Translation | Sampling parameters provided, but no CTranslate2 seed configured |
| Stochastic orchestrators | `random.seed(42)`; `Masker` also resets that global seed; batch-level choices share the RNG with augmentation operations |
| Triplet builder | `random.seed(42)` in `__main__`; context/definition sampling affected by input and set ordering |
| Dataset selection in launcher | Subsets use dataset seed labels **42, 123, 456**; the launcher loads prepared datasets and does not seed or rerun their construction |
| Trainer CLI | `torch.manual_seed(args.seed)`, `random.seed(args.seed)`, `np.random.seed(args.seed)` before initialization; default **42**, launcher values **42, 123, 456**. Earlier library-specific values 47/92/39 were replaced in `2b1f205`. Programmatic `Trainer(...)` use does not execute the CLI seeding block. |
| Split | pandas shuffle uses independent `random_state=42`, then positional 99/1 split |
| DataLoader/model | Training dropout and shuffled train/validation loaders use RNG state; no explicit loader generator or worker seed function |
| CUDA determinism | No separate CUDA seed call, deterministic-algorithm setting, or cuDNN determinism configuration is specified; fixed seeds alone do not establish exact reproducibility |
| Exploratory notebooks | PCA/TSNE references use `random_state=42`; KMeans uses `random_state=0`; not the paper's training seed grid |

Archived dataset rewrites also use Python seed 42, but are not the active scientific workflow. The current three-by-three grid is explicit in `train_all.sh`; regenerating its dataset seeds still requires control of all construction-stage RNGs and ordering. Fixed training seeds do not guarantee deterministic GPU runs. No expensive experiment was rerun to establish historical equivalence. The HF dataset/subset attributes and training seed are not dataclass fields, so the current W&B config export omits them; the launcher includes subset/seed labels in the run name, but a separate manifest must record dataset revision, command, config, and checkpoint identity.

## Expected Outputs

All destinations are generated at runtime and are generally Git-ignored. See Dataset Information for record schemas.

| Stage | Expected destination / content |
| --- | --- |
| Extraction | Six `local_datasets/raw_sentences/lemma_examples_samples_<analyzer>_<corpus>.json` files from the documented loop; appended JSONL batches |
| Deduplication | `local_datasets/raw_sentences/unique_lemma_sentences.jsonl` |
| Pseudo-labeling | `local_datasets/semi_supervised_2/assigned_meanings_mpnet.jsonl` and `lemmas_with_meanings_and_sentences_mpnet.json` |
| Benchmark-overlap filtering | `local_datasets/semi_supervised_2/lemmas_with_meanings_and_sentences_mpnet_filtered.json`; terminal counts, with sample exclusions in dry-run mode |
| Generation/merge | `local_datasets/semi_supervised_2/generated_sentences.jsonl` and `merged_collected_and_generated_mpnet.json` |
| Dropout / MLM / shuffling | In `local_datasets/augmented/dropout/`, `mask/`, or `token_shuffling/`: `augmented_sentences.jsonl` and `augmented_sentences_definitions.jsonl` |
| Back-translation | `local_datasets/augmented/translation/augmented_sentences_translated_v3.jsonl` and `augmented_sentences_translated_definitions.jsonl` |
| Stochastic combination | `local_datasets/augmented/all_together/augmented_sentences_3.jsonl` and `augmented_sentences_definitions_3.jsonl`; root-level `selected_augmenters_log.json` and `selected_augmenters_definitions_log.json` record batch-level choices |
| Default triplets | `local_datasets/semi_supervised_2/triplets_semi_supervised_all_augs_mixed_100.csv` |
| HF training inputs | Selected `victormuryn/wsd-training-dataset` subset in the library-managed cache; no local CSV export is performed |
| Checkpoints | Under configured save directory: `model_<run_id>_best` when eligible, `model_<run_id>_<epoch>_early_stopped` when triggered, and `model_<run_id>_final`. Ordinary `model_<run_id>_<epoch>` saving is commented out. |
| Matrix launcher | Per-job prefixed terminal output and trainer artifacts; no durable queue, success/failure summary file, or result aggregation. Temporary counter/lock files are cleaned up at launcher exit. |
| Training metrics | Console progress; W&B only when enabled. No complete local loss/metric CSV or resumable optimizer/scheduler/RNG checkpoint is implemented. |
| WSD | `eval_wsd.log`; optional root `badly_predicted.csv`; POS helper may create `data/pos_precalculation.pkl`. No complete prediction JSONL. Accuracy is printed by the CLI and returned by `evaluate_wsd`. |
| STS-B | Root `sts_results.csv`, with model index and `pearson_cosine`, `spearman_cosine` columns |
| MTEB | `cache/mteb_<model_id_with_slashes_replaced>/`; `eval/mteb_prediction/<same_id>/`; `eval/mteb_results/<same_id>/<task_name>_results.json` |
| Environment report | Console JSON, or `logs/environment-report.json` and `logs/requirements-resolved.txt` using the optional commands above |

Extraction and generation append; merging, triplet construction, augmentation writers, and STS summaries overwrite. Translation's existing resume logic is not reliable: it looks for `original`, while the writer saves `sentence`, and the writer opens in overwrite mode. Preserve outputs and use distinct experiment destinations. A checkpoint directory existing is not proof of a successful complete training run, because save/training exceptions are currently caught broadly.

Optional frequency reports in `services/utils_results.py` additionally require `data/frequents.pkl`. `prepare_frequent_dictionary` can create it from a compatible compressed frequency source, but that source's acquisition path is not documented. Frequency reports default to disabled and are not the missing sense-availability analysis.

## Pretrained Models and External AI Models

No immutable pretrained-model commit is pinned in the active model loaders. The official WSD data revision is separately pinned in the loading example above. Model identifiers below are taken from source; model availability, access conditions, and licenses need to be archived with the release.

| Identifier/resource | Provider and role | How loaded |
| --- | --- | --- |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | Sentence Transformers; pseudo-label teacher, training initialization/tokenizer, baseline evaluation | Automatically through Hugging Face libraries if not cached |
| `lang-uk/ukr-paraphrase-multilingual-mpnet-base` | lang-uk; STS-B/MTEB reference model | Automatically via SentenceTransformer |
| `victormuryn/mpnet-use-markov-pt` | Project-author model used by `demo.py` | Automatically via AutoModel/AutoTokenizer; no mapping to nine-run results supplied |
| `Goader/modern-liberta-large` | Goader; Ukrainian MLM replacements | Automatically through Transformers fill-mask pipeline; FP16 requested |
| `Helsinki-NLP/opus-mt-tc-big-zle-en` / `Helsinki-NLP/opus-mt-tc-big-en-zle` | Helsinki-NLP OPUS; Ukrainian–English–Ukrainian translation | Marian tokenizers download; weights must be separately supplied as the two local CTranslate2 directories |
| `Qwen/Qwen3-VL-8B-Instruct` | Qwen; generation for meanings with fewer than five contexts | Request to externally provisioned local server; no local model loader/server launcher in this repository |
| `intfloat/multilingual-e5-large-instruct` / `intfloat/multilingual-e5-large` | intfloat; MTEB baseline list; instruct variant also appears as a commented training alternative | Automatically via SentenceTransformer when selected. No explicit task-prompt policy is specified in project code. |
| `facebook/nllb-200-distilled-600M` | Meta; alternative CTranslate2 translator tokenizer | Automatically downloaded tokenizer; class is not used by active OPUS scripts |
| `facebook/nllb-200-distilled-600m` | Literal lowercase identifier in alternative Transformers translator | Tokenizer/model loader requests this spelling; equivalence/availability is not validated |
| `facebook/nllb-200-3.3B` | Commented alternative/conversion example | Not selected in active scripts; not a required paper dependency |
| `uk_core_news_sm` | spaCy Ukrainian pipeline | Separate spaCy download; exact model version absent |
| `20180506.uk.mova-institute.udpipe` | Ukrainian UDPipe model referenced throughout | Manual file; no checksum/download recipe supplied |

The [Ukrainian Sentence Embeddings (USE) collection](https://huggingface.co/collections/victormuryn/ukrainian-sentence-embeddings-use) is linked in the original project documentation. It does not replace a versioned checkpoint/configuration/seed manifest. Historical local checkpoint IDs in evaluation lists and archived mining scripts are not committed models. The manuscript lists additional embedding and zero-shot LLM baselines, but their complete evaluation configurations are not implemented here.

## Citations

If using this code, cite the manuscript and the relevant datasets, pretrained models, and software. Publication metadata remains provisional; no DOI is assigned here.

```bibtex
@article{muryn_laba_ukrainian_wsd,
  title   = {Improving Ukrainian word sense disambiguation with sense-aware sentence embeddings},
  author  = {Muryn, Victor and Laba, Yurii},
  journal = {PeerJ Computer Science},
  note    = {Manuscript under review}
}
```

The following references/links were already present in project documentation or the manuscript bibliography supplied for this audit:

- Laba et al. (2023), [Contextual Embeddings for Ukrainian: A Large Language Model Approach to Word Sense Disambiguation](https://aclanthology.org/2023.unlp-1.2/), DOI: 10.18653/v1/2023.unlp-1.2. This is prior benchmark work, not an identifier for the expanded snapshot.
- Chaplynskyi (2023), [Introducing UberText 2.0: A Corpus of Modern Ukrainian at Scale](https://aclanthology.org/2023.unlp-1.1/), DOI: 10.18653/v1/2023.unlp-1.1.
- Reimers and Gurevych (2019), [Sentence-BERT](https://aclanthology.org/D19-1410/), DOI: 10.18653/v1/D19-1410.
- Cer et al. (2017), [SemEval-2017 Task 1: Semantic Textual Similarity](https://aclanthology.org/S17-2001/). The Ukrainian translation also needs its own dataset attribution/version record.
- Muennighoff et al. (2023), [MTEB: Massive Text Embedding Benchmark](https://aclanthology.org/2023.eacl-main.148/), DOI: 10.18653/v1/2023.eacl-main.148. Cite the individual task datasets used in a frozen evaluation suite as well.
- Straka et al. (2016), [UDPipe](https://aclanthology.org/L16-1680/), and Honnibal et al. (2020), [spaCy](https://doi.org/10.5281/zenodo.1212303).
- Tiedemann and Thottingal (2020), [OPUS-MT](https://aclanthology.org/2020.eamt-1.61), for translation models.
- Haltiuk and Smywiński-Pohl (2025), [Ukrainian Modern LiBERT work](https://aclanthology.org/2025.unlp-1.14/), for the MLM resource.
- Bai et al. (2025), [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631), for generation.

## License

**No repository-level license file is currently present. TODO before archival/publication: the authors must select and add a code license.** This README does not grant a new license or silently assign one.

Third-party dictionary/corpus/benchmark data and pretrained weights have independent licenses. Project-generated examples and checkpoints may also be subject to source/model terms. Record those terms and redistribution permissions in the release manifest. The local UDPipe source archive's metadata reports MPL 2.0 for that package; that is not a license for this repository, the Ukrainian weights, or any dataset.

## Contribution Guidelines

Use a repository issue or pull request to report a reproducibility problem. Include the source revision, command, relevant config, environment report, input artifact identity/checksum, expected behavior, and traceback or observed output. Exclude credentials and restricted datasets.

For scientific changes, explain the effect on sampling, filtering, pooling, evaluation units, and reported results. Keep behavior changes separate from packaging/documentation changes, provide a small verifiable example, and preserve existing experimental code and artifact provenance. New experiment presets should record all seed values and immutable model/data revisions. Licensing and publication questions require the authors' decision.
