# Improving Ukrainian word sense disambiguation with sense-aware sentence embeddings

Research code accompanying the manuscript by Victor Muryn and Yurii Laba, submitted to **PeerJ Computer Science** as an **AI Application** article. The approach adapts multilingual embeddings by pairing Ukrainian sentences with dictionary definitions of the meanings expressed by their target words.

**Reproduction status:** the repository contains corpus extraction, pseudo-labeling, augmentation, triplet construction, training, and embedding-based evaluation code. It does **not yet provide an exact reproduction of all paper results**. The original environment, complete experiment/seed manifest, some data preparation steps, and result aggregation are missing; several implementation/manuscript discrepancies remain. Commands below describe the current implementation and state their prerequisites. See [Reproducibility issues found](#reproducibility-issues-found) before launching experiments.

Quick navigation: [datasets](#dataset-information) · [code](#code-structure) · [installation](#installation-and-requirements) · [hardware](#hardware-requirements) · [usage](#usage-instructions) · [method](#methodology) · [reproduction steps](#reproducing-the-paper) · [configurations](#reproducing-experimental-configurations) · [results](#reproducing-paper-tables-and-results) · [seeds](#randomness-and-seeds) · [outputs](#expected-outputs) · [models](#pretrained-models-and-external-ai-models)

## Project Description

Word sense disambiguation (WSD) selects the meaning of an ambiguous word in context: for example, whether Ukrainian *коса* refers to a braid or a geographical feature. A dictionary supplies the possible meanings; contextual examples supply evidence for choosing among them.

This project retrieves sentences containing ambiguous lemmas from UberText 2.0, assigns provisional meanings by comparing sentence and definition embeddings, and retains confident assignments. It supplements underrepresented meanings with generated sentences and creates transformed versions of sentences and definitions. A shared transformer encoder is then fine-tuned on sentence–definition triplets, bringing a sentence closer to its assigned definition and farther from another meaning of the same lemma.

The code supports investigating corpus/meaning coverage, training sense-aware embeddings, and evaluating WSD, Ukrainian STS-B, and Ukrainian MTEB tasks. Exploratory notebooks support coverage analysis. The zero-shot LLM comparison, human pseudo-label quality audit, sense-availability result tables, and nine-run statistical aggregation do not have complete executable reproduction workflows here. Generated training sentences use an LLM, but that script is **not** an LLM WSD evaluator.

## Dataset Information

Paths in tables below are **required external inputs or generated destinations**, not claims that those artifacts are included in a clone. Most data, checkpoints, and metric files are excluded by `.gitignore`. Python modules, notebooks, and configuration files linked elsewhere in this README are repository source files.

### External third-party resources

| Resource | Purpose and source | Acquisition and expected location/format | Preprocessing and redistribution status |
| --- | --- | --- | --- |
| Ukrainian dictionary/WSD source data | Definitions, ambiguous lemma inventory, and evaluation examples; prior benchmark described by [Laba et al. (2023)](https://aclanthology.org/2023.unlp-1.2/) | Manual. Current `services.config.SUM_PATH` is `datasets_pre_defined/sum_final.jsonlines`. Raw dictionary JSON Lines, schema below. **TODO: publish the exact download URL, snapshot date, and provenance.** The older README referred to `sum_fixed.jsonlines`; the active code overrides that path with `sum_final.jsonlines`. | Processed by `read_and_transform_data(..., homonym=True)` in `services/utils_data.py`. Underlying dictionary/benchmark license and redistribution permission are not recorded here. Do not infer permission from code availability. |
| Expanded WSD snapshot used in the manuscript | Paper reports 1,434 lemmas, 3,071 meanings, and 15,961 contextual examples after benchmark construction | No separately versioned processed benchmark or construction manifest is committed. A local raw `sum_final.jsonlines` was available during this audit; its identity is recorded below. **Its equivalence to the paper's processed benchmark is not established.** | Requires exact preprocessing software and source snapshot. Rebuilding the upstream dictionary extraction is not implemented in this repository. Publication URL/DOI and redistribution terms remain author TODOs. |
| UberText 2.0, sentence-split news, Wikipedia, and fiction | Naturally occurring unlabeled contexts; [project download page](https://lang.org.ua/en/ubertext/) and [corpus paper](https://aclanthology.org/2023.unlp-1.1/) | Manual download to `datasets_pre_defined/`; UTF-8 text, one sentence per line, compressed as `.txt.bz2`. The download commands below preserve URLs from the previous README. | Extraction normalizes whitespace, filters sentences, and matches lemmas with UDPipe or spaCy. Both analyzers' outputs are merged. License/redistribution terms are not preserved here; consult the corpus provider before redistributing extracted text. |
| Ukrainian STS-B, `anikol12/STSB-UK` | Sentence-level similarity evaluation; identifier in `eval/eval_stsb.py` | Downloaded through `datasets.load_dataset`, split `train`, into the library-managed cache. Expected columns: `sentence1`, `sentence2`, `score`. No dataset revision is specified. | Script changes the score of exactly identical sentence pairs to `1.0`; all other scores are used as supplied. The manuscript reports 5,749 pairs. Translation provenance, exact snapshot, and license need confirmation from the dataset provider. |
| Ukrainian text tasks selected by MTEB | Classification, clustering, retrieval, and bitext mining; task definitions come from the installed `mteb` package | Automatic task-specific downloads/caching. `mteb.get_tasks(languages=['ukr'], modalities=['text'])`, followed by an exact modality filter. No task/dataset revision or frozen task list is supplied. | Each underlying dataset has its own terms. The paper lists 11 tasks, but the script selects tasks dynamically, so a different MTEB installation can produce a different suite. |
| Ukrainian morphology resources | Corpus matching, target extraction, dictionary deduplication, and optional POS reports | Manual UDPipe weights expected at `models/20180506.uk.mova-institute.udpipe`; spaCy package `uk_core_news_sm` installed separately. Stanza Ukrainian resources may be downloaded when POS reports create a pipeline. | These are external model assets, not corpus files. Exact weight revisions and model-specific licenses are not recorded. The UDPipe model download URL/checksum is missing; installing the Python binding does not install this model. |

The 11 MTEB task names reported in the manuscript are SIB200, UkrFormal, SIB200ClusteringS2S, WebFAQQAs, WebFAQQuestions, NTREX, Bible-NLP, Flores, Tatoeba, Belebele, and WebFAQ. These are **manuscript labels**, not a verified list of runnable task identifiers for an unspecified MTEB version. Task-specific URLs, subsets, revisions, and license records must be recovered from the original evaluation environment.

### Source schema and snapshot identity

`read_and_transform_data` expects dictionary records with `lemma`, `prime`, `suffixes`, `tags`, `synsets`, `phrases`, `word_id`, and `url`. Each synset has `sense_id`, a list of `gloss` strings, and an `examples` list whose entries contain `ex_text`. It is not a loader for a flat sentence-label CSV. The processed in-memory table contains `lemma`, lists of `gloss` strings, and lists of `examples`, with one row per retained dictionary meaning.

The inspected raw records contain source links to the SUM-20 dictionary, for example [entry `wordid=1`](https://sum20ua.com/Entry/index?wordid=1). This identifies an underlying dictionary source; it does not establish a downloadable, redistributable benchmark snapshot or its extraction date.

The **local, untracked** raw file examined during README preparation had:

- Filename: `sum_final.jsonlines` at the repository root, rather than the configured data directory.
- Size: 456,658,460 bytes; 138,044 raw dictionary records.
- SHA-256: `6e4ecd7c9fde0a486826f6d033f14a7c020f9b503208abc4262cf0855401da6c`.

These are raw-file measurements, not counts of the expanded WSD evaluation set. A reviewer cannot obtain this file from the committed source alone. The authors must confirm its provenance and relationship to the manuscript before publishing it or treating the checksum as the paper's data identifier.

### Project-generated data and intermediate artifacts

| Artifact | Producer and purpose | Format and destination |
| --- | --- | --- |
| Target lemma list | Derived from the processed dictionary; used for corpus extraction | One lemma per line at `datasets_pre_defined/unique_lemmas_homonyms.txt`; the reproduction guide provides the derivation command |
| Extracted corpus batches | `collect_sentences/collect_ubertext_sentences.py` | JSON Lines despite the `.json` suffix: each line maps lemmas to sentence lists; output chosen with `--save_dataset` |
| Deduplicated corpus contexts | `local_datasets/raw_sentences/process_raw_sentences.py` | `local_datasets/raw_sentences/unique_lemma_sentences.jsonl`; each record has `lemma` and `sentences` |
| Confident pseudo-label assignments | `local_datasets/semi_supervised_2/assign_meaning_to_sentence.py` | `assigned_meanings_mpnet.jsonl` in the same directory; `lemma`, `sentence`, `similarity`, `probability`, `assigned_meaning` |
| Meaning-organized natural pool | Same pseudo-label script | `lemmas_with_meanings_and_sentences_mpnet.json` in the same directory; lemma → first gloss → `{meaning: {gloss, examples}, sentences: [...]}`. Empty sentence lists preserve uncovered meanings. The `examples` metadata comes from the dictionary; it is not the anchor pool. |
| Generated sentences | `collect_sentences/generate_sentences_4_absent_meanings.py` | `local_datasets/semi_supervised_2/generated_sentences.jsonl`; nested lemma/meaning objects with generated sentence strings |
| Merged natural/generated pool | `local_datasets/semi_supervised_2/merge_collected_and_generated.py` | `local_datasets/semi_supervised_2/merged_collected_and_generated_mpnet.json`; generated entries carry `source='generated'` and null similarity/probability |
| Transformation outputs | Scripts under `augment/` | JSON Lines with `sentence` and `augmented` (list of strings), under `local_datasets/augmented/`; exact filenames appear under Expected Outputs |
| Training triplets | `local_datasets/semi_supervised_2/form_triplets.py` | CSV with `lemma,anchor,positive,negative,anchor_target_word_ids,meaning_idx`; target indices are a JSON-encoded list inside the CSV field |
| Fine-tuned checkpoints | `services/trainer/trainer.py` | Hugging Face encoder/tokenizer and mean-pooling SentenceTransformer exports under `models/fine-tuned-models/` |

These artifacts are produced by this project, but contain or derive from third-party text/model outputs. No separate generated-data license, public archive, or artifact checksum manifest is committed. Their redistribution terms must be established separately from the code license.

Older triplet-rewriting scripts in `local_datasets/augmented/translation/` and the archived NT-Xent dataset builder automatically load the Hugging Face identifier `hellcaster/wsd-sentences`, split `back_translation`, expecting `sentence` and `augmented` columns. This is an additional historical project-data dependency, not an input to the active pipeline. No revision, license record, or confirmed mapping to the paper's datasets is supplied.

## Code Structure

```text
.
├── collect_sentences/             # Corpus extraction and LLM sentence generation
├── local_datasets/
│   ├── raw_sentences/             # Merge/deduplicate extracted batches
│   ├── semi_supervised_2/         # Active pseudo-labeling and triplet pipeline; EDA notebooks
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

No original requirements/lockfile, Conda environment, Dockerfile, Slurm job, shell-based experiment driver, committed W&B run export, or CUDA/driver version record was found. The new requirements files list dependencies inferred from imports; **they are not a recovered or validated paper environment**. Installing currently resolved versions can encounter API incompatibilities. Replace them with a tested, fully pinned environment once the authors recover the original records; do not label a new environment as the historical one.

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

- Obtain the UDPipe weight file and dictionary snapshot manually; see Dataset Information. spaCy is loaded at import time by shared embedding utilities, so it is needed even for several nominally UDPipe-based entry points.
- Hugging Face identifiers passed to `from_pretrained`/`SentenceTransformer` normally download weights/tokenizers when missing from the local cache. No model or dataset revision is pinned. No Hugging Face token is read explicitly by project code, and no gated-access requirement is recorded. If upstream access changes, use the libraries' authentication mechanism and record that dependency.
- Back-translation requires **already converted CTranslate2 OPUS models** at `models/translators/opus-mt-zle-en-ct2` and `models/translators/opus-mt-en-zle-ct2`. Tokenizers download automatically; converted weights do not. The repository has an NLLB converter command in comments, but no validated OPUS conversion recipe, source revisions, or conversion metadata. Obtain the original artifacts or recover and validate that recipe before running translation.
- Training loads `.env`. The original INI enables W&B under an author-specific entity. Use `reviewer_config.ini` to disable W&B, or set `wandb_entity`, `wandb_project_name`, and optionally `wandb_run_name` in your INI and supply `WANDB_API_KEY` using `.env.example` as a template. No author-account access is needed when logging is disabled.
- Generation uses `BASE_URL='http://localhost:8000/v1'`, `API_KEY='EMPTY'`, and `MODEL_NAME='Qwen/Qwen3-VL-8B-Instruct'` in its Python script. These are constants, **not environment variables**. An OpenAI-compatible server must already be serving that identifier. The earlier README identifies llama.cpp as the original server, but no launch command, version, quantization, or server configuration is supplied. The generation script does not launch the server or download Qwen weights.

## Hardware Requirements

The manuscript reports training on **one NVIDIA RTX 3090**. The trainer uses one selected device, defaults to CUDA when available, and accepts `--device cuda:0` or `--device cpu`. Although `enable_gpu_parallel=True` appears in the configuration, the current trainer does not implement that flag. It is not evidence of multi-GPU training.

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

The highest-probability meaning is retained only when **probability ≥ 0.9 AND cosine similarity ≥ 0.6**. Exact equality between the sentence and the meaning's first gloss is excluded. Batch size is **2048**. Filtering is inside the assignment script; there is no separate unfiltered-prediction artifact or filtering command. Dictionary examples remain metadata and are not encoded to produce these pseudo-labels.

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

`MAX_SENTENCES_PER_MEANING=300` caps sampled source sentences and distributes a **target of 300 candidate triplet draws** across each nonempty meaning. The script tries unused combinations within each source sentence, then permits repeats once combinations are exhausted. The computed unique-combination cap is not used by the final loop. Failed target-word matches are skipped, so output can be below 300 rows per meaning and is not guaranteed unique. Meanings with zero contexts produce no rows. This differs from the manuscript's maximum of 100 unique triplets.

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
| Train/validation split | CSV shuffled with pandas `random_state=42`; **99% / 1% by triplet row** |
| Validation / early stopping | Before each epoch and every **200** minibatches after batch index 0; patience **10** in INI, **15** in dataclass fallback |
| Trainable layers | `layers_to_unfreeze=0` means all layers trainable |
| AMP | FP16 plus GradScaler on CUDA; disabled on CPU |
| Data loading | **4** workers, prefetch factor **2**, persistent workers, pinned memory; both train and validation shuffled |
| Random initialization | Pretrained weights used; reinitialization settings are not wired into the trainer |

The supplied original INI points to a dropout CSV that the active default builder does not produce. The new `reviewer_config.ini` changes only the input CSV to the active builder's output and disables W&B; optimization/pooling settings are unchanged. It is a **current-code example, not a recovered paper-run configuration**.

Validation drives checkpoint selection using loss, not WSD accuracy. `_best` is saved only at an improving within-epoch check with `batch_count > 0`; epoch/final exports also exist. Final WSD evaluation loads `_final`, not `_best`. MNR and NT-Xent branches are experimental and have known defects; they are not required for the manuscript's cosine-triplet experiments.

### 10. WSD inference/evaluation

`WordSenseDetector` groups candidate gloss lists by lemma and predicts once for each processed **dictionary-meaning row**, whose `examples` field can contain multiple sentences. The evaluator's `max_sim_across_all_examples` strategy chooses the candidate meaning with the greatest cosine similarity across its individual definitions and all usable target-word example embeddings. This is a maximum, not the averaged meaning representation used during pseudo-labeling.

Inference uses mean-pooled target-word hidden states from the last layer and whole-definition mean embeddings. `prediction_accuracy` drops rows with null values, compares the first gold/predicted gloss, and computes accuracy over remaining rows. It does **not** produce one independent prediction for each of the manuscript's 15,961 contexts. Report the evaluation unit and dropped coverage explicitly; the current metric cannot be assumed to reproduce a context-level WSD table. Training `pool_targets=False` does not switch this WSD evaluator to full-sentence inference.

### 11–12. STS-B and MTEB

STS-B uses `EmbeddingSimilarityEvaluator` on full-sentence embeddings and the dataset's `train` split as evaluation data. Exactly identical strings receive target score 1.0; other values are not rescaled by the script. Pearson and Spearman cosine correlations are exported as percentages.

MTEB loads SentenceTransformer models, dynamically selects Ukrainian text tasks, invokes `mteb.evaluate`, caches results, saves task predictions, and serializes each task's `scores`. It does not freeze the paper's task suite, map checkpoints to configurations, or aggregate multiple runs into paper tables.

## Reproducing the Paper

**Read this as a dependency-ordered execution guide for the current code.** Steps with unavailable external assets or missing paper procedures are explicitly marked. Completing the executable stages alone does not resolve the manuscript discrepancies. Record every run's source revision, environment, input hashes, configuration, seeds, and output identity before claiming paper reproduction.

### Step 0 — Prepare the dictionary, morphology assets, and corpus

Obtain the exact author-approved raw dictionary snapshot and place it at the configured location. If using the locally supplied root-level file described above, the following copies it without overwriting an existing destination:

```bash
cp -n sum_final.jsonlines datasets_pre_defined/sum_final.jsonlines
```

This command requires that local file; it is not a download method. Obtain `models/20180506.uk.mova-institute.udpipe` separately and install the spaCy model as described above. **Blocked for a fresh clone until the dictionary and UDPipe asset locations are supplied.**

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

This new convenience command invokes existing preprocessing; it does not recover the missing upstream snapshot-construction process. Confirm the printed counts against the manuscript rather than assuming they match. The output lemma file is overwritten.

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

The **0.9 probability / 0.6 cosine** confidence filter already ran in Step 2; do not run a second invented filtering stage.

**Missing paper step:** the manuscript's “Benchmark isolation and leakage prevention” paragraph specifies removal of corpus contexts with cosine similarity ≥0.95 to benchmark examples before augmentation. No implementation or filtered artifact manifest for that operation was found. Recover the original procedure and inputs before treating the following stages as a paper reproduction. Do not silently substitute exact-match deduplication or a newly chosen embedding model.

### Step 4 — Generate examples and merge

Prerequisites: resolved Step 3, a local server serving the configured Qwen identifier, and fresh generation output. Review `INPUT_FILE`, `OUTPUT_FILE`, `MIN_SENTENCES`, `BASE_URL`, `API_KEY`, and `MODEL_NAME` in the generation script.

```bash
python -m collect_sentences.generate_sentences_4_absent_meanings
python -m local_datasets.semi_supervised_2.merge_collected_and_generated
```

Generation reads the meaning-organized natural pool and appends `generated_sentences.jsonl`. Merge reads that file plus the natural pool and overwrites `merged_collected_and_generated_mpnet.json`, all under `local_datasets/semi_supervised_2/`. The server's sampling state is not recorded by these commands. Merge warns about missing meanings or fewer than five examples, but does not remedy them. For the Natural configuration, skip generation/merge and point the triplet builder directly at the natural pool as described below.

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

Current defaults read the merged pool and all four individual transformation pairs. Output: `local_datasets/semi_supervised_2/triplets_semi_supervised_all_augs_mixed_300.csv`. The script sets Python's seed to 42, loads both spaCy and UDPipe, and computes MPNet target indices. **This produces the current 300-draw behavior, not the manuscript's 100-unique-triplet algorithm.** Do not change the constant alone and claim equivalence: uniqueness/sampling logic also differs.

### Step 7 — Train a current-code example

```bash
python -m services.trainer.trainer \
  --config services/trainer/reviewer_config.ini \
  --device cuda:0
```

Inputs: Step 6 CSV, pretrained encoder/tokenizer, and morphology assets. Output: epoch, optional best/early-stopped, and final model directories under `models/fine-tuned-models/`. The reviewer INI disables W&B and selects Step 6's default CSV. With W&B disabled, the run identifier is a timestamp. Record that identifier and the full INI immediately; original seed values are still hard-coded in the trainer. Final WSD evaluation also requires the dictionary snapshot and POS-report resources.

For another configuration, edit `train_data_path` and pooling in your chosen INI before running. The trainer's parser currently accepts a nonexistent config path and silently falls back to dataclass defaults; verify that the intended file exists and was loaded. The original `fine_tuning_config.ini` remains available but requires its dropout CSV and either authorized W&B settings or logging disabled.

### Steps 8–10 — Evaluate WSD, STS-B, and MTEB

Set a shell variable to an **actual exported model directory**. This command prompts for the path so no nonexistent run identifier is presented as a runnable checkpoint:

```bash
printf 'Path to the exported model directory: '
read -r MODEL_PATH
python -m eval.eval_wsd --model-path "$MODEL_PATH" --device cuda:0 --no-reports
python -m eval.eval_stsb --models "$MODEL_PATH" --device cuda:0
python -m eval.eval_mteb --models "$MODEL_PATH" --device cuda:0
```

WSD inputs include the dictionary and morphology resources. STS-B/MTEB load their external datasets. Outputs follow the Usage and Expected Outputs sections. Explicitly record whether the directory is `_best`, an epoch export, or `_final`: the trainer itself evaluates `_final`, and no manifest maps historical checkpoints to the paper's result rows. Each STS invocation overwrites the same CSV, so preserve it before the next invocation or pass multiple models together. MTEB uses model-path-based caches; reusing a path with changed weights can reuse stale results.

There is no executable Step 11 for the paper's zero-shot LLM comparison or a final command that regenerates all manuscript tables.

## Reproducing Experimental Configurations

The following **manuscript names** can be related to existing data pools. They are not named CLI presets or a recovered manifest of the original experiments. Selection currently requires editing constants in `local_datasets/semi_supervised_2/form_triplets.py` and saving that source/config with the run.

| Paper configuration | `DATASET_PATH` choice | Augmentation selection in the triplet builder | Reproduction limit |
| --- | --- | --- | --- |
| Natural | `local_datasets/semi_supervised_2/lemmas_with_meanings_and_sentences_mpnet.json` | Set both `USE_AUGMENTED=False` and `USE_DEFINITIONS_AUGMENTED=False` | Natural pool exists only after preparation; isolation step and paper sampling algorithm missing |
| Generation | `local_datasets/semi_supervised_2/merged_collected_and_generated_mpnet.json` | Both flags false | Requires original generation settings/artifacts |
| Generation + MLM | Same merged pool | Both flags true; retain only the `mask/` entry in each path tuple | Original augmentation seed settings unavailable |
| Generation + Dropout | Same merged pool | Both flags true; retain only the `dropout/` entry in each tuple | Original INI names a dropout CSV, but no matching saved builder configuration is supplied |
| Generation + Back-translation | Same merged pool | Both flags true; retain only the `translation/` entry in each tuple | Converted weights/decoding provenance unavailable |
| Generation + Shuffling | Same merged pool | Both flags true; retain only the `token_shuffling/` entry in each tuple | Paper's verbal window description differs from implementation |
| Stochastic combination | Same merged pool | Both flags true; select only the currently commented `all_together/` entry in each tuple | Stop/continue probability and branch retention require resolution |
| All combined | Same merged pool | Both flags true; four individual methods, the active defaults | This mapping is supported by the code's pool union, but no original per-run manifest confirms the precise published composition |

Change `OUTPUT_CSV` to a distinct destination for every dataset/configuration and point the training INI's `train_data_path` at it. Filenames are under user control; no seed-specific filenames or artifacts for all these rows are currently supplied. The default filename does not change automatically when flags change.

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

The manuscript specifies **three dataset-construction seeds × three training seeds = nine runs per configuration/pooling setting**. The repository supplies only one fixed set of library seeds, no three-value seed lists, and no loop/launcher mapping datasets to training runs. Exact nine-run reproduction is therefore unavailable. Recover the three dataset seed values, the three complete training RNG settings, and the generated-data provenance before constructing a run matrix. Repeating the current command nine times does not implement the manuscript design.

## Reproducing Paper Tables and Results

Paper result labels below refer to groups identifiable in the supplied manuscript sources. The numeric manuscript tables are not treated as executable result artifacts.

| Paper result | Implemented script/notebook | Input | Output / missing link |
| --- | --- | --- | --- |
| Dictionary/corpus coverage | `local_datasets/sum_and_ubertext_eda.ipynb` | Dictionary snapshot; deduplicated contexts | Interactive counts, descriptive statistics, plots; no consolidated table exporter |
| Meaning coverage after confidence filtering | `local_datasets/semi_supervised_2/eda_lemmas_with_meanings.ipynb` | Meaning-organized natural pool | Interactive counts/distributions, including uncovered meanings/lemmas; no publication-table generator |
| Pseudo-label human QA | Assignment script supplies candidate records only | Retained assignments plus human labels | The manuscript's 400-record sample, annotations, sampling seed, and agreement/CI computation are not provided |
| WSD baselines | `python -m eval.eval_wsd` with explicit model arguments as above | Chosen model, dictionary, morphology assets | Printed dictionary-sense-row accuracy, optional error CSV; full paper baseline roster and context-level protocol not reproduced |
| Augmentation/pooling comparison | Triplet builder → trainer → WSD evaluator | Selected pools, INI, model exports | Per-run loss/checkpoints and WSD return value; no nine-run mapping, mean/SD aggregation, or table script |
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
| Generation | No seed, temperature, or top-p in the API request; server defaults and model/quantization determine sampling |
| Standalone dropout/shuffling | Python `random` used without an entry-point seed |
| MLM | `Masker(seed=42)` resets the global Python RNG; replacement sampling uses PyTorch without an explicit seed in that stage |
| Translation | Sampling parameters provided, but no CTranslate2 seed configured |
| Stochastic orchestrators | `random.seed(42)`; `Masker` also resets that global seed; batch-level choices share the RNG with augmentation operations |
| Triplet builder | `random.seed(42)` in `__main__`; context/definition sampling affected by input and set ordering |
| Trainer | `torch.manual_seed(47)`, `random.seed(92)`, `np.random.seed(39)` at module import; these are three libraries in **one run**, not three training runs |
| Split | pandas shuffle uses independent `random_state=42`, then positional 99/1 split |
| DataLoader/model | Training dropout and shuffled train/validation loaders use RNG state; no explicit loader generator or worker seed function |
| CUDA determinism | No separate CUDA seed call, deterministic-algorithm setting, or cuDNN determinism configuration is specified; fixed seeds alone do not establish exact reproducibility |
| Exploratory notebooks | PCA/TSNE references use `random_state=42`; KMeans uses `random_state=0`; not the paper's training seed grid |

Archived dataset rewrites also use Python seed 42, but are not the active scientific workflow. Neither code nor supplied manuscript provides the complete three-by-three seed values. No expensive experiment was rerun to infer them. A future seed interface must address all stage-specific RNGs and ordering rather than merely looping over `torch.manual_seed`.

## Expected Outputs

All destinations are generated at runtime and are generally Git-ignored. See Dataset Information for record schemas.

| Stage | Expected destination / content |
| --- | --- |
| Extraction | Six `local_datasets/raw_sentences/lemma_examples_samples_<analyzer>_<corpus>.json` files from the documented loop; appended JSONL batches |
| Deduplication | `local_datasets/raw_sentences/unique_lemma_sentences.jsonl` |
| Pseudo-labeling | `local_datasets/semi_supervised_2/assigned_meanings_mpnet.jsonl` and `lemmas_with_meanings_and_sentences_mpnet.json` |
| Generation/merge | `local_datasets/semi_supervised_2/generated_sentences.jsonl` and `merged_collected_and_generated_mpnet.json` |
| Dropout / MLM / shuffling | In `local_datasets/augmented/dropout/`, `mask/`, or `token_shuffling/`: `augmented_sentences.jsonl` and `augmented_sentences_definitions.jsonl` |
| Back-translation | `local_datasets/augmented/translation/augmented_sentences_translated_v3.jsonl` and `augmented_sentences_translated_definitions.jsonl` |
| Stochastic combination | `local_datasets/augmented/all_together/augmented_sentences_3.jsonl` and `augmented_sentences_definitions_3.jsonl`; root-level `selected_augmenters_log.json` and `selected_augmenters_definitions_log.json` record batch-level choices |
| Default triplets | `local_datasets/semi_supervised_2/triplets_semi_supervised_all_augs_mixed_300.csv` |
| Checkpoints | Under configured save directory: `model_<run_id>_<epoch>`, `model_<run_id>_best` when eligible, `model_<run_id>_<epoch>_early_stopped` when triggered, and `model_<run_id>_final` |
| Training metrics | Console progress; W&B only when enabled. No complete local loss/metric CSV or resumable optimizer/scheduler/RNG checkpoint is implemented. |
| WSD | `eval_wsd.log`; optional root `badly_predicted.csv`; POS helper may create `data/pos_precalculation.pkl`. No complete prediction JSONL. Accuracy is printed by the CLI and returned by `evaluate_wsd`. |
| STS-B | Root `sts_results.csv`, with model index and `pearson_cosine`, `spearman_cosine` columns |
| MTEB | `cache/mteb_<model_id_with_slashes_replaced>/`; `eval/mteb_prediction/<same_id>/`; `eval/mteb_results/<same_id>/<task_name>_results.json` |
| Environment report | Console JSON, or `logs/environment-report.json` and `logs/requirements-resolved.txt` using the optional commands above |

Extraction and generation append; merging, triplet construction, augmentation writers, and STS summaries overwrite. Translation's existing resume logic is not reliable: it looks for `original`, while the writer saves `sentence`, and the writer opens in overwrite mode. Preserve outputs and use distinct experiment destinations. A checkpoint directory existing is not proof of a successful complete training run, because save/training exceptions are currently caught broadly.

Optional frequency reports in `services/utils_results.py` additionally require `data/frequents.pkl`. `prepare_frequent_dictionary` can create it from a compatible compressed frequency source, but that source's acquisition path is not documented. Frequency reports default to disabled and are not the missing sense-availability analysis.

## Pretrained Models and External AI Models

No `revision=` argument or immutable model commit is pinned in the active loaders. Model identifiers below are taken from source; model availability, access conditions, and licenses need to be archived with the release.

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

## Reproducibility issues found

This audit inspected all Python modules, configuration files, notebook sources/metadata, the previous README, Git-tracked file inventory, and locally supplied manuscript/supplement sources. There were no committed environment/lockfiles, Docker/Slurm/shell launchers, automated tests, CI configuration, or full W&B run exports to recover. The local dictionary and UDPipe archive were inspected without running model inference. Findings refer to the supplied manuscript sections, since the manuscript files themselves were untracked at audit time.

### Remaining issues requiring resolution

| Severity | File / relevant location | Problem | Recommended fix |
| --- | --- | --- | --- |
| **critical** | `services/config.py::SUM_PATH`; `services/utils_data.py::read_and_transform_data`; manuscript “Ukrainian WSD benchmark” | No published, immutable expanded snapshot/construction manifest. Local raw file is not verified against 1,434/3,071/15,961 processed counts. | Confirm source, preprocessing, counts, hash, public archive/DOI, and redistribution permission; preserve the exact processed candidate inventory. |
| **critical** | `assign_meaning_to_sentence.py::process_lemma`; manuscript “Benchmark isolation and leakage prevention” | Paper's cosine ≥0.95 benchmark-overlap removal has no implementation/artifact manifest here. | Recover the original filter, embedding settings, excluded-record list, and stage ordering; audit overlap before new training. |
| **critical** | `local_datasets/semi_supervised_2/form_triplets.py::MAX_SENTENCES_PER_MEANING`, `get_recommended_number_of_sentences`, final sampling loop; manuscript “Contrastive triplet construction” | Code targets 300 draws, ignores computed unique cap, and permits repeats; manuscript says at most 100 unique triplets. | Recover the actual experiment version/data. Resolve algorithm and manuscript together; do not change only the numeric constant. |
| **critical** | `services/word_sense_detector.py::run`; `PredictionStrategy.max_sim_across_all_examples`; `services/utils_results.py::prediction_accuracy` | Evaluation aggregates dictionary examples into one sense-row prediction and excludes null rows; manuscript describes contextual-example WSD. | Confirm which protocol produced each result; recover/export per-context predictions and coverage if that is the reported unit. |
| **critical** | `services/trainer/trainer.py` module seeds; augmentation entry points; manuscript “Training procedure” | No three dataset seeds × three training settings, run manifest, or aggregation driver. | Supply exact seed lists/library mapping, dataset hashes, checkpoint IDs, and mean/SD aggregation procedure. |
| **important** | `requirements.txt`; notebook metadata | New dependency inventory is unpinned and unvalidated; only notebook Python 3.10.14 is established. | Recover the original environment, pin transitive packages and system/CUDA versions, then run an integration smoke test. |
| **important** | `augment/common.py::markov_process`; `augment/augment_all_together*.py`; manuscript “Combining Augmentations” | Code stops with probability 0.75; manuscript describes continuing with 0.75. Code reduces variants starting at step two; wording says after the second augmentation. | Confirm original stochastic process and archive selection logs; reconcile documentation/code before reproducing this configuration. |
| **important** | `augment/augment_all_together*.py::main`, `final_augmented_texts[original] = augmented_list` | Multiple branches mapping to one source overwrite earlier branches. | Recover intended retention rule, add an isolated branching example, and version any correction as a behavior change. |
| **important** | `services/config.py::MIN_LEMMA_LENTH`; `services/utils_data.py::read_and_transform_data`; manuscript benchmark inclusion criteria | Code keeps lengths >3 (minimum four); paper says excludes lengths <3 (minimum three). Deduplication also depends on unpinned spaCy. | Confirm the historical filtering code and publish processed benchmark counts/inventory. |
| **important** | `augment/token_shuffling/token_shuffler.py::_local_shuffle`; manuscript “Local word shuffling” | Code uses overlapping length-three chunks with stride two; text describes possible swaps within three preceding/following words. | Clarify the operation actually used; retain exact augmentation artifacts. |
| **important** | `collect_sentences/generate_sentences_4_absent_meanings.py::main` | Local server/quantization/sampling settings absent; parsed count and uniqueness not enforced; append-only reruns repeat work. | Provide server launch/configuration, generation seed/defaults and original outputs; add validated count/resume handling separately. |
| **important** | `augment/translation/back_translator.py::HelsinkiCTranslateTranslator`; `services/config.py::PATH_TO_SOURCE_UDPIPE` | Required converted OPUS and UDPipe weights absent; exact acquisition/conversion revisions unknown. | Archive redistributable assets or publish validated download/conversion instructions and hashes. |
| **important** | `form_triplets.py::get_target_word_embedding_idx`; `services/trainer/datasets.py`; `losses.py::_batch_mean_pooling` | Indices built without truncation, training truncates to 128, mask checks only `>=0`; large indices can address another row or exceed bounds. | Align index creation with training tokenization and validate bounds/target retention; quantify affected examples before changing datasets. |
| **important** | `services/utils_embedding_calculation_v2.py::same_lemma`, `_find_target_word_in_tokenized_text`, `get_target_word_embedding` | Fuzzy substring matching and reconstructed subwords can select a wrong target; training/inference differ on repeated occurrences. | Validate against annotated target spans, expose rejection counts, and use a consistent documented alignment policy. |
| **important** | `eval/eval_mteb.py::main`; `eval/eval_stsb.py::main`; all model loaders | Model/data revisions, MTEB task list/subsets/prompts, and original library versions are not pinned. | Recover immutable model/data/task manifests; test API compatibility and avoid stale caches when weights change. |
| **important** | `eval/eval_stsb.py::main`; manuscript STS-B description | Code resets identical-string pair scores to 1.0; manuscript describes normalized dataset scores and translation artifacts but does not clearly specify this correction. | Document the exact scoring policy and archive evaluated targets; confirm it matches reported correlations. |
| **important** | `eval/`; coverage notebooks; manuscript QA/sense-availability/LLM tables | Missing LLM evaluator, human QA sample/annotations, sense-availability analysis, full baseline manifest, and multi-run table aggregation. | Add original scripts and permissible raw predictions/annotations with table-to-run mappings. |
| **important** | `services/trainer/trainer.py::_setup_data` | Row split follows triplet/augmentation construction; related anchors/definitions can cross partitions. This matches the manuscript but can make validation optimistic. | Measure overlap and document the interpretation; any grouped split is a new experiment, not a silent reproduction fix. |
| **important** | `services/trainer/trainer.py::_save_model`, `train`, `evaluate_epoch` | Exceptions are caught broadly; best checkpoints are not saved for improvements at pre-epoch checks; final evaluation uses `_final`; no exact training-state resume. | Recover the published checkpoint-selection policy, surface failures, save provenance/training state, and validate export/reload. |
| **important** | `collect_sentences/collect_ubertext_sentences.py::load_tokenizer_model`, `_normalize_text_udpipe`; `services/trainer/data_factory.py::_get_collate_fn` | Global NLP models and nested collators are incompatible with some spawn-based workers. UDPipe collection returns lemmas from only the last parsed sentence in a source line. | Specify/test process-start environment; initialize workers explicitly; check multi-sentence-line handling before changing extraction. |
| **important** | `services/trainer/losses.py::MNRLoss.forward`, `NTXentLoss.__init__`; `trainer.py::_init_loss` | MNR drops target indices; NT-Xent expects a `.backbone` wrapper but receives raw AutoModel. | Mark as unsupported experimental branches until repaired and tested; avoid them for the paper's triplet runs. |
| **important** | `augment/translation/augment_translation.py::main`; `augment/common.py::ThreadedWriter` | Resume reads wrong record key and writer truncates files; writer errors are printed and can discard batches. | Preserve original outputs, implement explicit safe resume/failure handling, and verify input/output counts. |
| **important** | Repository root; external data/model resources | No code license or complete third-party redistribution record. | Authors select a code license and document each dataset/model's terms before archival. |
| **minor** | `services/trainer/training_config.py::from_config`; original INI | Missing config silently falls back; original INI names missing dropout CSV and author W&B entity. | Validate config existence/types and supply recovered per-experiment presets; reviewer example avoids the default path/account mismatch. |
| **minor** | `trainer.py::_setup_optimizer`, `train_epoch`; `training_config.py` | `warmup_ratio=0` with `apply_warmup=True` calls `.step()` on `None`; GPU-parallel/reinitialization flags are unused. | Guard scheduler use and reject/document unsupported options; defaults used by the paper are unaffected by the zero-ratio branch. |
| **minor** | `demo.py::main`; evaluation script defaults; archived code | Fixed devices/checkpoint IDs and outdated paths remain in defaults/archive; optional frequency source missing. | Use explicit evaluation CLI overrides; supply manifests for historical paths and document optional report inputs. |

### Supporting fixes made with this README

- Added source-derived `requirements.txt`, optional notebook requirements, `.env.example`, and a dependency-free environment-report script. Added Git-ignore exceptions so requirements files can be committed.
- Aligned generation/merge inputs and outputs with the active `semi_supervised_2` pseudo-label pool and the merged filename expected downstream. This changes file routing, not generation/pseudo-label parameters. Archived scripts were preserved.
- Aligned OPUS augmentation output directories with triplet-builder inputs; corrected definition translation to pass `batch['sentence']` instead of iterating the batch dictionary's keys. Newly generated definition translations therefore correct an execution/data-routing defect and must not be represented as recovered historical outputs.
- Added model/device selection to STS-B/MTEB CLIs and model/tokenizer/dictionary/device/report selection to WSD CLI. Existing default model lists and metric behavior were preserved; WSD CLI now prints the returned accuracy explicitly.
- Added `reviewer_config.ini`, selecting the active builder output with W&B disabled and unchanged optimization/pooling settings.
- Explicitly requested hidden states in demo model loading, as required by its existing pooling utilities.

No thresholds, margins, scientific seed values, triplet counts, stochastic transition logic, evaluation aggregation units, or reported results were changed. No expensive experiment was run. README/source/CLI checks and small isolated supporting-code checks do not establish model-training or paper-result reproducibility.
