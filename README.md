# 🧠 Semi-Supervised Word Sense Disambiguation for Ukrainian Sentence Embeddings

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/platform-Linux-lightgrey)
![Status](https://img.shields.io/badge/status-research-orange)

A research codebase for **Word Sense Disambiguation (WSD)** and **embedding fine-tuning** for the Ukrainian language. The project builds lemma-centered datasets, augments them, trains contrastive models, and evaluates the resulting embeddings on WSD and MTEB-style benchmarks.

---

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Repository Map](#repository-map)
- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Data Inputs](#data-inputs)
- [Execution Order](#execution-order)
- [Fast Path](#fast-path)
- [Reproducibility Notes](#reproducibility-notes)
- [Citation](#citation)

---

## Pipeline Overview

```mermaid
A[🗂️ Collect Sentences] --> B[🏷️ Assign Meanings]
B --> C[🔀 Augment Data]
C --> D[📐 Build Triplets]
D --> E[🏋️ Train Model]
E --> F[📊 Evaluate]
```

**What this repository does:**

- Collects lemma-specific Ukrainian sentences from UberText 2.0
- Assigns candidate meanings via a semi-supervised pipeline
- Builds triplet contrastive datasets (anchor, positive, negative)
- Augments data with dropout, masking, token shuffling, translation, and Markov mixing
- Fine-tunes transformer encoders for sense-aware representations
- Evaluates models on WSD, STS, and MTEB tasks

---

## Repository Map

```
.
├── collect_sentences/       # UberText sentence collection and generation utilities
├── local_datasets/          # Dataset building, semi-supervised pipeline, triplet formation
├── augment/                 # Augmentation modules and orchestrators
├── services/                # Core logic: config, embedding utilities, UDPipe, training code
│   └── trainer/             # Training entry point and fine_tuning_config.ini
├── eval/                    # WSD, STS, and MTEB evaluation scripts
├── models/                  # Trained checkpoints and local model assets
│   └── translators/         # Local translator models used by translation augmenters
└── datasets_pre_defined/    # Raw corpora, lemma lists, and evaluation datasets
```

---

## Prerequisites

- 🐧 Linux environment (recommended)
- 🐍 Python 3.10+
- 🎮 CUDA-capable GPU (strongly recommended for training and evaluation)
- UDPipe model file at `models/20180506.uk.mova-institute.udpipe`
- SpaCy Ukrainian model (`uk_core_news_sm`) if using SpaCy-based steps

---

## Environment Setup

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

Install core dependencies:

```bash
pip install -r requirements.txt
python -m spacy download uk_core_news_sm
```

---

## Data Inputs

### 1. UberText 2.0

Download sentence-split corpora (news / wiki / fiction) from [lang.org.ua](https://lang.org.ua/en/ubertext/) and place them in `datasets_pre_defined/`:

```bash
cd datasets_pre_defined
wget https://lang.org.ua/static/downloads/ubertext2.0/fiction/sentenced/ubertext.fiction.filter_rus_gcld+short.text_only.txt.bz2
wget https://lang.org.ua/static/downloads/ubertext2.0/wikipedia/sentenced/ubertext.wikipedia.filter_rus_gcld+short.text_only.txt.bz2
wget https://lang.org.ua/static/downloads/ubertext2.0/news/sentenced/ubertext.news.filter_rus_gcld+short.text_only.txt.bz2
cd ..
```

### 2. Lemma List

Create `datasets_pre_defined/unique_lemmas_homonyms.txt` with one lemma per line. This can be taken from the SUM dataset or compiled from other Ukrainian lexical resources.

### 3. Evaluation Datasets

The WSD evaluation expects:

- `datasets_pre_defined/sum_fixed.jsonlines` — available on Hugging Face from prior Ukrainian NLP research.

Default paths are configured in `services/config.py`.

---

## Execution Order

Below is the recommended end-to-end pipeline. Some scripts in `local_datasets/` are experimental variants from earlier iterations — the stages below reflect the current active pipeline.

### Stage A — Collect and Normalize Raw Sentences

**1. Collect sentences containing target lemmas:**

```bash
python3 -m collect_sentences.collect_ubertext_sentences \
    --source_dataset datasets_pre_defined/ubertext.news.filter_rus_gcld+short.text_only.txt.bz2 \
    --save_dataset local_datasets/raw_sentences/lemma_examples_samples_udpipe_news.json \
    --lemmas_file datasets_pre_defined/unique_lemmas_homonyms.txt \
    --tokenizer udpipe
```

Repeat for each UberText source (`news` / `wiki` / `fiction`) and with both `--tokenizer udpipe` and `--tokenizer spacy`.

**2. Merge and deduplicate:**

```bash
python3 -m local_datasets.raw_sentences.process_raw_sentences
```

Output: `local_datasets/raw_sentences/unique_lemma_sentences.jsonl`

---

### Stage B — Assign Meanings and Prepare Semi-Supervised Data

**3. Assign candidate meanings to collected sentences:**

```bash
python3 -m local_datasets.semi_supervised_2.assign_meaning_to_sentence
```

**4. *(Optional)* Generate extra sentences for meanings with low coverage:**

```bash
python3 -m collect_sentences.generate_sentences_4_absent_meanings
```

This step calls an OpenAI-compatible API endpoint. The original research used a local [Llama.cpp](https://github.com/ggerganov/llama.cpp) server with `Qwen/Qwen3-VL-8B-Instruct`. 

**5. Merge collected and generated sentences:**

```bash
python3 -m local_datasets.semi_supervised_2.merge_collected_and_generated
```

Output: `local_datasets/semi_supervised_2/merged_collected_and_generated_mpnet.json`

---

### Stage C — Augmentation

**6. Run all augmentation strategies:**

```bash
python3 -m augment.augment_all_together
python3 -m augment.augment_all_together_definitions
```

> ⚠️ Translation augmenters require local translator models under `models/translators/`. Individual augmentation scripts in `augment/` can also be run separately and merged manually for finer control.

---

### Stage D — Build Triplets for Contrastive Training

**7. Build triplet training CSV:**

```bash
python3 -m local_datasets.semi_supervised_2.form_triplets
```

Output: `local_datasets/semi_supervised_2/triplets_semi_supervised_all_augs_mixed_300.csv`

---

### Stage E — Train

**8. Configure training in `services/trainer/fine_tuning_config.ini`, then launch:**

```bash
python3 -m services.trainer.trainer \
    --config services/trainer/fine_tuning_config.ini \
    --device cuda:0
```

---

### Stage F — Evaluate

**9. WSD evaluation:**

```bash
python3 -m eval.eval_wsd
```

**10. MTEB evaluation:**

```bash
python3 -m eval.eval_mteb
```

**11. STS evaluation:**

```bash
python3 -m eval.eval_stsb
```

---

### Stage G — Demo / Qualitative Check

**12. Run the demo:**

```bash
python3 demo.py
```

Edit the model path, target lemma, and input sentences at the top of `demo.py` to inspect sense-aware embeddings for your own examples.

---

## ⚡ Fast Path

If prepared triplets already exist in `local_datasets/semi_supervised_2/`, skip Stages A–D and run:

```bash
python3 -m services.trainer.trainer --config services/trainer/fine_tuning_config.ini --device cuda:0
python3 -m eval.eval_wsd
python3 -m eval.eval_mteb
```

---

## Reproducibility Notes

- Several scripts contain hardcoded paths and constants near the top of the file — review and update them before running on a new machine.
- Files in `local_datasets/` include experimental variants from earlier iterations of the pipeline; they can be safely ignored unless you are investigating prior experiments.
- Training results may vary slightly depending on GPU type, driver version, and library versions. Pinning dependencies mitigates this.

---

## Citation

To be added once the research paper is published. For now, please contact the author if you wish to reference this codebase in your work.