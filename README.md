# ConEFU v2

## Instructions

### Create environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Collect sentences from UberText 2.0 
1. Download UberText 2.0 dataset from [here](https://lang.org.ua/en/ubertext/)
We used `news`, `wiki`, `fiction` splits of the dataset. We used "split into sentences" version of the dataset. Put sentences into `datasets_pre_defined` folder. Or use following command.
```bash
cd datasets_pre_defined
wget https://lang.org.ua/static/downloads/ubertext2.0/fiction/sentenced/ubertext.fiction.filter_rus_gcld+short.text_only.txt.bz2
wget https://lang.org.ua/static/downloads/ubertext2.0/wikipedia/sentenced/ubertext.wikipedia.filter_rus_gcld+short.text_only.txt.bz2
wget https://lang.org.ua/static/downloads/ubertext2.0/news/sentenced/ubertext.news.filter_rus_gcld+short.text_only.txt.bz2
```
2. Create a file with lemmas of interest in `datasets_pre_defined/unique_lemmas_homonyms.txt`, one lemma per line.

3. Run collection script
```bash
python collect_ubertext_sentences.py --source_dataset <path_to_ubertext_dataset> --save_dataset <path_to_save_gathered_dataset> --lemmas_file <path_to_lemmas_file>
```

Use ```python collect_ubertext_sentences.py --help``` to see all available options. Also, you man modify default paths and parameters in `services/config.py` file.

We gathered data using `udpipe` and `spacy` tokenizers. You can choose either of them using `--tokenizer` argument.

### Process gathered dataset
To process the gathered dataset, use `datasets/raw_sentences/process_raw_sentences.py` script. This will merge all splits of gathered dataset into one file and filter out duplicate sentences.
```bash
python datasets/raw_sentences/process_raw_sentences.py
```