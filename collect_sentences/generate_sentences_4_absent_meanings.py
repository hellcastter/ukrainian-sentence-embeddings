"""
Generate synthetic sentences for meanings that lack sufficient examples using LLM.
The generated sentences are saved in a JSONL file for further processing.
"""

import json
from collections import defaultdict

from tqdm import tqdm
from openai import OpenAI


## Configuration
INPUT_FILE = "local_datasets/semi_supervised/lemmas_with_meanings_and_sentences.json"
OUTPUT_FILE = "local_datasets/semi_supervised/generated_sentences.jsonl"

MIN_SENTENCES = 5
BASE_URL = "http://localhost:8000/v1"
API_KEY = "EMPTY"
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"


## Prompt template
GENERATE_PROMPT = """Ти експерт з української мови, зокрема з лексикографії. Твоє завдання - допомогти створити приклади речень для значень слів, які наразі не мають достатньої кількості прикладів.

Мені потрібна допомога зі словом "{lemma}". 
Це слово має таке значення:
{glosses} 

Ось кілька прикладів використання цього значення в реченнях: 
{existing_sentences}.

Твоя задача - згенерувати додаткові приклади речень, які ілюструють це значення. Будь ласка, створи {to_generate} нових речень, які є унікальними та природними для української мови. Уникай повторення будь-яких існуючих прикладів. 
Будь ласка, надай лише згенеровані речення у вигляді маркованого списку, без додаткових пояснень чи тексту.
Приклад списку:
- Речення 1
- Речення 2
- Речення 3

Згенеровані речення:
"""


def main():
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    out_data = defaultdict(dict)

    ## Load current data
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    # count how many meanings need generation
    n_meanings_to_generate = 0
    for lemma, meanings in data.items():
        for meaning_entry in meanings.values():
            sentences = meaning_entry.get("sentences", [])

            if len(sentences) < MIN_SENTENCES:
                n_meanings_to_generate += 1

    ## Generate sentences
    pbar = tqdm(total=n_meanings_to_generate, desc="Generating sentences")
    with open(OUTPUT_FILE, "a") as out_f:
        for lemma, meanings in data.items():
            for meaning_entry in meanings.values():
                meaning = meaning_entry.get("meaning", {})
                sentences = meaning_entry.get("sentences", [])

                to_generate = max(0, MIN_SENTENCES - len(sentences))
                if to_generate <= 0:  # nothing to generate
                    continue

                # format glosses and existing sentences to include in the prompt
                glosses_list = meaning.get("gloss", [])
                glosses = "\n".join([f"- {g}" for g in glosses_list])

                examples = meaning.get("examples", [])
                existing_sentences = "\n".join(f"- {s}" for s in examples)

                prompt = GENERATE_PROMPT.format(
                    lemma=lemma,
                    glosses=glosses,
                    existing_sentences=existing_sentences,
                    to_generate=to_generate,
                )
                print(f"Lemma: {lemma}, Glosses: {glosses_list}")

                ## Generate via API
                messages = [{"role": "user", "content": prompt}]
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                )

                output_text = response.choices[0].message.content
                print(output_text)

                # parse output sentences
                output_text = output_text.strip().split("\n")
                new_sentences = [
                    sent.strip("- ").strip()
                    for sent in output_text
                    if sent.startswith("-")
                ]

                out_data[lemma][glosses_list[0]] = {
                    "meaning": meaning,
                    "sentences": new_sentences,
                }

                pbar.update(1)

            # save every lemma
            out_f.write(json.dumps({lemma: out_data[lemma]}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
