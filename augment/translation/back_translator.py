from abc import ABC, abstractmethod
from typing import List, Union

import torch
import sentencepiece as spm
from ctranslate2 import Translator as CT2Translator
from transformers import AutoModelForSeq2SeqLM, MarianTokenizer, AutoTokenizer, pipeline


class Translator(ABC):
    @abstractmethod
    def translate(self, texts: List[str], **kwargs) -> List[str]:
        pass


class HelsinkiCTranslateTranslator(Translator):
    def __init__(
        self, model_path: str, tokenizer_name: str, device="cuda", device_index=None
    ):
        self.model_name_or_path = model_path
        self.translator = CT2Translator(
            model_path, device=device, compute_type="float16", device_index=device_index
        )
        self.tokenizer = MarianTokenizer.from_pretrained(tokenizer_name)

    @torch.no_grad()
    def translate(self, texts: List[str], languages: tuple[str, str] | None, **kwargs):
        if "en-zle" in self.model_name_or_path:
            texts = [f">>ukr<< {sentence}" for sentence in texts]

        # Map Hugging Face arguments to CTranslate2 arguments
        ct2_kwargs = {
            "max_decoding_length": kwargs.get("max_length", 256),
            "sampling_temperature": kwargs.get("temperature", 1.0),
            "sampling_topk": kwargs.get("top_k", 50),
            "sampling_topp": kwargs.get("top_p", 1.0),
            "beam_size": kwargs.get("num_beams", 1),
            "num_hypotheses": kwargs.get("num_return_sequences", 1),
        }

        # CTranslate2 works with token lists, not raw strings
        source_tokens = [
            self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(t))
            for t in texts
        ]

        results = self.translator.translate_batch(
            source_tokens, max_batch_size=kwargs.get("batch_size", 128), **ct2_kwargs
        )

        output_texts = []
        for r in results:
            for hypo in r.hypotheses:
                text = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(hypo))
                output_texts.append(text)

        return output_texts


class NLLB200CTranslateTranslator(Translator):
    map_language_codes = {
        "en": "eng_Latn",
        "uk": "ukr_Cyrl",
        "pl": "pol_Latn",
    }

    def __init__(
        self, model_path: str, langs=List[str], device="cuda", device_index=None
    ):
        self.model_name_or_path = model_path
        self.translator = CT2Translator(
            model_path, device=device, compute_type="float16", device_index=device_index
        )

        self.tokenizers = {}
        for lang in langs:
            src_lang = self.map_language_codes[lang]
            self.tokenizers[src_lang] = AutoTokenizer.from_pretrained(
                "facebook/nllb-200-distilled-600M", src_lang=src_lang
            )

    @torch.no_grad()
    def translate(self, texts: List[str], languages: tuple[str, str], **kwargs):
        src_lang = self.map_language_codes[languages[0]]
        tgt_lang = self.map_language_codes[languages[1]]

        tokenizer = self.tokenizers[src_lang]

        sources = [tokenizer.convert_ids_to_tokens(tokenizer.encode(s)) for s in texts]

        target_prefix = [[tgt_lang]] * len(sources)

        # Map Hugging Face arguments to CTranslate2 arguments
        ct2_kwargs = {
            "max_decoding_length": kwargs.get("max_length", 512),
            "sampling_temperature": kwargs.get("temperature", 0.8),
            "sampling_topk": kwargs.get("top_k", 50),
            "sampling_topp": kwargs.get("top_p", 0.95),
            "beam_size": kwargs.get("num_beams", 1),
            "num_hypotheses": kwargs.get("num_return_sequences", 1),
        }

        results = self.translator.translate_batch(
            sources,
            max_batch_size=kwargs.get("batch_size", 128),
            target_prefix=target_prefix,
            **ct2_kwargs,
        )

        translations = []
        for result in results:
            for hypo in result.hypotheses:
                tokens = [t for t in hypo if t not in self.map_language_codes.values()]

                text = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens)).strip()
                translations.append(text)

        return translations


class NLLB200TransformersTranslator(Translator):
    map_language_codes = {
        "en": "eng_Latn",
        "uk": "ukr_Cyrl",
    }

    def __init__(
        self,
        model_path: str = None,
        device="cuda",
        device_index=None,
        src_lang="eng_Latn",
        tgt_lang="ukr_Cyrl",
    ):
        self.model_name_or_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            "facebook/nllb-200-distilled-600m"
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/nllb-200-distilled-600m",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

        self.model = self.model.half()
        self.model.to(device)

        self.translator = pipeline(
            "translation",
            model=self.model,
            tokenizer=self.tokenizer,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            max_length=512,
        )

    @torch.no_grad()
    def translate(self, texts: List[str], languages: tuple[str, str], **kwargs):
        output = self.translator(texts)
        translated_text = []

        for i in output:
            translated_text.append(i["translation_text"])

        return translated_text


class BackTranslator:
    def __init__(
        self,
        pivot_models: list[Translator],
        languages: list[tuple[str, str] | None],
        max_length: int = 512,
        top_k: int = 50,
        top_p: float = 0.95,
        temperature: float = 0.8,
        **kwargs,
    ):
        self.max_length = max_length
        self.gen_kwargs = dict(
            top_k=top_k, top_p=top_p, temperature=temperature, **kwargs
        )

        self.translators = pivot_models
        self.languages = languages

    @torch.no_grad()
    def augment(self, texts: Union[str, List[str]], n: int = 1):
        if isinstance(texts, str):
            texts = [texts]
            
        sentences_per_translation = int(n ** (1 / (len(self.translators))))
        total_sentences = sentences_per_translation ** len(self.translators)
        
        if total_sentences != n:
            print(
                f"Warning: The requested number of augmentations {n} cannot be evenly distributed across {len(self.translators)} translators. "
                f"Using {total_sentences} augmentations instead. "
                f"Try using n={total_sentences} or n={sentences_per_translation ** (len(self.translators) - 1)}."
            )

        # Subsequent models process those variations 1-to-1
        # This keeps the total count at (Input Batch Size * n)
        for translator, langs in zip(self.translators, self.languages):
            texts = translator.translate(
                texts,
                languages=langs,
                max_length=self.max_length,
                num_return_sequences=sentences_per_translation,
                **self.gen_kwargs,
            )

        return texts
