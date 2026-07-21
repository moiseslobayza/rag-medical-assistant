from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .config import DEFAULT_RAG_CONFIG


TorchLoader = Callable[[], Any]
TokenizerFactory = Callable[[str], Any]
LanguageModelFactory = Callable[[str], Any]


def _load_torch() -> Any:
    import torch

    return torch


def _load_tokenizer(model_name: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def _load_language_model(model_name: str) -> Any:
    from transformers import AutoModelForSeq2SeqLM

    return AutoModelForSeq2SeqLM.from_pretrained(model_name)


def build_prompt(question: str, context: str) -> str:
    return f"""
Use the context to answer the question.
Answer only in Spanish.
Do not invent information.
If the context does not contain enough information, say that the consultorio should be contacted directly.

Context:
{context}

Question:
{question}

Answer in Spanish:
""".strip()


class FlanT5Generator:
    """Generate Spanish answers with the current FLAN-T5 contract."""

    def __init__(
        self,
        llm_model_name: str = DEFAULT_RAG_CONFIG.llm_model,
        *,
        torch_loader: TorchLoader | None = None,
        tokenizer_factory: TokenizerFactory | None = None,
        language_model_factory: LanguageModelFactory | None = None,
    ) -> None:
        self.llm_model_name = llm_model_name

        load_torch = torch_loader if torch_loader is not None else _load_torch
        create_tokenizer = (
            tokenizer_factory if tokenizer_factory is not None else _load_tokenizer
        )
        create_language_model = (
            language_model_factory
            if language_model_factory is not None
            else _load_language_model
        )

        self.torch = load_torch()
        self.device = "cuda" if self.torch.cuda.is_available() else "cpu"
        self.tokenizer = create_tokenizer(self.llm_model_name)
        self.llm = create_language_model(self.llm_model_name).to(self.device)

    def generate(self, *, question: str, context: str) -> str:
        prompt = build_prompt(question=question, context=context)
        inputs = self.tokenizer(
            prompt,
            return_tensors=DEFAULT_RAG_CONFIG.tokenizer_return_tensors,
            truncation=DEFAULT_RAG_CONFIG.tokenizer_truncation,
            max_length=DEFAULT_RAG_CONFIG.tokenizer_max_length,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with self.torch.inference_mode():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=DEFAULT_RAG_CONFIG.max_new_tokens,
                do_sample=DEFAULT_RAG_CONFIG.do_sample,
                num_beams=DEFAULT_RAG_CONFIG.num_beams,
            )

        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=DEFAULT_RAG_CONFIG.skip_special_tokens,
        )
        return response.strip()
