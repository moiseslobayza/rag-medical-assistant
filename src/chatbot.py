from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .config import DEFAULT_RAG_CONFIG
from .knowledge_base import (
    REQUIRED_COLUMNS as KNOWLEDGE_BASE_REQUIRED_COLUMNS,
    build_documents,
    build_ids,
    build_metadatas,
    load_knowledge_base,
    validate_knowledge_base,
)
from .retrieval import (
    ChromaRetriever,
    RetrievalResult,
    prepare_documents,
    prepare_query,
    uses_e5,
    validate_top_k,
)


class MedicalRAGChatbot:
    """RAG chatbot for administrative questions about a medical office."""

    REQUIRED_COLUMNS = set(KNOWLEDGE_BASE_REQUIRED_COLUMNS)

    def __init__(
        self,
        knowledge_base: pd.DataFrame,
        embedding_model: str = DEFAULT_RAG_CONFIG.embedding_model,
        llm_model: str = DEFAULT_RAG_CONFIG.llm_model,
        collection_name: str = DEFAULT_RAG_CONFIG.collection_name,
        top_k: int = DEFAULT_RAG_CONFIG.top_k,
    ) -> None:
        self.knowledge_base = self._validate_knowledge_base(knowledge_base)
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        self.collection_name = collection_name
        self.top_k = self._validate_top_k(top_k)

        self.documents = self._build_documents(self.knowledge_base)
        self.ids = build_ids(self.documents)
        self.metadatas = self._build_metadatas(self.knowledge_base)

        self.retriever = ChromaRetriever(
            documents=self.documents,
            ids=self.ids,
            metadatas=self.metadatas,
            embedding_model_name=self.embedding_model_name,
            collection_name=self.collection_name,
            top_k=self.top_k,
        )
        self.documents = self.retriever.documents
        self.ids = self.retriever.ids
        self.metadatas = self.retriever.metadatas
        self.embedding_model = self.retriever.embedding_model
        self.client = self.retriever.client
        self.collection = self.retriever.collection

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        self.llm = AutoModelForSeq2SeqLM.from_pretrained(self.llm_model_name).to(
            self.device
        )

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        **kwargs,
    ) -> "MedicalRAGChatbot":
        knowledge_base = load_knowledge_base(path)
        return cls(knowledge_base=knowledge_base, **kwargs)

    @classmethod
    def _validate_knowledge_base(cls, knowledge_base: pd.DataFrame) -> pd.DataFrame:
        return validate_knowledge_base(
            knowledge_base,
            required_columns=cls.REQUIRED_COLUMNS,
        )

    @staticmethod
    def _validate_top_k(top_k: int) -> int:
        return validate_top_k(top_k)

    @staticmethod
    def _build_documents(knowledge_base: pd.DataFrame) -> list[str]:
        return build_documents(knowledge_base)

    @staticmethod
    def _build_metadatas(knowledge_base: pd.DataFrame) -> list[dict[str, str]]:
        return build_metadatas(knowledge_base)

    def _uses_e5(self) -> bool:
        return uses_e5(self.embedding_model_name)

    def _prepare_documents(self, documents: Iterable[str]) -> list[str]:
        return prepare_documents(documents, self.embedding_model_name)

    def _prepare_query(self, question: str) -> str:
        return prepare_query(question, self.embedding_model_name)

    def retrieve_context(self, question: str) -> RetrievalResult:
        self.retriever.embedding_model_name = self.embedding_model_name
        self.retriever.embedding_model = self.embedding_model
        self.retriever.documents = self.documents
        self.retriever.top_k = self.top_k
        self.retriever.collection = self.collection
        return self.retriever.retrieve(question)

    def answer(self, question: str) -> str:
        retrieval = self.retrieve_context(question)
        context = "\n\n".join(retrieval.documents)

        prompt = f"""
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

        inputs = self.tokenizer(
            prompt,
            return_tensors=DEFAULT_RAG_CONFIG.tokenizer_return_tensors,
            truncation=DEFAULT_RAG_CONFIG.tokenizer_truncation,
            max_length=DEFAULT_RAG_CONFIG.tokenizer_max_length,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
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
