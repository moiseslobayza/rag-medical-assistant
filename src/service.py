from __future__ import annotations

from typing import Protocol

from .retrieval import RetrievalResult


class Retriever(Protocol):
    def retrieve(self, question: str) -> RetrievalResult: ...


class Generator(Protocol):
    def generate(self, *, question: str, context: str) -> str: ...


class RAGService:
    """Coordinate validation, retrieval, context assembly and generation."""

    def __init__(self, retriever: Retriever, generator: Generator) -> None:
        self.retriever = retriever
        self.generator = generator

    def retrieve_context(self, question: str) -> RetrievalResult:
        if not question.strip():
            raise ValueError("The question cannot be empty.")
        return self.retriever.retrieve(question)

    def answer(self, question: str) -> str:
        retrieval = self.retrieve_context(question)
        context = "\n\n".join(retrieval.documents)
        return self.generator.generate(question=question, context=context)
