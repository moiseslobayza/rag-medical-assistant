from __future__ import annotations

from unittest.mock import Mock

import pytest

from src.retrieval import RetrievalResult
from src.service import RAGService


def test_service_coordinates_retrieval_context_and_generation_in_order() -> None:
    events: list[str] = []
    question = "  Consulta administrativa  "
    retrieval = RetrievalResult(
        documents=["Documento segundo", "Documento primero"],
        metadatas=[{"orden": "segundo"}, {"orden": "primero"}],
        distances=[0.1, 0.2],
    )
    retriever = Mock(name="retriever")
    generator = Mock(name="generator")

    def retrieve(received_question: str) -> RetrievalResult:
        events.append("retrieve")
        return retrieval

    def generate(*, question: str, context: str) -> str:
        events.append("generate")
        return "Respuesta generada."

    retriever.retrieve.side_effect = retrieve
    generator.generate.side_effect = generate
    service = RAGService(retriever=retriever, generator=generator)

    response = service.answer(question)

    assert events == ["retrieve", "generate"]
    retriever.retrieve.assert_called_once_with(question)
    generator.generate.assert_called_once_with(
        question=question,
        context="Documento segundo\n\nDocumento primero",
    )
    assert response == "Respuesta generada."


def test_service_rejects_blank_question_before_retrieval_or_generation() -> None:
    retriever = Mock(name="retriever")
    generator = Mock(name="generator")
    service = RAGService(retriever=retriever, generator=generator)

    with pytest.raises(ValueError, match="The question cannot be empty"):
        service.answer("   ")

    retriever.retrieve.assert_not_called()
    generator.generate.assert_not_called()
