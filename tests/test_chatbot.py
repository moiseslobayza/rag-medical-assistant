from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src import MedicalRAGChatbot, RetrievalResult


VALID_ROWS = [
    {
        "pregunta": "¿Cómo puedo sacar un turno?",
        "respuesta": "Podés solicitarlo por WhatsApp.",
    },
    {
        "pregunta": "¿Cómo confirmo mi turno?",
        "respuesta": "Respondé el mensaje de confirmación.",
    },
]


def write_knowledge_base(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pregunta", "respuesta"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def build_chatbot(
    tmp_path: Path,
    *,
    rows: list[dict[str, str]] | None = None,
    top_k: int = 3,
) -> MedicalRAGChatbot:
    path = write_knowledge_base(
        tmp_path / "knowledge_base.csv",
        VALID_ROWS if rows is None else rows,
    )
    return MedicalRAGChatbot.from_csv(path, top_k=top_k)


def test_from_csv_loads_valid_knowledge_base(
    tmp_path: Path,
    pipeline_doubles: object,
) -> None:
    chatbot = build_chatbot(tmp_path)

    assert chatbot.knowledge_base.to_dict("records") == VALID_ROWS


def test_from_csv_rejects_missing_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "missing_column.csv"
    path.write_text('pregunta\n"¿Cómo solicito un turno?"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required columns: respuesta"):
        MedicalRAGChatbot.from_csv(path)


@pytest.mark.parametrize(
    "csv_contents",
    [
        "pregunta,respuesta\n",
        'pregunta,respuesta\n,"Respuesta sin pregunta"\n"Pregunta sin respuesta",\n',
        'pregunta,respuesta\n"   ","  "\n',
    ],
    ids=["no-rows", "missing-values", "whitespace-only"],
)
def test_from_csv_rejects_knowledge_base_without_valid_rows(
    tmp_path: Path,
    csv_contents: str,
) -> None:
    path = tmp_path / "invalid_rows.csv"
    path.write_text(csv_contents, encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="The knowledge base contains no valid records",
    ):
        MedicalRAGChatbot.from_csv(path)


def test_documents_and_e5_passage_prefix_are_indexed(
    tmp_path: Path,
    pipeline_doubles: object,
) -> None:
    build_chatbot(tmp_path)

    expected_documents = [
        "Pregunta frecuente: ¿Cómo puedo sacar un turno?\n"
        "Respuesta: Podés solicitarlo por WhatsApp.",
        "Pregunta frecuente: ¿Cómo confirmo mi turno?\n"
        "Respuesta: Respondé el mensaje de confirmación.",
    ]
    assert pipeline_doubles.indexed["documents"] == expected_documents
    assert pipeline_doubles.indexed["ids"] == ["doc_0", "doc_1"]
    first_embedding_call = pipeline_doubles.embedding.encode.call_args_list[0]
    assert first_embedding_call.args[0] == [
        f"passage: {document}" for document in expected_documents
    ]


def test_e5_query_prefix_is_sent_to_embedder(
    tmp_path: Path,
    pipeline_doubles: object,
) -> None:
    chatbot = build_chatbot(tmp_path, top_k=1)

    retrieval = chatbot.retrieve_context("  ¿Cómo reservo?  ")

    last_embedding_call = pipeline_doubles.embedding.encode.call_args_list[-1]
    assert last_embedding_call.args[0] == ["query: ¿Cómo reservo?"]
    assert retrieval == RetrievalResult(
        documents=[pipeline_doubles.indexed["documents"][0]],
        metadatas=[pipeline_doubles.indexed["metadatas"][0]],
        distances=[0.1],
    )


@pytest.mark.parametrize("top_k", [0, -1])
def test_rejects_non_positive_top_k(
    tmp_path: Path,
    top_k: int,
) -> None:
    path = write_knowledge_base(tmp_path / "knowledge_base.csv", VALID_ROWS)

    with pytest.raises(
        ValueError,
        match="top_k must be greater than or equal to 1",
    ):
        MedicalRAGChatbot.from_csv(path, top_k=top_k)


def test_answer_rejects_blank_question(
    tmp_path: Path,
    pipeline_doubles: object,
) -> None:
    chatbot = build_chatbot(tmp_path)

    with pytest.raises(ValueError, match="The question cannot be empty"):
        chatbot.answer("   ")

    pipeline_doubles.collection.query.assert_not_called()
    pipeline_doubles.language_model.generate.assert_not_called()


def test_pipeline_uses_retrieval_context_and_generation_doubles(
    tmp_path: Path,
    pipeline_doubles: object,
) -> None:
    chatbot = build_chatbot(tmp_path, top_k=5)

    question = "¿Cómo confirmo mi turno?"
    answer = chatbot.answer(question)

    assert answer == "Respuesta generada de prueba."
    assert pipeline_doubles.collection.query.call_args.kwargs["n_results"] == 2
    prompt = pipeline_doubles.tokenizer.call_args.args[0]
    assert question in prompt
    assert all(
        document in prompt for document in pipeline_doubles.indexed["documents"]
    )
    pipeline_doubles.language_model.generate.assert_called_once()
    pipeline_doubles.torch.inference_mode.assert_called_once_with()
