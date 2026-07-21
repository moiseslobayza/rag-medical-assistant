from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pandas as pd
import pytest

import src.retrieval as retrieval_module
from src.evaluation import (
    EmbeddingRetrieverEvaluator,
    EvaluationCase,
    compare_embedding_models,
)
from src.retrieval import ChromaRetriever


E5_MODEL = "intfloat/multilingual-e5-small"
QUESTIONS = [
    "Pregunta uno",
    "Pregunta dos",
    "Pregunta tres",
    "Pregunta cuatro",
    "Pregunta cinco",
]

SUMMARY_COLUMNS = [
    "model",
    "top_k",
    "hit_rate",
    "hits",
    "total",
    "mean_expected_rank",
]
DETAIL_COLUMNS = [
    "model",
    "query",
    "expected_question",
    "top_k",
    "hit",
    "expected_rank",
    "top_result",
    "top_distance",
]


class ListResult:
    def __init__(self, values: list[list[float]]) -> None:
        self.values = values

    def tolist(self) -> list[list[float]]:
        return self.values


def make_knowledge_base(questions: list[str] | None = None) -> pd.DataFrame:
    selected_questions = QUESTIONS if questions is None else questions
    return pd.DataFrame(
        {
            "pregunta": selected_questions,
            "respuesta": [
                f"Respuesta {index}"
                for index in range(1, len(selected_questions) + 1)
            ],
        }
    )


def install_evaluation_doubles(
    monkeypatch: pytest.MonkeyPatch,
    query_rankings: list[tuple[list[str], list[float]]] | None = None,
) -> SimpleNamespace:
    embedding = Mock(name="evaluation_embedding_model")
    embedding.encode.side_effect = lambda texts, **kwargs: ListResult(
        [[float(index + 1), 0.0] for index in range(len(texts))]
    )
    embedding_factory = Mock(
        name="sentence_transformer_factory",
        return_value=embedding,
    )

    rankings = iter(query_rankings or [])
    collection = Mock(name="evaluation_chroma_collection")

    def query_collection(
        *,
        query_embeddings: list[list[float]],
        n_results: int,
    ) -> dict[str, list[list[object]]]:
        try:
            questions, distances = next(rankings)
        except StopIteration as error:
            raise AssertionError("Unexpected Chroma query in evaluation test.") from error

        return {
            "documents": [
                [f"Documento recuperado: {question}" for question in questions[:n_results]]
            ],
            "metadatas": [
                [
                    {"pregunta_original": question}
                    for question in questions[:n_results]
                ]
            ],
            "distances": [distances[:n_results]],
        }

    collection.query.side_effect = query_collection
    client = Mock(name="evaluation_chroma_client")
    client.create_collection.return_value = collection
    chroma_factory = Mock(name="chroma_factory", return_value=client)

    monkeypatch.setattr(
        retrieval_module,
        "_load_embedding_model",
        embedding_factory,
    )
    monkeypatch.setattr(
        retrieval_module,
        "_create_chroma_client",
        chroma_factory,
    )

    return SimpleNamespace(
        embedding=embedding,
        embedding_factory=embedding_factory,
        collection=collection,
        client=client,
        chroma_factory=chroma_factory,
    )


def build_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    *,
    questions: list[str] | None = None,
    embedding_model: str = E5_MODEL,
    query_rankings: list[tuple[list[str], list[float]]] | None = None,
) -> SimpleNamespace:
    selected_questions = QUESTIONS if questions is None else questions
    rankings = query_rankings
    if rankings is None:
        rankings = [
            (
                selected_questions,
                [0.1 * (index + 1) for index in range(len(selected_questions))],
            )
        ]

    doubles = install_evaluation_doubles(monkeypatch, rankings)
    doubles.evaluator = EmbeddingRetrieverEvaluator(
        knowledge_base=make_knowledge_base(selected_questions),
        embedding_model=embedding_model,
    )
    return doubles


def test_evaluate_returns_contract_columns_metrics_and_query_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rankings = [
        (QUESTIONS, [0.01, 0.02, 0.03, 0.04, 0.05]),
        (
            [QUESTIONS[0], QUESTIONS[2], QUESTIONS[1], *QUESTIONS[3:]],
            [0.11, 0.12, 0.13, 0.14, 0.15],
        ),
        (
            [QUESTIONS[0], QUESTIONS[1], QUESTIONS[3], QUESTIONS[4], QUESTIONS[2]],
            [0.21, 0.22, 0.23, 0.24, 0.25],
        ),
    ]
    doubles = build_evaluator(monkeypatch, query_rankings=rankings)
    retrieve = Mock(wraps=doubles.evaluator.retriever.retrieve)
    monkeypatch.setattr(doubles.evaluator.retriever, "retrieve", retrieve)
    cases = [
        EvaluationCase("Consulta uno", QUESTIONS[0]),
        EvaluationCase("Consulta dos", QUESTIONS[1]),
        EvaluationCase("Consulta tres", QUESTIONS[2]),
    ]

    summary, details = doubles.evaluator.evaluate(
        cases,
        top_k_values=(5, 1, 3, 3),
    )

    assert list(summary.columns) == SUMMARY_COLUMNS
    assert list(details.columns) == DETAIL_COLUMNS
    assert details["query"].tolist() == [
        case.query for case in cases for _ in range(3)
    ]
    assert details["top_k"].tolist() == [1, 3, 5] * len(cases)
    assert details["hit"].tolist() == [
        True,
        True,
        True,
        False,
        True,
        True,
        False,
        False,
        True,
    ]
    assert details["expected_rank"].tolist() == [1, 1, 1, 3, 3, 3, 5, 5, 5]
    assert retrieve.call_args_list == [call(case.query) for case in cases]
    assert doubles.evaluator.retriever.top_k == 5

    metrics = summary.set_index("top_k")
    assert summary["top_k"].tolist() == [5, 3, 1]
    assert metrics.loc[1, "hit_rate"] == pytest.approx(1 / 3)
    assert metrics.loc[3, "hit_rate"] == pytest.approx(2 / 3)
    assert metrics.loc[5, "hit_rate"] == pytest.approx(1.0)
    assert metrics["hits"].to_dict() == {5: 3, 3: 2, 1: 1}
    assert metrics["total"].to_dict() == {5: 3, 3: 3, 1: 3}
    assert metrics["mean_expected_rank"].to_dict() == {
        5: 3.0,
        3: 3.0,
        1: 3.0,
    }


def test_details_identify_an_expected_document_not_retrieved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doubles = build_evaluator(monkeypatch)

    summary, details = doubles.evaluator.evaluate(
        [EvaluationCase("Consulta sin resultado", "Pregunta inexistente")]
    )

    assert details["hit"].tolist() == [False, False, False]
    assert details["expected_rank"].isna().all()
    assert details["top_result"].tolist() == [QUESTIONS[0]] * 3
    assert details["top_distance"].tolist() == [0.1] * 3
    assert summary["hit_rate"].tolist() == [0.0, 0.0, 0.0]
    assert summary["hits"].tolist() == [0, 0, 0]
    assert summary["total"].tolist() == [1, 1, 1]
    assert summary["mean_expected_rank"].isna().all()


@pytest.mark.parametrize(
    ("embedding_model", "prepared_documents", "prepared_query"),
    [
        (
            "INTFLOAT/MULTILINGUAL-E5-SMALL",
            [
                "passage: Pregunta frecuente: Pregunta uno\nRespuesta: Respuesta 1",
                "passage: Pregunta frecuente: Pregunta dos\nRespuesta: Respuesta 2",
            ],
            "query: Consulta",
        ),
        (
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            [
                "Pregunta frecuente: Pregunta uno\nRespuesta: Respuesta 1",
                "Pregunta frecuente: Pregunta dos\nRespuesta: Respuesta 2",
            ],
            "Consulta",
        ),
    ],
)
def test_evaluation_embedding_prefixes_normalization_and_chroma_payload(
    monkeypatch: pytest.MonkeyPatch,
    embedding_model: str,
    prepared_documents: list[str],
    prepared_query: str,
) -> None:
    questions = QUESTIONS[:2]
    doubles = build_evaluator(
        monkeypatch,
        questions=questions,
        embedding_model=embedding_model,
    )

    _, details = doubles.evaluator.evaluate(
        [EvaluationCase("  Consulta  ", questions[0])],
        top_k_values=(1,),
    )

    assert doubles.embedding.encode.call_args_list == [
        call(
            prepared_documents,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        call(
            [prepared_query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
    ]
    raw_documents = [
        "Pregunta frecuente: Pregunta uno\nRespuesta: Respuesta 1",
        "Pregunta frecuente: Pregunta dos\nRespuesta: Respuesta 2",
    ]
    doubles.collection.add.assert_called_once_with(
        ids=["doc_0", "doc_1"],
        documents=raw_documents,
        embeddings=[[1.0, 0.0], [2.0, 0.0]],
        metadatas=[
            {"pregunta_original": "Pregunta uno"},
            {"pregunta_original": "Pregunta dos"},
        ],
    )
    create_call = doubles.client.create_collection.call_args
    assert create_call.kwargs["name"].startswith("embedding_eval_")
    assert create_call.kwargs["metadata"] == {"hnsw:space": "cosine"}
    assert details.loc[0, "query"] == "  Consulta  "


def test_evaluate_caps_query_top_k_and_preserves_current_small_corpus_totals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    questions = QUESTIONS[:2]
    doubles = build_evaluator(monkeypatch, questions=questions)

    summary, details = doubles.evaluator.evaluate(
        [EvaluationCase("Consulta", questions[1])],
        top_k_values=(1, 3, 5),
    )

    doubles.collection.query.assert_called_once_with(
        query_embeddings=[[1.0, 0.0]],
        n_results=2,
    )
    assert details["top_k"].tolist() == [1, 2, 2]
    assert details["hit"].tolist() == [False, True, True]

    metrics = summary.set_index("top_k")
    assert metrics.loc[1, "hits"] == 0
    assert metrics.loc[1, "total"] == 1
    assert metrics.loc[1, "mean_expected_rank"] == 2.0
    assert metrics.loc[2, "hits"] == 2
    assert metrics.loc[2, "total"] == 2
    assert metrics.loc[2, "hit_rate"] == 1.0


@pytest.mark.parametrize(
    ("cases", "top_k_values", "message"),
    [
        ([], (1, 3, 5), "At least one evaluation case is required."),
        (
            [EvaluationCase("Consulta", QUESTIONS[0])],
            (),
            "top_k values must be greater than or equal to 1.",
        ),
        (
            [EvaluationCase("Consulta", QUESTIONS[0])],
            (0, 1),
            "top_k values must be greater than or equal to 1.",
        ),
        (
            [EvaluationCase("Consulta", QUESTIONS[0])],
            (-1, 3),
            "top_k values must be greater than or equal to 1.",
        ),
    ],
    ids=["no-cases", "no-top-k", "zero-top-k", "negative-top-k"],
)
def test_evaluate_rejects_invalid_cases_or_top_k_before_retrieval(
    monkeypatch: pytest.MonkeyPatch,
    cases: list[EvaluationCase],
    top_k_values: tuple[int, ...],
    message: str,
) -> None:
    doubles = build_evaluator(monkeypatch)
    doubles.embedding.reset_mock()

    with pytest.raises(ValueError) as error:
        doubles.evaluator.evaluate(cases, top_k_values)

    assert str(error.value) == message
    doubles.embedding.encode.assert_not_called()
    doubles.collection.query.assert_not_called()


def test_evaluate_rejects_blank_query_before_embedding_or_chroma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doubles = build_evaluator(monkeypatch)
    doubles.embedding.reset_mock()

    with pytest.raises(ValueError) as error:
        doubles.evaluator.evaluate(
            [EvaluationCase("   ", QUESTIONS[0])],
            top_k_values=(1,),
        )

    assert str(error.value) == "Evaluation queries cannot be empty."
    doubles.embedding.encode.assert_not_called()
    doubles.collection.query.assert_not_called()


@pytest.mark.parametrize(
    ("knowledge_base", "message"),
    [
        (
            pd.DataFrame({"pregunta": ["Pregunta"]}),
            "Missing required columns: respuesta",
        ),
        (
            pd.DataFrame({"pregunta": ["   "], "respuesta": ["  "]}),
            "The knowledge base contains no valid records.",
        ),
    ],
    ids=["missing-column", "no-valid-records"],
)
def test_evaluator_rejects_invalid_knowledge_base_before_integrations(
    monkeypatch: pytest.MonkeyPatch,
    knowledge_base: pd.DataFrame,
    message: str,
) -> None:
    doubles = install_evaluation_doubles(monkeypatch)

    with pytest.raises(ValueError) as error:
        EmbeddingRetrieverEvaluator(knowledge_base, E5_MODEL)

    assert str(error.value) == message
    doubles.embedding_factory.assert_not_called()
    doubles.chroma_factory.assert_not_called()


def test_from_csv_cleans_rows_and_builds_current_evaluation_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "knowledge_base.csv"
    path.write_text(
        "pregunta,respuesta\n"
        '"  Pregunta valida  "," Respuesta valida "\n'
        ',"Respuesta sin pregunta"\n'
        '"Pregunta sin respuesta",\n',
        encoding="utf-8",
    )
    doubles = install_evaluation_doubles(monkeypatch)

    evaluator = EmbeddingRetrieverEvaluator.from_csv(path, E5_MODEL)

    assert isinstance(evaluator.retriever, ChromaRetriever)
    assert evaluator.documents is evaluator.retriever.documents
    assert evaluator.ids is evaluator.retriever.ids
    assert evaluator.metadatas is evaluator.retriever.metadatas
    assert evaluator.embedding_model is evaluator.retriever.embedding_model
    assert evaluator.client is evaluator.retriever.client
    assert evaluator.collection is evaluator.retriever.collection
    doubles.embedding_factory.assert_called_once_with(E5_MODEL)
    doubles.chroma_factory.assert_called_once_with()
    assert evaluator.knowledge_base.to_dict("records") == [
        {"pregunta": "Pregunta valida", "respuesta": "Respuesta valida"}
    ]
    assert evaluator.documents == [
        "Pregunta frecuente: Pregunta valida\nRespuesta: Respuesta valida"
    ]
    assert evaluator.ids == ["doc_0"]
    assert evaluator.metadatas == [{"pregunta_original": "Pregunta valida"}]


def test_from_csv_rejects_missing_file_before_loading_integrations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.csv"
    doubles = install_evaluation_doubles(monkeypatch)

    with pytest.raises(FileNotFoundError) as error:
        EmbeddingRetrieverEvaluator.from_csv(missing_path, E5_MODEL)

    assert str(error.value) == f"Knowledge base not found: {missing_path}"
    doubles.embedding_factory.assert_not_called()
    doubles.chroma_factory.assert_not_called()


def test_chroma_failure_propagates_instead_of_becoming_a_retrieval_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doubles = build_evaluator(monkeypatch)
    doubles.embedding.reset_mock()
    doubles.collection.query.side_effect = RuntimeError("Chroma unavailable")

    with pytest.raises(RuntimeError, match="Chroma unavailable"):
        doubles.evaluator.evaluate(
            [EvaluationCase("Consulta", QUESTIONS[0])],
            top_k_values=(1,),
        )

    doubles.embedding.encode.assert_called_once_with(
        ["query: Consulta"],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    doubles.collection.query.assert_called_once_with(
        query_embeddings=[[1.0, 0.0]],
        n_results=1,
    )


def test_compare_embedding_models_preserves_requested_model_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "knowledge_base.csv"
    models = ["model-z", "model-a"]
    cases = [EvaluationCase("Consulta", QUESTIONS[0])]
    evaluators: list[SimpleNamespace] = []

    def build_fake_evaluator(*, path: Path, embedding_model: str) -> SimpleNamespace:
        evaluator = SimpleNamespace()
        evaluator.evaluate = Mock(
            return_value=(
                pd.DataFrame(
                    [
                        {
                            "model": embedding_model,
                            "top_k": 1,
                            "hit_rate": 1.0,
                            "hits": 1,
                            "total": 1,
                            "mean_expected_rank": 1.0,
                        }
                    ],
                    columns=SUMMARY_COLUMNS,
                ),
                pd.DataFrame(
                    [
                        {
                            "model": embedding_model,
                            "query": "Consulta",
                            "expected_question": QUESTIONS[0],
                            "top_k": 1,
                            "hit": True,
                            "expected_rank": 1,
                            "top_result": QUESTIONS[0],
                            "top_distance": 0.1,
                        }
                    ],
                    columns=DETAIL_COLUMNS,
                ),
            )
        )
        evaluators.append(evaluator)
        return evaluator

    evaluator_factory = Mock(side_effect=build_fake_evaluator)
    monkeypatch.setattr(
        EmbeddingRetrieverEvaluator,
        "from_csv",
        evaluator_factory,
    )

    summary, details = compare_embedding_models(
        knowledge_base_path=path,
        embedding_models=models,
        cases=cases,
        top_k_values=(1,),
    )

    assert summary["model"].tolist() == models
    assert details["model"].tolist() == models
    assert evaluator_factory.call_args_list == [
        call(path=path, embedding_model=model) for model in models
    ]
    for evaluator in evaluators:
        evaluator.evaluate.assert_called_once_with(
            cases=cases,
            top_k_values=(1,),
        )


def test_compare_embedding_models_rejects_empty_model_list() -> None:
    with pytest.raises(ValueError) as error:
        compare_embedding_models(
            knowledge_base_path="unused.csv",
            embedding_models=[],
            cases=[EvaluationCase("Consulta", QUESTIONS[0])],
        )

    assert str(error.value) == "At least one embedding model is required."
