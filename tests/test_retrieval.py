from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

import src
import src.chatbot as chatbot_module
from src.retrieval import ChromaRetriever, RetrievalResult


DOCUMENTS = ["Documento uno", "Documento dos"]
IDS = ["doc_0", "doc_1"]
METADATAS = [
    {
        "pregunta_original": "Pregunta uno",
        "respuesta_original": "Respuesta uno",
    },
    {
        "pregunta_original": "Pregunta dos",
        "respuesta_original": "Respuesta dos",
    },
]
DOCUMENT_EMBEDDINGS = [[1.0, 0.0], [2.0, 0.0]]
QUERY_EMBEDDING = [[9.0, 0.0]]
QUERY_DOCUMENTS = ["Documento dos", "Documento uno"]
QUERY_METADATAS = [METADATAS[1], METADATAS[0]]
QUERY_DISTANCES = [0.12, 0.34]


class ListResult:
    def __init__(self, values: list[list[float]]) -> None:
        self.values = values

    def tolist(self) -> list[list[float]]:
        return self.values


def build_retriever(
    *,
    embedding_model_name: str = "intfloat/multilingual-e5-small",
    top_k: int = 3,
) -> SimpleNamespace:
    embedding = Mock(name="embedding_model")

    def encode(texts: list[str], **kwargs: object) -> ListResult:
        if len(texts) == len(DOCUMENTS):
            return ListResult(DOCUMENT_EMBEDDINGS)
        return ListResult(QUERY_EMBEDDING)

    embedding.encode.side_effect = encode
    embedding_model_factory = Mock(return_value=embedding)

    collection = Mock(name="chroma_collection")

    def query_collection(
        *,
        query_embeddings: list[list[float]],
        n_results: int,
    ) -> dict[str, list[list[object]]]:
        return {
            "documents": [QUERY_DOCUMENTS[:n_results]],
            "metadatas": [QUERY_METADATAS[:n_results]],
            "distances": [QUERY_DISTANCES[:n_results]],
        }

    collection.query.side_effect = query_collection
    client = Mock(name="chroma_client")
    client.create_collection.return_value = collection
    chroma_client_factory = Mock(return_value=client)

    retriever = ChromaRetriever(
        documents=DOCUMENTS,
        ids=IDS,
        metadatas=METADATAS,
        embedding_model_name=embedding_model_name,
        collection_name="medical_office_knowledge",
        top_k=top_k,
        embedding_model_factory=embedding_model_factory,
        chroma_client_factory=chroma_client_factory,
    )
    return SimpleNamespace(
        retriever=retriever,
        embedding=embedding,
        embedding_model_factory=embedding_model_factory,
        collection=collection,
        client=client,
        chroma_client_factory=chroma_client_factory,
    )


@pytest.mark.parametrize(
    ("embedding_model_name", "prepared_documents", "prepared_query"),
    [
        (
            "intfloat/multilingual-e5-small",
            ["passage: Documento uno", "passage: Documento dos"],
            "query: Consulta",
        ),
        (
            "INTFLOAT/MULTILINGUAL-E5-SMALL",
            ["passage: Documento uno", "passage: Documento dos"],
            "query: Consulta",
        ),
        (
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            DOCUMENTS,
            "Consulta",
        ),
    ],
)
def test_embedding_prefixes_are_model_specific_and_normalized(
    embedding_model_name: str,
    prepared_documents: list[str],
    prepared_query: str,
) -> None:
    doubles = build_retriever(embedding_model_name=embedding_model_name)

    doubles.retriever.retrieve("  Consulta  ")

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


def test_chroma_index_preserves_payload_collection_and_cosine_distance() -> None:
    doubles = build_retriever()

    doubles.embedding_model_factory.assert_called_once_with(
        "intfloat/multilingual-e5-small"
    )
    doubles.chroma_client_factory.assert_called_once_with()
    doubles.client.create_collection.assert_called_once_with(
        name="medical_office_knowledge",
        metadata={"hnsw:space": "cosine"},
    )
    doubles.collection.add.assert_called_once_with(
        ids=IDS,
        documents=DOCUMENTS,
        embeddings=DOCUMENT_EMBEDDINGS,
        metadatas=METADATAS,
    )


@pytest.mark.parametrize(
    ("top_k", "expected_result_count"),
    [(1, 1), (5, 2)],
)
def test_retrieve_caps_top_k_and_converts_chroma_results_without_reordering(
    top_k: int,
    expected_result_count: int,
) -> None:
    doubles = build_retriever(top_k=top_k)

    result = doubles.retriever.retrieve("Consulta")

    doubles.collection.query.assert_called_once_with(
        query_embeddings=QUERY_EMBEDDING,
        n_results=expected_result_count,
    )
    assert result == RetrievalResult(
        documents=QUERY_DOCUMENTS[:expected_result_count],
        metadatas=QUERY_METADATAS[:expected_result_count],
        distances=QUERY_DISTANCES[:expected_result_count],
    )


def test_blank_query_fails_before_embedding_or_chroma_query() -> None:
    doubles = build_retriever()
    doubles.embedding.reset_mock()

    with pytest.raises(ValueError, match="The question cannot be empty"):
        doubles.retriever.retrieve("   ")

    doubles.embedding.encode.assert_not_called()
    doubles.collection.query.assert_not_called()


def test_retrieval_result_remains_available_from_public_locations() -> None:
    assert src.RetrievalResult is RetrievalResult
    assert chatbot_module.RetrievalResult is RetrievalResult


def test_import_src_does_not_import_chroma_or_sentence_transformers() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = """
import importlib.abc
import sys
from types import ModuleType

for module_name in ("pandas", "torch"):
    sys.modules[module_name] = ModuleType(module_name)

transformers = ModuleType("transformers")
transformers.AutoModelForSeq2SeqLM = object()
transformers.AutoTokenizer = object()
sys.modules["transformers"] = transformers

blocked = {"chromadb", "sentence_transformers"}

class BlockedImportFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] in blocked:
            raise AssertionError(f"unexpected integration import: {fullname}")
        return None

sys.meta_path.insert(0, BlockedImportFinder())

import src
import src.chatbot as chatbot_module
import src.retrieval as retrieval_module

assert src.MedicalRAGChatbot is chatbot_module.MedicalRAGChatbot
assert src.RetrievalResult is chatbot_module.RetrievalResult
assert src.RetrievalResult is retrieval_module.RetrievalResult
assert not blocked.intersection(sys.modules)
"""
    environment = {
        **os.environ,
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }

    completed = subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
