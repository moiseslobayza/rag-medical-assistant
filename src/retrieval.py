from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .config import DEFAULT_RAG_CONFIG


class EmbeddingOutput(Protocol):
    def tolist(self) -> list[list[float]]: ...


class EmbeddingModel(Protocol):
    def encode(
        self,
        texts: Sequence[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> EmbeddingOutput: ...


class ChromaCollection(Protocol):
    def add(
        self,
        *,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
    ) -> None: ...

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
    ) -> dict[str, list[list[Any]]]: ...


class ChromaClient(Protocol):
    def create_collection(
        self,
        *,
        name: str,
        metadata: dict[str, str],
    ) -> ChromaCollection: ...


EmbeddingModelFactory = Callable[[str], EmbeddingModel]
ChromaClientFactory = Callable[[], ChromaClient]


@dataclass(frozen=True)
class RetrievalResult:
    documents: list[str]
    metadatas: list[dict[str, str]]
    distances: list[float]


def _load_embedding_model(model_name: str) -> EmbeddingModel:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _create_chroma_client() -> ChromaClient:
    import chromadb

    return chromadb.EphemeralClient()


def uses_e5(embedding_model_name: str) -> bool:
    return "e5" in embedding_model_name.lower()


def prepare_documents(
    documents: Iterable[str],
    embedding_model_name: str,
) -> list[str]:
    if uses_e5(embedding_model_name):
        return [f"passage: {document}" for document in documents]
    return list(documents)


def prepare_query(question: str, embedding_model_name: str) -> str:
    question = question.strip()
    if not question:
        raise ValueError("The question cannot be empty.")

    if uses_e5(embedding_model_name):
        return f"query: {question}"
    return question


def validate_top_k(top_k: int) -> int:
    if top_k < 1:
        raise ValueError("top_k must be greater than or equal to 1.")
    return top_k


class ChromaRetriever:
    """Embed, index and retrieve knowledge-base documents with Chroma."""

    def __init__(
        self,
        documents: Sequence[str],
        ids: Sequence[str],
        metadatas: Sequence[dict[str, str]],
        embedding_model_name: str = DEFAULT_RAG_CONFIG.embedding_model,
        collection_name: str = DEFAULT_RAG_CONFIG.collection_name,
        top_k: int = DEFAULT_RAG_CONFIG.top_k,
        *,
        embedding_model_factory: EmbeddingModelFactory | None = None,
        chroma_client_factory: ChromaClientFactory | None = None,
    ) -> None:
        self.documents = list(documents)
        self.ids = list(ids)
        self.metadatas = list(metadatas)
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        self.top_k = validate_top_k(top_k)

        create_embedding_model = embedding_model_factory or _load_embedding_model
        self.embedding_model = create_embedding_model(self.embedding_model_name)
        document_embeddings = self._embed(
            prepare_documents(self.documents, self.embedding_model_name)
        )

        create_client = chroma_client_factory or _create_chroma_client
        self.client = create_client()
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": DEFAULT_RAG_CONFIG.chroma_distance_metric},
        )
        self.collection.add(
            ids=self.ids,
            documents=self.documents,
            embeddings=document_embeddings,
            metadatas=self.metadatas,
        )

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embedding_model.encode(
            texts,
            convert_to_numpy=DEFAULT_RAG_CONFIG.embedding_convert_to_numpy,
            normalize_embeddings=DEFAULT_RAG_CONFIG.normalize_embeddings,
        ).tolist()

    def retrieve(self, question: str) -> RetrievalResult:
        prepared_question = prepare_query(question, self.embedding_model_name)
        question_embedding = self._embed([prepared_question])

        result_count = min(self.top_k, len(self.documents))
        results = self.collection.query(
            query_embeddings=question_embedding,
            n_results=result_count,
        )

        return RetrievalResult(
            documents=results["documents"][0],
            metadatas=results["metadatas"][0],
            distances=results["distances"][0],
        )
