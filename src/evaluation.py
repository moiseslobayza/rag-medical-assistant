from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from uuid import uuid4

import pandas as pd

from .knowledge_base import (
    REQUIRED_COLUMNS as KNOWLEDGE_BASE_REQUIRED_COLUMNS,
    build_documents,
    build_ids,
    build_metadatas,
    load_knowledge_base,
    validate_knowledge_base,
)
from .retrieval import ChromaRetriever, RetrievalResult


@dataclass(frozen=True)
class EvaluationCase:
    query: str
    expected_question: str


class EmbeddingRetrieverEvaluator:
    """Evaluate semantic retrieval independently from the generative model."""

    REQUIRED_COLUMNS = set(KNOWLEDGE_BASE_REQUIRED_COLUMNS)

    def __init__(
        self,
        knowledge_base: pd.DataFrame,
        embedding_model: str,
    ) -> None:
        self.knowledge_base = self._validate_knowledge_base(knowledge_base)
        self.embedding_model_name = embedding_model
        self.documents = build_documents(self.knowledge_base)
        self.ids = build_ids(self.documents)
        self.metadatas = [
            {"pregunta_original": metadata["pregunta_original"]}
            for metadata in build_metadatas(self.knowledge_base)
        ]

        collection_name = f"embedding_eval_{uuid4().hex[:12]}"
        self.retriever = self._create_retriever(
            embedding_model_name=self.embedding_model_name,
            ids=self.ids,
            documents=self.documents,
            metadatas=self.metadatas,
            collection_name=collection_name,
        )
        self.documents = self.retriever.documents
        self.ids = self.retriever.ids
        self.metadatas = self.retriever.metadatas
        self.embedding_model = self.retriever.embedding_model
        self.client = self.retriever.client
        self.collection = self.retriever.collection

    @staticmethod
    def _create_retriever(
        *,
        documents: Sequence[str],
        ids: Sequence[str],
        metadatas: Sequence[dict[str, str]],
        embedding_model_name: str,
        collection_name: str,
    ) -> ChromaRetriever:
        return ChromaRetriever(
            documents=documents,
            ids=ids,
            metadatas=metadatas,
            embedding_model_name=embedding_model_name,
            collection_name=collection_name,
            top_k=len(documents),
        )

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        embedding_model: str,
    ) -> "EmbeddingRetrieverEvaluator":
        return cls(
            knowledge_base=load_knowledge_base(path),
            embedding_model=embedding_model,
        )

    @classmethod
    def _validate_knowledge_base(cls, knowledge_base: pd.DataFrame) -> pd.DataFrame:
        return validate_knowledge_base(
            knowledge_base,
            required_columns=cls.REQUIRED_COLUMNS,
        )

    def _retrieve(self, query: str, top_k: int) -> RetrievalResult:
        if not query.strip():
            raise ValueError("Evaluation queries cannot be empty.")

        self.retriever.embedding_model_name = self.embedding_model_name
        self.retriever.embedding_model = self.embedding_model
        self.retriever.documents = self.documents
        self.retriever.collection = self.collection
        self.retriever.top_k = top_k
        return self.retriever.retrieve(query)

    def evaluate(
        self,
        cases: Sequence[EvaluationCase],
        top_k_values: Sequence[int] = (1, 3, 5),
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not cases:
            raise ValueError("At least one evaluation case is required.")

        normalized_top_k = sorted(set(top_k_values))
        if not normalized_top_k or normalized_top_k[0] < 1:
            raise ValueError("top_k values must be greater than or equal to 1.")

        max_top_k = min(max(normalized_top_k), len(self.documents))
        detail_rows: list[dict[str, object]] = []

        for case in cases:
            result = self._retrieve(case.query, max_top_k)

            retrieved_questions = [
                str(metadata["pregunta_original"])
                for metadata in result.metadatas
            ]
            distances = [float(value) for value in result.distances]

            try:
                expected_rank = retrieved_questions.index(case.expected_question) + 1
            except ValueError:
                expected_rank = None

            for top_k in normalized_top_k:
                effective_top_k = min(top_k, len(self.documents))
                hit = expected_rank is not None and expected_rank <= effective_top_k

                detail_rows.append(
                    {
                        "model": self.embedding_model_name,
                        "query": case.query,
                        "expected_question": case.expected_question,
                        "top_k": effective_top_k,
                        "hit": hit,
                        "expected_rank": expected_rank,
                        "top_result": retrieved_questions[0],
                        "top_distance": distances[0],
                    }
                )

        details = pd.DataFrame(detail_rows)
        summary = (
            details.groupby(["model", "top_k"], as_index=False)
            .agg(
                hit_rate=("hit", "mean"),
                hits=("hit", "sum"),
                total=("hit", "size"),
                mean_expected_rank=("expected_rank", "mean"),
            )
            .sort_values(["hit_rate", "mean_expected_rank"], ascending=[False, True])
            .reset_index(drop=True)
        )

        return summary, details


def compare_embedding_models(
    knowledge_base_path: str | Path,
    embedding_models: Sequence[str],
    cases: Sequence[EvaluationCase],
    top_k_values: Sequence[int] = (1, 3, 5),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not embedding_models:
        raise ValueError("At least one embedding model is required.")

    summaries: list[pd.DataFrame] = []
    details: list[pd.DataFrame] = []

    for model_name in embedding_models:
        evaluator = EmbeddingRetrieverEvaluator.from_csv(
            path=knowledge_base_path,
            embedding_model=model_name,
        )
        model_summary, model_details = evaluator.evaluate(
            cases=cases,
            top_k_values=top_k_values,
        )
        summaries.append(model_summary)
        details.append(model_details)

    return (
        pd.concat(summaries, ignore_index=True),
        pd.concat(details, ignore_index=True),
    )
