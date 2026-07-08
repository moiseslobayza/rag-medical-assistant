from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from uuid import uuid4

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class EvaluationCase:
    query: str
    expected_question: str


class EmbeddingRetrieverEvaluator:
    """Evaluate semantic retrieval independently from the generative model."""

    REQUIRED_COLUMNS = {"pregunta", "respuesta"}

    def __init__(
        self,
        knowledge_base: pd.DataFrame,
        embedding_model: str,
    ) -> None:
        self.knowledge_base = self._validate_knowledge_base(knowledge_base)
        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        self.documents = [
            f"Pregunta frecuente: {row.pregunta}\nRespuesta: {row.respuesta}"
            for row in self.knowledge_base.itertuples(index=False)
        ]
        self.metadatas = [
            {"pregunta_original": str(row.pregunta)}
            for row in self.knowledge_base.itertuples(index=False)
        ]
        self.ids = [f"doc_{index}" for index in range(len(self.documents))]

        prepared_documents = self._prepare_documents(self.documents)
        embeddings = self.embedding_model.encode(
            prepared_documents,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

        self.client = chromadb.EphemeralClient()
        collection_name = f"embedding_eval_{uuid4().hex[:12]}"
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.collection.add(
            ids=self.ids,
            documents=self.documents,
            embeddings=embeddings,
            metadatas=self.metadatas,
        )

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        embedding_model: str,
    ) -> "EmbeddingRetrieverEvaluator":
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {csv_path}")

        return cls(
            knowledge_base=pd.read_csv(csv_path),
            embedding_model=embedding_model,
        )

    @classmethod
    def _validate_knowledge_base(cls, knowledge_base: pd.DataFrame) -> pd.DataFrame:
        missing_columns = cls.REQUIRED_COLUMNS - set(knowledge_base.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required columns: {missing}")

        validated = knowledge_base.copy()
        validated = validated.dropna(subset=["pregunta", "respuesta"])
        validated["pregunta"] = validated["pregunta"].astype(str).str.strip()
        validated["respuesta"] = validated["respuesta"].astype(str).str.strip()
        validated = validated[
            (validated["pregunta"] != "") & (validated["respuesta"] != "")
        ].reset_index(drop=True)

        if validated.empty:
            raise ValueError("The knowledge base contains no valid records.")

        return validated

    def _uses_e5(self) -> bool:
        return "e5" in self.embedding_model_name.lower()

    def _prepare_documents(self, documents: Sequence[str]) -> list[str]:
        if self._uses_e5():
            return [f"passage: {document}" for document in documents]
        return list(documents)

    def _prepare_query(self, query: str) -> str:
        query = query.strip()
        if not query:
            raise ValueError("Evaluation queries cannot be empty.")

        if self._uses_e5():
            return f"query: {query}"
        return query

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
            prepared_query = self._prepare_query(case.query)
            query_embedding = self.embedding_model.encode(
                [prepared_query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).tolist()

            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=max_top_k,
            )

            retrieved_questions = [
                str(metadata["pregunta_original"])
                for metadata in results["metadatas"][0]
            ]
            distances = [float(value) for value in results["distances"][0]]

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
