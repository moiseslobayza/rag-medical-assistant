from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


@dataclass(frozen=True)
class RetrievalResult:
    documents: list[str]
    metadatas: list[dict[str, str]]
    distances: list[float]


class MedicalRAGChatbot:
    """RAG chatbot for administrative questions about a medical office."""

    REQUIRED_COLUMNS = {"pregunta", "respuesta"}

    def __init__(
        self,
        knowledge_base: pd.DataFrame,
        embedding_model: str = "intfloat/multilingual-e5-small",
        llm_model: str = "google/flan-t5-small",
        collection_name: str = "medical_office_knowledge",
        top_k: int = 3,
    ) -> None:
        self.knowledge_base = self._validate_knowledge_base(knowledge_base)
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        self.collection_name = collection_name
        self.top_k = self._validate_top_k(top_k)

        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.documents = self._build_documents(self.knowledge_base)
        self.ids = [f"doc_{index}" for index in range(len(self.documents))]
        self.metadatas = self._build_metadatas(self.knowledge_base)

        prepared_documents = self._prepare_documents(self.documents)
        document_embeddings = self.embedding_model.encode(
            prepared_documents,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

        self.client = chromadb.EphemeralClient()
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.collection.add(
            ids=self.ids,
            documents=self.documents,
            embeddings=document_embeddings,
            metadatas=self.metadatas,
        )

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
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {csv_path}")

        knowledge_base = pd.read_csv(csv_path)
        return cls(knowledge_base=knowledge_base, **kwargs)

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

    @staticmethod
    def _validate_top_k(top_k: int) -> int:
        if top_k < 1:
            raise ValueError("top_k must be greater than or equal to 1.")
        return top_k

    @staticmethod
    def _build_documents(knowledge_base: pd.DataFrame) -> list[str]:
        return [
            f"Pregunta frecuente: {row.pregunta}\nRespuesta: {row.respuesta}"
            for row in knowledge_base.itertuples(index=False)
        ]

    @staticmethod
    def _build_metadatas(knowledge_base: pd.DataFrame) -> list[dict[str, str]]:
        return [
            {
                "pregunta_original": str(row.pregunta),
                "respuesta_original": str(row.respuesta),
            }
            for row in knowledge_base.itertuples(index=False)
        ]

    def _uses_e5(self) -> bool:
        return "e5" in self.embedding_model_name.lower()

    def _prepare_documents(self, documents: Iterable[str]) -> list[str]:
        if self._uses_e5():
            return [f"passage: {document}" for document in documents]
        return list(documents)

    def _prepare_query(self, question: str) -> str:
        question = question.strip()
        if not question:
            raise ValueError("The question cannot be empty.")

        if self._uses_e5():
            return f"query: {question}"
        return question

    def retrieve_context(self, question: str) -> RetrievalResult:
        prepared_question = self._prepare_query(question)
        question_embedding = self.embedding_model.encode(
            [prepared_question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

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
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                num_beams=4,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
