from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.config import DEFAULT_RAG_CONFIG, RAGConfig


def test_rag_config_preserves_current_defaults() -> None:
    assert DEFAULT_RAG_CONFIG == RAGConfig(
        knowledge_base_path=Path("data/knowledge_base.csv"),
        embedding_model="intfloat/multilingual-e5-small",
        llm_model="google/flan-t5-small",
        collection_name="medical_office_knowledge",
        top_k=3,
        chroma_distance_metric="cosine",
        embedding_convert_to_numpy=True,
        normalize_embeddings=True,
        tokenizer_return_tensors="pt",
        tokenizer_truncation=True,
        tokenizer_max_length=512,
        max_new_tokens=100,
        do_sample=False,
        num_beams=4,
        skip_special_tokens=True,
    )


def test_rag_config_is_immutable() -> None:
    config = RAGConfig()

    with pytest.raises(FrozenInstanceError):
        config.top_k = 5  # type: ignore[misc]
