from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RAGConfig:
    """Immutable defaults for the current RAG pipeline."""

    knowledge_base_path: Path = Path("data/knowledge_base.csv")
    embedding_model: str = "intfloat/multilingual-e5-small"
    llm_model: str = "google/flan-t5-small"
    collection_name: str = "medical_office_knowledge"
    top_k: int = 3
    chroma_distance_metric: str = "cosine"
    embedding_convert_to_numpy: bool = True
    normalize_embeddings: bool = True
    tokenizer_return_tensors: str = "pt"
    tokenizer_truncation: bool = True
    tokenizer_max_length: int = 512
    max_new_tokens: int = 100
    do_sample: bool = False
    num_beams: int = 4
    skip_special_tokens: bool = True


DEFAULT_RAG_CONFIG = RAGConfig()
