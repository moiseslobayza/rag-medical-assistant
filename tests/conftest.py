from __future__ import annotations

import sys
from contextlib import nullcontext
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest


def _integration_dependency_used(*args: object, **kwargs: object) -> None:
    raise AssertionError("A unit test tried to use a real integration dependency.")


def _install_import_stub(name: str, **attributes: object) -> None:
    module = ModuleType(name)
    module.__spec__ = ModuleSpec(name, loader=None)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    sys.modules[name] = module


# src.chatbot imports these libraries at module load time. Replacing them before
# importing the package guarantees that this suite cannot load real models or
# contact external services.
_install_import_stub("chromadb", EphemeralClient=_integration_dependency_used)
_install_import_stub(
    "sentence_transformers",
    SentenceTransformer=_integration_dependency_used,
)
_install_import_stub(
    "transformers",
    AutoModelForSeq2SeqLM=SimpleNamespace(
        from_pretrained=_integration_dependency_used
    ),
    AutoTokenizer=SimpleNamespace(from_pretrained=_integration_dependency_used),
)
_install_import_stub(
    "torch",
    cuda=SimpleNamespace(is_available=lambda: False),
    inference_mode=_integration_dependency_used,
)

import src.chatbot as chatbot_module  # noqa: E402


class ListResult:
    def __init__(self, values: list[list[float]]) -> None:
        self.values = values

    def tolist(self) -> list[list[float]]:
        return self.values


class FakeTensor:
    def to(self, device: str) -> "FakeTensor":
        return self


@pytest.fixture
def pipeline_doubles(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    indexed: dict[str, Any] = {}

    embedding = Mock(name="embedding_model")
    embedding.encode.side_effect = lambda texts, **kwargs: ListResult(
        [[float(index + 1), 0.0] for index in range(len(texts))]
    )

    collection = Mock(name="chroma_collection")
    collection.add.side_effect = lambda **kwargs: indexed.update(
        {key: list(value) for key, value in kwargs.items()}
    )

    def query_collection(
        *,
        query_embeddings: list[list[float]],
        n_results: int,
    ) -> dict[str, list[list[Any]]]:
        return {
            "documents": [indexed["documents"][:n_results]],
            "metadatas": [indexed["metadatas"][:n_results]],
            "distances": [
                [round(0.1 * (index + 1), 1) for index in range(n_results)]
            ],
        }

    collection.query.side_effect = query_collection
    chroma_client = Mock(name="chroma_client")
    chroma_client.create_collection.return_value = collection

    tokenizer = Mock(name="tokenizer")
    tokenizer.return_value = {
        "input_ids": FakeTensor(),
        "attention_mask": FakeTensor(),
    }
    tokenizer.decode.return_value = "  Respuesta generada de prueba.  "

    language_model = Mock(name="language_model")
    language_model.to.return_value = language_model
    language_model.generate.return_value = [[101, 102]]

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=Mock(return_value=False)),
        inference_mode=Mock(side_effect=lambda: nullcontext()),
    )

    monkeypatch.setattr(
        chatbot_module,
        "SentenceTransformer",
        Mock(return_value=embedding),
    )
    monkeypatch.setattr(
        chatbot_module.chromadb,
        "EphemeralClient",
        Mock(return_value=chroma_client),
    )
    monkeypatch.setattr(
        chatbot_module,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
    )
    monkeypatch.setattr(
        chatbot_module,
        "AutoModelForSeq2SeqLM",
        SimpleNamespace(from_pretrained=Mock(return_value=language_model)),
    )
    monkeypatch.setattr(chatbot_module, "torch", fake_torch)

    return SimpleNamespace(
        embedding=embedding,
        indexed=indexed,
        collection=collection,
        chroma_client=chroma_client,
        tokenizer=tokenizer,
        language_model=language_model,
        torch=fake_torch,
    )
