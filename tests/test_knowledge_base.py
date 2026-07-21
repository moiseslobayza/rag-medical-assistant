from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.knowledge_base import (
    build_documents,
    build_ids,
    build_metadatas,
    load_knowledge_base,
    validate_knowledge_base,
)


def write_rows(path: Path, rows: list[dict[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pregunta", "respuesta"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_knowledge_base_preserves_cleaning_order_and_artifacts(tmp_path: Path) -> None:
    path = write_rows(
        tmp_path / "knowledge_base.csv",
        [
            {"pregunta": "  Primera pregunta  ", "respuesta": " Primera respuesta "},
            {"pregunta": "", "respuesta": "Fila inválida"},
            {"pregunta": "Segunda pregunta", "respuesta": "  Segunda respuesta"},
            {"pregunta": "Fila inválida", "respuesta": "   "},
        ],
    )

    validated = validate_knowledge_base(load_knowledge_base(path))
    documents = build_documents(validated)

    assert validated.to_dict("records") == [
        {"pregunta": "Primera pregunta", "respuesta": "Primera respuesta"},
        {"pregunta": "Segunda pregunta", "respuesta": "Segunda respuesta"},
    ]
    assert documents == [
        "Pregunta frecuente: Primera pregunta\nRespuesta: Primera respuesta",
        "Pregunta frecuente: Segunda pregunta\nRespuesta: Segunda respuesta",
    ]
    assert build_ids(documents) == ["doc_0", "doc_1"]
    assert build_metadatas(validated) == [
        {
            "pregunta_original": "Primera pregunta",
            "respuesta_original": "Primera respuesta",
        },
        {
            "pregunta_original": "Segunda pregunta",
            "respuesta_original": "Segunda respuesta",
        },
    ]


def test_load_knowledge_base_rejects_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError) as error:
        load_knowledge_base(missing_path)

    assert str(error.value) == f"Knowledge base not found: {missing_path}"
