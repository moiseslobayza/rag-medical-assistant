from __future__ import annotations

from collections.abc import Collection, Sequence
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"pregunta", "respuesta"}


def load_knowledge_base(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Knowledge base not found: {csv_path}")

    return pd.read_csv(csv_path)


def validate_knowledge_base(
    knowledge_base: pd.DataFrame,
    *,
    required_columns: Collection[str] = REQUIRED_COLUMNS,
) -> pd.DataFrame:
    missing_columns = set(required_columns) - set(knowledge_base.columns)
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


def build_documents(knowledge_base: pd.DataFrame) -> list[str]:
    return [
        f"Pregunta frecuente: {row.pregunta}\nRespuesta: {row.respuesta}"
        for row in knowledge_base.itertuples(index=False)
    ]


def build_ids(documents: Sequence[str]) -> list[str]:
    return [f"doc_{index}" for index in range(len(documents))]


def build_metadatas(knowledge_base: pd.DataFrame) -> list[dict[str, str]]:
    return [
        {
            "pregunta_original": str(row.pregunta),
            "respuesta_original": str(row.respuesta),
        }
        for row in knowledge_base.itertuples(index=False)
    ]
