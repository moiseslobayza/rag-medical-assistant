from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock

import pytest

import main as main_module
import src


STARTUP_OUTPUT = (
    "Cargando modelos y base de conocimiento...\n"
    "\nRAG Medical Assistant listo.\n"
    "Escribí una consulta administrativa o 'salir' para finalizar.\n\n"
)


def test_main_delegates_current_defaults_to_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chatbot = Mock(name="chatbot")
    chatbot_class = Mock(name="MedicalRAGChatbot")
    chatbot_class.from_csv.return_value = chatbot

    def run_factory(*, chatbot_factory: Callable[[], object]) -> None:
        chatbot_factory()

    run_cli = Mock(side_effect=run_factory)
    monkeypatch.setattr(main_module, "MedicalRAGChatbot", chatbot_class)
    monkeypatch.setattr(main_module, "run_cli", run_cli)

    main_module.main()

    assert main_module.KNOWLEDGE_BASE_PATH == Path("data/knowledge_base.csv")
    run_cli.assert_called_once_with(chatbot_factory=main_module._create_chatbot)
    chatbot_class.from_csv.assert_called_once_with(
        Path("data/knowledge_base.csv"),
        embedding_model="intfloat/multilingual-e5-small",
        llm_model="google/flan-t5-small",
        top_k=3,
    )


def test_main_py_preserves_guard_and_executes_real_cli_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    chatbot = Mock(name="chatbot")
    chatbot_class = Mock(name="MedicalRAGChatbot")
    chatbot_class.from_csv.return_value = chatbot
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return "salir"

    monkeypatch.setattr(src, "MedicalRAGChatbot", chatbot_class)
    monkeypatch.setattr("builtins.input", fake_input)
    main_path = str(Path(main_module.__file__))

    runpy.run_path(main_path, run_name="not_main")
    chatbot_class.from_csv.assert_not_called()

    runpy.run_path(main_path, run_name="__main__")

    chatbot_class.from_csv.assert_called_once_with(
        Path("data/knowledge_base.csv"),
        embedding_model="intfloat/multilingual-e5-small",
        llm_model="google/flan-t5-small",
        top_k=3,
    )
    chatbot.answer.assert_not_called()
    assert prompts == ["Usuario: "]
    assert capsys.readouterr().out == STARTUP_OUTPUT + "Asistente finalizado.\n"
