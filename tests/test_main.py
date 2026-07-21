from __future__ import annotations

import runpy
from pathlib import Path
from unittest.mock import Mock, call

import pytest

import main as main_module
import src


STARTUP_OUTPUT = (
    "Cargando modelos y base de conocimiento...\n"
    "\nRAG Medical Assistant listo.\n"
    "Escribí una consulta administrativa o 'salir' para finalizar.\n\n"
)


def run_cli_with_doubles(
    monkeypatch: pytest.MonkeyPatch,
    inputs: list[str],
    *,
    answers: list[object] | None = None,
) -> tuple[Mock, Mock, list[str]]:
    chatbot = Mock(name="chatbot")
    if answers is not None:
        chatbot.answer.side_effect = answers

    chatbot_class = Mock(name="MedicalRAGChatbot")
    chatbot_class.from_csv.return_value = chatbot

    prompts: list[str] = []
    input_values = iter(inputs)

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return next(input_values)

    monkeypatch.setattr(main_module, "MedicalRAGChatbot", chatbot_class)
    monkeypatch.setattr("builtins.input", fake_input)

    main_module.main()
    return chatbot_class, chatbot, prompts


def test_main_uses_current_defaults_and_handles_blank_answer_and_exit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    chatbot_class, chatbot, prompts = run_cli_with_doubles(
        monkeypatch,
        ["   ", "  ¿Cómo confirmo mi turno?  ", "SALIR"],
        answers=["Respuesta de prueba."],
    )

    assert main_module.KNOWLEDGE_BASE_PATH == Path("data/knowledge_base.csv")
    chatbot_class.from_csv.assert_called_once_with(
        Path("data/knowledge_base.csv"),
        embedding_model="intfloat/multilingual-e5-small",
        llm_model="google/flan-t5-small",
        top_k=3,
    )
    chatbot.answer.assert_called_once_with("¿Cómo confirmo mi turno?")
    assert prompts == ["Usuario: ", "Usuario: ", "Usuario: "]
    assert capsys.readouterr().out == (
        STARTUP_OUTPUT
        + "Asistente: Ingresá una consulta válida.\n\n"
        + "Asistente: Respuesta de prueba.\n\n"
        + "Asistente finalizado.\n"
    )


def test_main_reports_answer_error_and_continues_the_conversation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, chatbot, prompts = run_cli_with_doubles(
        monkeypatch,
        ["  consulta fallida  ", "consulta recuperada", "quit"],
        answers=[RuntimeError("fallo controlado"), "Respuesta recuperada."],
    )

    assert chatbot.answer.call_args_list == [
        call("consulta fallida"),
        call("consulta recuperada"),
    ]
    assert prompts == ["Usuario: ", "Usuario: ", "Usuario: "]
    assert capsys.readouterr().out == (
        STARTUP_OUTPUT
        + "Error al procesar la consulta: fallo controlado\n\n"
        + "Asistente: Respuesta recuperada.\n\n"
        + "Asistente finalizado.\n"
    )


def test_main_accepts_exit_alias_case_insensitively(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, chatbot, prompts = run_cli_with_doubles(monkeypatch, ["  ExIt  "])

    chatbot.answer.assert_not_called()
    assert prompts == ["Usuario: "]
    assert capsys.readouterr().out == STARTUP_OUTPUT + "Asistente finalizado.\n"


def test_main_py_executes_the_cli_entrypoint(
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

    runpy.run_path(str(Path(main_module.__file__)), run_name="__main__")

    chatbot_class.from_csv.assert_called_once()
    chatbot.answer.assert_not_called()
    assert prompts == ["Usuario: "]
    assert capsys.readouterr().out == STARTUP_OUTPUT + "Asistente finalizado.\n"
