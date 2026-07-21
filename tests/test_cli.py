from __future__ import annotations

from unittest.mock import Mock, call

import src.cli as cli_module


STARTUP_OUTPUT = (
    "Cargando modelos y base de conocimiento...\n"
    "\nRAG Medical Assistant listo.\n"
    "Escribí una consulta administrativa o 'salir' para finalizar.\n\n"
)


def run_cli_with_doubles(
    inputs: list[str],
    *,
    answers: list[object] | None = None,
) -> tuple[Mock, Mock, list[str], list[tuple[str, str | None]], str]:
    chatbot = Mock(name="chatbot")
    if answers is not None:
        chatbot.answer.side_effect = answers

    events: list[tuple[str, str | None]] = []

    def create_chatbot() -> Mock:
        events.append(("factory", None))
        return chatbot

    chatbot_factory = Mock(side_effect=create_chatbot)
    prompts: list[str] = []
    input_values = iter(inputs)

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return next(input_values)

    output_parts: list[str] = []

    def fake_print(message: str) -> None:
        events.append(("print", message))
        output_parts.append(f"{message}\n")

    cli_module.main(
        chatbot_factory=chatbot_factory,
        input_fn=fake_input,
        print_fn=fake_print,
    )
    return chatbot_factory, chatbot, prompts, events, "".join(output_parts)


def test_cli_preserves_defaults_blank_answer_and_exit_behavior() -> None:
    chatbot_factory, chatbot, prompts, events, output = run_cli_with_doubles(
        ["   ", "  ¿Cómo confirmo mi turno?  ", "SALIR"],
        answers=["Respuesta de prueba."],
    )

    chatbot_factory.assert_called_once_with()
    assert events[:2] == [
        ("print", "Cargando modelos y base de conocimiento..."),
        ("factory", None),
    ]
    chatbot.answer.assert_called_once_with("¿Cómo confirmo mi turno?")
    assert prompts == ["Usuario: ", "Usuario: ", "Usuario: "]
    assert output == (
        STARTUP_OUTPUT
        + "Asistente: Ingresá una consulta válida.\n\n"
        + "Asistente: Respuesta de prueba.\n\n"
        + "Asistente finalizado.\n"
    )


def test_cli_reports_answer_error_and_continues_the_conversation() -> None:
    _, chatbot, prompts, _, output = run_cli_with_doubles(
        ["  consulta fallida  ", "consulta recuperada", "quit"],
        answers=[RuntimeError("fallo controlado"), "Respuesta recuperada."],
    )

    assert chatbot.answer.call_args_list == [
        call("consulta fallida"),
        call("consulta recuperada"),
    ]
    assert prompts == ["Usuario: ", "Usuario: ", "Usuario: "]
    assert output == (
        STARTUP_OUTPUT
        + "Error al procesar la consulta: fallo controlado\n\n"
        + "Asistente: Respuesta recuperada.\n\n"
        + "Asistente finalizado.\n"
    )


def test_cli_accepts_exit_alias_case_insensitively() -> None:
    _, chatbot, prompts, _, output = run_cli_with_doubles(["  ExIt  "])

    chatbot.answer.assert_not_called()
    assert prompts == ["Usuario: "]
    assert output == STARTUP_OUTPUT + "Asistente finalizado.\n"
