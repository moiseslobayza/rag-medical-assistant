from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


class Assistant(Protocol):
    def answer(self, question: str) -> str: ...


AssistantFactory = Callable[[], Assistant]
InputFunction = Callable[[str], str]
PrintFunction = Callable[[str], None]


def main(
    chatbot_factory: AssistantFactory,
    *,
    input_fn: InputFunction | None = None,
    print_fn: PrintFunction | None = None,
) -> None:
    read = input_fn if input_fn is not None else input
    write = print_fn if print_fn is not None else print

    write("Cargando modelos y base de conocimiento...")
    chatbot = chatbot_factory()

    write("\nRAG Medical Assistant listo.")
    write("Escribí una consulta administrativa o 'salir' para finalizar.\n")

    while True:
        question = read("Usuario: ").strip()

        if question.lower() in {"salir", "exit", "quit"}:
            write("Asistente finalizado.")
            break

        if not question:
            write("Asistente: Ingresá una consulta válida.\n")
            continue

        try:
            response = chatbot.answer(question)
            write(f"Asistente: {response}\n")
        except Exception as exc:
            write(f"Error al procesar la consulta: {exc}\n")
