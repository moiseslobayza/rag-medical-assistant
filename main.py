from pathlib import Path

from src import MedicalRAGChatbot


KNOWLEDGE_BASE_PATH = Path("data/knowledge_base.csv")


def main() -> None:
    print("Cargando modelos y base de conocimiento...")

    chatbot = MedicalRAGChatbot.from_csv(
        KNOWLEDGE_BASE_PATH,
        embedding_model="intfloat/multilingual-e5-small",
        llm_model="google/flan-t5-small",
        top_k=3,
    )

    print("\nRAG Medical Assistant listo.")
    print("Escribí una consulta administrativa o 'salir' para finalizar.\n")

    while True:
        question = input("Usuario: ").strip()

        if question.lower() in {"salir", "exit", "quit"}:
            print("Asistente finalizado.")
            break

        if not question:
            print("Asistente: Ingresá una consulta válida.\n")
            continue

        try:
            response = chatbot.answer(question)
            print(f"Asistente: {response}\n")
        except Exception as exc:
            print(f"Error al procesar la consulta: {exc}\n")


if __name__ == "__main__":
    main()
