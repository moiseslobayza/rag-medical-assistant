from src import MedicalRAGChatbot
from src.config import DEFAULT_RAG_CONFIG


KNOWLEDGE_BASE_PATH = DEFAULT_RAG_CONFIG.knowledge_base_path


def main() -> None:
    print("Cargando modelos y base de conocimiento...")

    chatbot = MedicalRAGChatbot.from_csv(
        KNOWLEDGE_BASE_PATH,
        embedding_model=DEFAULT_RAG_CONFIG.embedding_model,
        llm_model=DEFAULT_RAG_CONFIG.llm_model,
        top_k=DEFAULT_RAG_CONFIG.top_k,
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
