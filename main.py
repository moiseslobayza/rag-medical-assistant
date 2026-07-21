from src import MedicalRAGChatbot
from src.cli import main as run_cli
from src.config import DEFAULT_RAG_CONFIG


KNOWLEDGE_BASE_PATH = DEFAULT_RAG_CONFIG.knowledge_base_path


def _create_chatbot() -> MedicalRAGChatbot:
    return MedicalRAGChatbot.from_csv(
        KNOWLEDGE_BASE_PATH,
        embedding_model=DEFAULT_RAG_CONFIG.embedding_model,
        llm_model=DEFAULT_RAG_CONFIG.llm_model,
        top_k=DEFAULT_RAG_CONFIG.top_k,
    )


def main() -> None:
    run_cli(chatbot_factory=_create_chatbot)


if __name__ == "__main__":
    main()
