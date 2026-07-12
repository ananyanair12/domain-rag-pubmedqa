"""
Central configuration for the RAG pipeline.
Loads settings from environment variables (see .env.example).
"""
import os

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    def load_dotenv() -> bool:
        return False

load_dotenv()


class Config:
    # --- LLM (generation) ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    # --- Embeddings ---
    BASELINE_EMBEDDING_MODEL = os.getenv(
        "BASELINE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    FINETUNED_EMBEDDING_PATH = os.getenv(
        "FINETUNED_EMBEDDING_PATH", "./models/finetuned-embedder-pubmedqa"
    )

    # --- Chunking ---
    CHUNK_SIZES_TO_ABLATE = [256, 512, 1024]
    SELECTED_CHUNK_SIZE = int(os.getenv("SELECTED_CHUNK_SIZE", "512"))
    CHUNK_OVERLAP_RATIO = 0.1  # 10% overlap between chunks

    # --- Vector store ---
    VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "chroma")
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")

    # --- Retrieval ---
    TOP_K_RETRIEVAL = 20      # candidates pulled from the vector store
    TOP_K_RERANKED = 5        # final chunks kept after reranking
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # --- Data paths ---
    DATA_RAW_DIR = "./data/raw"
    DATA_PROCESSED_DIR = "./data/processed"
    TRAIN_PAIRS_PATH = f"{DATA_PROCESSED_DIR}/train_pairs.jsonl"
    EVAL_QUERIES_PATH = f"{DATA_PROCESSED_DIR}/eval_queries.jsonl"

    # --- Fine-tuning ---
    FT_EPOCHS = 4
    FT_BATCH_SIZE = 16
    FT_WARMUP_RATIO = 0.1
    FT_LEARNING_RATE = 2e-5

    # --- Results / reports ---
    RESULTS_DIR = "./results"


config = Config()
