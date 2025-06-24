from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    # Existing configurations (based on project context)
    EMBEDDING_SERVER_URL: str = "http://localhost:8080"
    EMBEDDING_SERVER_BATCH_SIZE: int = 32 # Default, can be overridden by .env
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION: str = "crawl4mcp-rrf"
    VECTOR_DIM: int = 1024 # Vector dimension for embeddings (matches BAAI/bge-large-en-v1.5)
    LOG_LEVEL: str = "INFO"
    LOG_FILENAME: str = "crawl4mcp_server.log"
    MARKDOWN_CHUNK_SIZE: int = 1000
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    QDRANT_UPSERT_BATCH_SIZE: int = 100
    SPLADE_BATCH_SIZE: int = 32
    CRAWLER_VERBOSE: bool = False # Default, can be overridden by .env
    MAX_CONCURRENT_REQUESTS: int = 50 # Default optimized for high-performance systems

    # MCP Server settings
    MCP_SERVER_NAME: str = "crawl4mcp"
    MCP_HOST: str = "0.0.0.0"
    MCP_PORT: int = 9130
    MCP_TIMEOUT: int = 1200
    MCP_PATH_PREFIX: str = "/mcp" # Default, can be overridden by .env's PATH

    DEFAULT_ALLOWED_EXTENSIONS: List[str] = [
        ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".sh", ".rst", ".html", ".htm",
        ".c", ".h", ".cpp", ".hpp", ".java", ".js", ".ts", ".cs", ".go", ".php", ".rb", ".swift", ".kt"
    ]
    DEFAULT_LOCAL_IGNORE_PATTERNS: List[str] = [
        ".git/*", "*.db", "*.sqlite", "*.log", "__pycache__/*", "*.DS_Store",
        "node_modules/*", "dist/*", "build/*", "venv/*", ".venv/*", "env/*", ".env/*"
    ]

    # New configurations for Agentic RAG and Reranking
    USE_AGENTIC_RAG: bool = False # Default to False, enable explicitly
    CODE_BLOCK_MIN_LENGTH: int = 200 # Min characters for a code block to be extracted
    USE_RERANKING: bool = False # Default to False, enable explicitly
    CROSS_ENCODER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANK_TOP_N: int = 10 # Number of initial results to rerank

    # For loading .env file if present (optional)
    # For this to work, you'd typically have python-dotenv installed
    # and call load_dotenv() before creating an instance of Settings.
    # However, Pydantic can often load from environment variables directly.

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore" # Ignore extra fields from .env

# Global settings instance
settings = Settings()

# Example of how to load .env if you were using python-dotenv explicitly
# from dotenv import load_dotenv
# load_dotenv()
# settings = Settings()
