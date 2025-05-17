# Technical Context

## Technologies used
- **Python 3.12+**: Core programming language for the MCP server.
- **FastMCP**: Framework used for building the Model Context Protocol server.
  - **API Note**: The server uses FastMCP with specific transport methods. The available methods include:
    - `run()`: Synchronous method for running the server
    - `run_sse_async()`: Asynchronous method specifically for SSE transport
    - Note: `run_async()` method mentioned in newer documentation is not available in the current version
- **Crawl4AI**: Primary library for web crawling, page processing, and Markdown conversion.
- **Text Embeddings Inference (TEI) Server**: Used to serve the `BAAI/bge-large-en-v1.5` model.
    - **Model:** `BAAI/bge-large-en-v1.5` (1024 dimensions) for generating text embeddings.
- **`requests` library**: Used by the MCP server to make HTTP POST requests to the TEI server's `/embed` endpoint.
- **Qdrant**: Open-source vector database used for storing and querying text embeddings and associated metadata.
    - **`qdrant-client`**: Python library for interacting with the Qdrant instance.
- **Docker & Docker Compose**: For containerizing the MCP server and (as per user setup) the TEI server. Qdrant is also typically run in Docker but managed separately in the user's current setup.
- **OpenAI API (Optional, Currently Disabled)**: The system has latent code to use an OpenAI model (e.g., `gpt-4o` via `SUMMARIZATION_MODEL_CHOICE`) for contextual summaries, but this feature is currently disabled in `src/utils.py` to avoid payload size issues with the TEI server. The `openai` Python library is an optional dependency.
- **`uv`**: For Python package and virtual environment management.

## Development setup
- Python virtual environment managed with `uv`.
- Configuration driven by environment variables in a `.env` file (e.g., `EMBEDDING_SERVER_URL`, `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`, `VECTOR_DIM`).
- The MCP server runs as a Docker container, orchestrated by `docker-compose.yml`.
- **External Services Required:** A running Qdrant instance and a running TEI server (serving `BAAI/bge-large-en-v1.5`) are prerequisites and their URLs must be configured in the `.env` file.

## Technical constraints & Key Configurations
- **Network Accessibility:** The MCP server container needs network access to the TEI server and the Qdrant instance at the specified URLs/ports.
- **Embedding Dimension:** `VECTOR_DIM` environment variable **must be set to 1024** to match the output of `BAAI/bge-large-en-v1.5`.
- **Chunk Size:** Content is chunked to a target of 750 characters using `smart_chunk_markdown` to manage payload sizes for the TEI server and align with typical model token limits.
- **TEI Server Payload Limit:** The TEI server has its own HTTP payload limits (default 2MB). The 750-character client-side chunking (with summaries disabled) is designed to stay within this.
- **Model Token Limit:** `BAAI/bge-large-en-v1.5` has a maximum sequence length of 512 tokens. Text chunks exceeding this (after tokenization by the TEI server) will be truncated by the model.
- **Crawl4AI Dependencies:** Browser dependencies for Crawl4AI are handled within its Docker image or require `crawl4ai-setup` for local (non-Docker) development.

## Memory requirements for parallel crawling (configurable) 