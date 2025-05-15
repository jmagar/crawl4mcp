<h1 align="center">Crawl4AI RAG MCP Server</h1>

<p align="center">
  <em>Web Crawling and RAG Capabilities for AI Agents and AI Coding Assistants</em>
</p>

A powerful implementation of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io) integrated with [Crawl4AI](https://crawl4ai.com) and [Qdrant](https://qdrant.tech/) for providing AI agents and AI coding assistants with advanced web crawling and RAG capabilities.

This server uses a self-hosted [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) model for generating text embeddings, and Qdrant for vector storage and search.
Optionally, it can use an OpenAI model (e.g., GPT-3.5-turbo, GPT-4o-mini) for generating contextual summaries to enhance RAG retrieval.

## Features

*   **Smart Web Crawling**: Utilizes Crawl4AI to intelligently crawl various web content types:
    *   Single web pages
    *   Sitemaps (`sitemap.xml`)
    *   Text files listing URLs (`llms.txt`)
    *   Recursive crawling of websites within a specified depth
*   **Self-Hosted Embeddings**: Generates embeddings using a local BAAI/bge-large-en-v1.5 model via the included TEI server.
*   **Vector Storage**: Stores crawled content and embeddings in the included Qdrant database.
*   **RAG Queries**: Performs semantic search over the stored content using RAG.
*   **Contextual Summaries (Optional)**: If configured with an OpenAI API key and model, generates query-focused summaries of text chunks before embedding to improve retrieval relevance.
*   **Source Filtering**: Allows RAG queries to be filtered by specific source domains.
*   **MCP Integration**: Exposes crawling and RAG functionalities as MCP tools.
*   **Easy Deployment**: Includes a `docker-compose.yml` that orchestrates the MCP server, Text Embeddings Inference (TEI) server, and Qdrant database.

## Setup and Installation

### Prerequisites

*   Python 3.12+ (only if planning to run the MCP server locally outside Docker)
*   Docker and Docker Compose
*   NVIDIA GPU with drivers installed (if using the TEI server with GPU acceleration as configured in `docker-compose.yml`)
*   (Optional) OpenAI API Key and a chosen model if you want to enable contextual summaries.
*   (Optional) `HUGGING_FACE_HUB_TOKEN` environment variable set in your system or `.env` file if the TEI server needs to download models from a private Hugging Face Hub repository.

### 1. Clone the Repository

   ```bash
   git clone https://github.com/jmagar/crawl4mcp.git # Or your fork
   cd crawl4mcp
   ```

### 2. Configure Environment Variables

Create a `.env` file in the project root by copying `.env.example`:

   ```bash
cp .env.example .env
```

Now, edit the `.env` file. Many settings have sensible defaults for the Docker Compose setup:

```env
# MCP Server Transport (sse or stdio)
TRANSPORT=sse
HOST=0.0.0.0
PORT=8051 # Port for the MCP server

# Self-Hosted Embedding Server (Defaults to the service in docker-compose.yml)
# EMBEDDING_SERVER_URL=http://localhost:8080 # TEI runs on host network, port 8080

# Qdrant Configuration (Defaults to the service in docker-compose.yml)
# QDRANT_URL=http://qdrant:6333 # Service name for Qdrant within Docker network
QDRANT_API_KEY= # Optional: Your Qdrant API Key if using an external Qdrant Cloud instance
QDRANT_COLLECTION=crawled_pages # Or your preferred collection name
VECTOR_DIM=1024 # Dimension for BAAI/bge-large-en-v1.5 embeddings

# Hugging Face Hub Token (if your TEI model needs it)
# HUGGING_FACE_HUB_TOKEN=your_hf_token_here

# --- Optional: For contextual summaries using an OpenAI model ---
OPENAI_API_KEY= # Your OpenAI API Key (if using contextual summaries)
SUMMARIZATION_MODEL_CHOICE=gpt-3.5-turbo # Or gpt-4o-mini, etc. (if using contextual summaries)
```

**Important Notes:**

*   The `docker-compose.yml` sets default values for `EMBEDDING_SERVER_URL` (to `http://localhost:8080`) and `QDRANT_URL` (to `http://qdrant:6333`) if they are not provided in the `.env` file. These defaults are suitable for the included services.
*   If `OPENAI_API_KEY` or `SUMMARIZATION_MODEL_CHOICE` are not set, contextual summaries will be disabled.
*   Ensure `VECTOR_DIM` is set to `1024` for `BAAI/bge-large-en-v1.5`.
*   If your `BAAI/bge-large-en-v1.5` model is private or requires authentication to download, provide your `HUGGING_FACE_HUB_TOKEN` in the `.env` file.

### 3. Build and Run with Docker Compose

This is the recommended way to run the entire stack (MCP Server, TEI Embeddings Server, Qdrant).

```bash
docker-compose up --build -d
```

This command will:

*   Build the Docker image for the MCP server.
*   Pull images for TEI and Qdrant if not already present.
*   Start all three services.
*   The MCP server will be accessible based on your `HOST` and `PORT` configuration (e.g., `http://localhost:8051` if `PORT=8051`).
*   Qdrant data will be persisted in a Docker volume named `qdrant_data`.
*   TEI model weights will be persisted to a local `./models` directory.

To stop the services:
```bash
docker-compose down
```

### 4. (Alternative) Local Python Environment Setup for MCP Server

If you prefer to run only the MCP server locally (e.g., for development) and connect to separately managed TEI and Qdrant instances:

```bash
# Create and activate a virtual environment (recommended)
uv venv
source .venv/bin/activate

# Install dependencies (including optional openai if you plan to use it)
uv pip install -e ".[openai]" # or just `uv pip install -e .` if not using summaries

# Run playwright browser setup for Crawl4AI
crawl4ai-setup

# Ensure your .env file points to your externally running TEI and Qdrant instances.
# Then run the MCP server:
python src/crawl4ai_mcp.py
```

## MCP Tools

Once the server is running, it provides the following MCP tools:

*   `mcp_crawl4ai_crawl_single_page(url: str)`: Crawls a single web page and stores its content.
*   `mcp_crawl4ai_smart_crawl_url(url: str, max_depth: Optional[int] = 3, max_concurrent: Optional[int] = 10, chunk_size: Optional[int] = 5000)`: Intelligently crawls a URL (detects sitemaps, text files, or regular webpages for recursive crawling) and stores the content.
*   `mcp_crawl4ai_perform_rag_query(query: str, source: Optional[str] = None, match_count: Optional[int] = 5)`: Performs a RAG query on the stored content.
*   `mcp_crawl4ai_get_available_sources()`: Retrieves a list of unique source domains that have been crawled.

## Development

Ensure you have `uv` installed for managing the Python environment.

```bash
# Install dependencies
uv pip install -e ".[openai]"

# To run linters/formatters (example using ruff)
# uv run ruff format .
# uv run ruff check .
```

## Memory Bank (For Cline - AI Assistant)

This project uses a concept called a "Memory Bank" to help AI assistants like Cline maintain context across sessions. If you are Cline or a similar AI, please refer to the `cline_docs/` directory and the custom instructions provided at the start of your interaction.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.