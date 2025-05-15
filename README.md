<h1 align="center">Crawl4AI RAG MCP Server</h1>

<p align="center">
  <em>Web Crawling and RAG Capabilities for AI Agents and AI Coding Assistants</em>
</p>

A powerful implementation of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io) integrated with [Crawl4AI](https://crawl4ai.com) and [Qdrant](https://qdrant.tech/) for providing AI agents and AI coding assistants with advanced web crawling and RAG capabilities.

This server uses a self-hosted [BGE-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) model for generating text embeddings, and Qdrant for vector storage and search.
Optionally, it can use an OpenAI model (e.g., GPT-3.5-turbo, GPT-4o-mini) for generating contextual summaries to enhance RAG retrieval.

## Features

*   **Smart Web Crawling**: Utilizes Crawl4AI to intelligently crawl various web content types:
    *   Single web pages
    *   Sitemaps (`sitemap.xml`)
    *   Text files listing URLs (`llms.txt`)
    *   Recursive crawling of websites within a specified depth
*   **Self-Hosted Embeddings**: Generates embeddings using a local BAAI/bge-large-en-v1.5 model via a TEI server.
*   **Vector Storage**: Stores crawled content and embeddings in a Qdrant database.
*   **RAG Queries**: Performs semantic search over the stored content using RAG.
*   **Contextual Summaries (Optional)**: If configured with an OpenAI API key and model, generates query-focused summaries of text chunks before embedding to improve retrieval relevance.
*   **Source Filtering**: Allows RAG queries to be filtered by specific source domains.
*   **MCP Integration**: Exposes crawling and RAG functionalities as MCP tools.
*   **Easy Deployment**: Includes Dockerfile and docker-compose.yml for straightforward setup.

## Setup and Installation

### Prerequisites

*   Python 3.12+
*   Docker and Docker Compose
*   Access to a running Qdrant instance.
*   A running Text Embeddings Inference (TEI) server with the BAAI/bge-large-en-v1.5 model.
    *   Example TEI server endpoint: `http://your-tei-server-ip:port/embed`
*   (Optional) OpenAI API Key and a chosen model if you want to enable contextual summaries.

### 1. Clone the Repository

   ```bash
   git clone https://github.com/coleam00/mcp-crawl4ai-rag.git
   cd mcp-crawl4ai-rag
   ```

### 2. Configure Environment Variables

Create a `.env` file in the project root by copying `.env.example`:

   ```bash
cp .env.example .env
```

Now, edit the `.env` file with your specific configurations:

```env
# MCP Server Transport (sse or stdio)
TRANSPORT=sse
HOST=0.0.0.0
PORT=8051 # Or any port you prefer for the MCP server

# Self-Hosted Embedding Server
EMBEDDING_SERVER_URL=http://10.1.0.7/embed # Replace with your TEI server URL

# Qdrant Configuration
QDRANT_URL=http://your-qdrant-ip:6333 # Replace with your Qdrant server URL
QDRANT_API_KEY= # Optional: Your Qdrant Cloud API Key if using Qdrant Cloud
QDRANT_COLLECTION=crawled_pages # Or your preferred collection name
VECTOR_DIM=1024 # Dimension for BGE-large-en-v1.5 embeddings

# --- Optional: For contextual summaries using an OpenAI model ---
OPENAI_API_KEY= # Your OpenAI API Key (if using contextual summaries)
SUMMARIZATION_MODEL_CHOICE=gpt-3.5-turbo # Or gpt-4o-mini, etc. (if using contextual summaries)
```

**Important Notes:**

*   If `OPENAI_API_KEY` or `SUMMARIZATION_MODEL_CHOICE` are not set, contextual summaries will be disabled, and the server will only embed the raw text chunks.
*   Ensure `VECTOR_DIM` is set to `1024` for `BAAI/bge-large-en-v1.5`.

### 3. Build and Run with Docker Compose

This is the recommended way to run the server.

```bash
docker compose up --build -d
```

This command will:

*   Build the Docker image for the MCP server.
*   Start the MCP server container.
*   The server will be accessible based on your `HOST` and `PORT` configuration (e.g., `http://localhost:8051` if `PORT=8051`).

### 4. (Alternative) Local Python Environment Setup

If you prefer not to use Docker for the MCP server (Qdrant and TEI server would still ideally be separate):

```bash
# Create and activate a virtual environment (recommended)
uv venv
source .venv/bin/activate

# Install dependencies (including optional openai if you plan to use it)
uv pip install -e ".[openai]" # or just `uv pip install -e .` if not using summaries

# Run playwright browser setup for Crawl4AI
crawl4ai-setup

# Run the MCP server
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