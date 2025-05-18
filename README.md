<h1 align="center">🚀 Crawl4AI RAG MCP Server 🚀</h1>

<p align="center">
  <em>Enhance AI Agents & Coding Assistants with Web Crawling 🕸️, RAG 🧠, and Analytics 📊 Capabilities!</em>
</p>

A powerful implementation of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io) integrated with [Crawl4AI](https://crawl4ai.com) and [Qdrant](https://qdrant.tech/). This server empowers AI agents and coding assistants by providing them with advanced web crawling, Retrieval Augmented Generation (RAG), and data analytics functionalities.

At its core, this server leverages a self-hosted [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) model for generating text embeddings and utilizes Qdrant for efficient vector storage and search. It has been extensively debugged to ensure tool robustness, especially for Qdrant interactions and dependency management within Docker.

## ✨ Features

*   **🤖 Smart Web Crawling**: Powered by Crawl4AI, it intelligently crawls diverse web content:
    *   Single web pages
    *   Sitemaps (`sitemap.xml`)
    *   Text files listing URLs (`llms.txt`)
    *   Recursive website crawling up to a specified depth
    *   Git Repositories
*   **⚙️ Self-Hosted Embeddings**: Generates high-quality embeddings locally using the BAAI/bge-large-en-v1.5 model via the included Text Embeddings Inference (TEI) server.
*   **💾 Vector Storage**: Securely stores crawled content and their embeddings in the bundled Qdrant vector database.
*   **🔍 RAG Queries**: Enables semantic search over the stored content using Retrieval Augmented Generation techniques.
*   **🔀 Hybrid Search**: Combines vector similarity with keyword-based filtering for more precise search results.
*   **📊 Advanced Analytics & Management**:
    *   `get_collection_stats`: Provides detailed statistics about a Qdrant collection, including vector counts, configuration (vector params, payload schema), and performance metrics (optimizer status, cluster info).
    *   `get_available_sources`: Lists unique source domains from crawled data.
    *   `cluster_content`: Performs K-means clustering on stored vectors, identifies potential themes, and can generate interactive 2D visualizations of clusters (requires optional dependencies).
*   **🎯 Source Filtering**: Allows RAG queries and other operations to be refined by specific source domains.
*   **🔌 MCP Integration**: Seamlessly exposes all functionalities as tools compatible with the Model Context Protocol.
*   **📦 Easy Deployment & Development**:
    *   Comes with a `docker-compose.yml` file that orchestrates the entire stack: MCP server, TEI server, and Qdrant database.
    *   Features live reload for the MCP server in Docker using `develop.watch` for a smoother development experience.
*   **🛡️ Robust Tooling**: Tools are designed with fallback mechanisms for critical services (like Qdrant client initialization) to ensure functionality even when not run within a full server request context.

## 🛠️ Setup and Installation

### Prerequisites

*   Python 3.10+ (Dockerfile uses `python:3.12-slim`, ensure local matches if not using Docker for the server)
*   Docker and Docker Compose
*   NVIDIA GPU with up-to-date drivers (recommended for the TEI server with GPU acceleration as configured in `docker-compose.yml`)
*   (Optional) An OpenAI API Key and a chosen model if you wish to enable any OpenAI-dependent features (currently, contextual summaries are not a primary feature but the framework supports it).
*   (Optional) `HUGGING_FACE_HUB_TOKEN` environment variable (set in your system or `.env` file) if the TEI server needs to download models from a private Hugging Face Hub repository.

### 1. Clone the Repository 📂

   ```bash
   git clone https://github.com/jmagar/crawl4mcp.git # Or your fork
   cd crawl4mcp
   ```

### 2. Configure Environment Variables ⚙️

Create a `.env` file in the project root by copying the example file:

   ```bash
cp .env.example .env
```

Now, open and edit the `.env` file. Many settings have sensible defaults that work well with the Docker Compose setup:

```env
# MCP Server Transport (sse or stdio)
TRANSPORT=sse
HOST=0.0.0.0
PORT=8051 # Port for the MCP server

# Self-Hosted Embedding Server (Defaults to the service in docker-compose.yml)
# EMBEDDING_SERVER_URL=http://localhost:8080 # TEI server runs on the host network, port 8080

# Qdrant Configuration (Defaults to the service in docker-compose.yml)
# QDRANT_URL=http://qdrant:6333 # Service name for Qdrant within the Docker network
QDRANT_API_KEY= # Optional: Your Qdrant API Key if using an external Qdrant Cloud instance
QDRANT_COLLECTION=crawled_pages # Or your preferred collection name
VECTOR_DIM=1024 # Dimension for BAAI/bge-large-en-v1.5 embeddings

# Hugging Face Hub Token (if your TEI model requires it for download)
# HUGGING_FACE_HUB_TOKEN=your_hf_token_here

# Optional: For any features using OpenAI (e.g., if contextual summaries were re-enabled)
# OPENAI_API_KEY=
# SUMMARIZATION_MODEL_CHOICE=gpt-3.5-turbo

# Optional: For Crawl4AI Browser behavior
# CRAWLER_VERBOSE=false
```

**Key Points for `.env`:**

*   The `docker-compose.yml` provides defaults for `EMBEDDING_SERVER_URL` (to `http://localhost:8080`) and `QDRANT_URL` (to `http://qdrant:6333`) if they aren't set in your `.env`. These are tailored for the included services.
*   `VECTOR_DIM` must be `1024` for the `BAAI/bge-large-en-v1.5` model.
*   If the `BAAI/bge-large-en-v1.5` model is private or requires authentication from Hugging Face Hub, ensure your `HUGGING_FACE_HUB_TOKEN` is set in the `.env` file.

### 3. Build and Run with Docker Compose 🐳

This is the **recommended method** to run the entire application stack (MCP Server, TEI Embeddings Server, and Qdrant database).

```bash
docker compose up --build -d # Use 'docker-compose' if 'docker compose' is not available
```

This single command will:

*   🛠️ Build the Docker image for the MCP server (including optional analytics dependencies and NLTK data).
*   📥 Pull the necessary images for the TEI server and Qdrant database if they're not already on your system.
*   🚀 Start all three services, fully orchestrated.
*   🔄 Enable live reload for the MCP server: changes in the `./src` directory will trigger an automatic rebuild and restart of the service.
*   🌐 The MCP server will be accessible based on your `HOST` and `PORT` settings (e.g., `http://localhost:8051` if `PORT=8051`).
*   💾 Qdrant data will be persisted in a Docker volume named `qdrant_data`.
*   🧠 TEI model weights will be saved to a local `./models` directory.

To stop all services:
```bash
docker compose down # Or 'docker-compose down'
```

### 4. (Alternative) Local Python Environment Setup for MCP Server 💻

If you prefer to run only the MCP server locally (e.g., for development purposes) and connect to TEI and Qdrant instances that are managed separately:

```bash
# Create and activate a virtual environment (highly recommended)
uv venv
source .venv/bin/activate

# Install dependencies (include optional groups as needed)
uv pip install -e "."                       # Base installation
# For analytics and visualization tools (like cluster_content with visualization):
uv pip install -e ".[visualization]"
# If any OpenAI features were to be used:
# uv pip install -e ".[openai]"
# To include all optional dependencies:
# uv pip install -e ".[openai,visualization]"

# Set up Playwright browsers for Crawl4AI
crawl4ai-setup

# Important: Ensure your .env file correctly points to your externally running TEI and Qdrant instances.
# Then, run the MCP server:
python -m src.app
```

## 🛠️ MCP Tools Provided

Once the server is up and running, it exposes the following tools through the Model Context Protocol:

*   `mcp_crawl4ai_crawl_single_page(url: str)`: Crawls a single web page and stores its content.
*   `mcp_crawl4ai_smart_crawl_url(url: str, max_depth: Optional[int] = 1, chunk_size: Optional[int] = None, max_concurrent: Optional[int] = 5)`: Intelligently crawls a given URL. It can detect and process sitemaps, text files containing URL lists, or perform recursive crawls on regular web pages. Content is then stored. (Defaults from `crawling_utils.AsyncWebCrawlerConfig`)
*   `mcp_crawl4ai_crawl_repo(repo_url: str, branch: Optional[str] = None, chunk_size: Optional[int] = 2000, chunk_overlap: Optional[int] = 200)`: Clones a Git repository (using `/usr/bin/git`), processes files (common code/text extensions by default), chunks their content, and stores them for RAG.
*   `mcp_crawl4ai_perform_rag_query(query: str, source: Optional[str] = None, match_count: int = 5)`: Executes a Retrieval Augmented Generation query against the indexed content.
*   `mcp_crawl4ai_get_available_sources()`: Fetches a list of unique source domains that have been successfully crawled and stored.
*   `mcp_crawl4ai_perform_hybrid_search(query: str, filter_text: Optional[str] = None, vector_weight: float = 0.7, keyword_weight: float = 0.3, source: Optional[str] = None, match_count: int = 5)`: Performs a hybrid search combining vector similarity with keyword/text-based filtering.
*   `mcp_crawl4ai_get_collection_stats(collection_name: Optional[str] = None, include_segments: bool = False)`: Gets detailed statistics about a Qdrant collection including vector count, configuration, payload schema, optimizer status, and Qdrant cluster status.
*   `mcp_crawl4ai_find_similar_content(content_text: str, filter_source: Optional[str] = None, match_count: int = 5)`: Finds similar content based on input text, optionally filtering by source. (Note: `filter_condition` in the underlying util is richer, tool exposes `filter_source`).
*   `mcp_crawl4ai_get_similar_items(item_id: str, filter_source: Optional[str] = None, match_count: int = 5)`: Finds similar items based on vector similarity using an existing item ID, optionally filtering by source.
*   `mcp_crawl4ai_cluster_content(source_filter: Optional[str] = None, num_clusters: Optional[int] = None, sample_size: Optional[int] = 1000, include_visualization: bool = False)`: Fetches vectors (optionally filtered by source and sampled), performs K-means clustering, and returns cluster information. If `include_visualization` is true and visualization dependencies (`.[visualization]`) are installed, it also generates and returns an HTML string for a 2D t-SNE plot of the clusters.

## 🧑‍💻 Development

Ensure you have `uv` installed for managing the Python project environment.

```bash
# Install all dependencies, including optional ones for development
uv pip install -e ".[visualization]" # Add other groups like [openai] if needed

# Example: To run linters/formatters (e.g., using Ruff if configured)
# uv run ruff format .
# uv run ruff check .
```

## 🙌 Contributing

Contributions are highly welcome! Whether it's a bug report, feature request, or a pull request, please feel free to engage with the project.