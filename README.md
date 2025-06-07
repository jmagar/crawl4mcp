<h1 align="center">🚀 Crawl4MCP RAG Server 🚀</h1>

<p align="center">
  <em>Enhance AI Agents & Coding Assistants with Web Crawling 🕸️, RAG 🧠, and Analytics 📊 Capabilities!</em>
</p>

An MCP (Model Context Protocol) server built with FastMCP, integrated with Crawl4AI and Qdrant. This server empowers AI agents by providing advanced web crawling, Retrieval Augmented Generation (RAG), and data analytics tools.

It uses a self-hosted BAAI/bge-large-en-v1.5 model (via a TEI server) for text embeddings and Qdrant for vector storage.

## ✨ Features

*   **🤖 Smart Web Crawling**: Powered by Crawl4AI:
    *   Single pages, sitemaps, text files with URL lists.
    *   Recursive website crawling.
    *   Git Repositories.
*   **⚙️ Self-Hosted Embeddings**: Uses BAAI/bge-large-en-v1.5 via a Text Embeddings Inference (TEI) server.
*   **💾 Self-Hosted Vector Storage**: Qdrant for crawled content and embeddings.
*   **🔍 RAG & Hybrid Search**: Semantic and keyword-enhanced search.
*   **📊 Advanced Analytics & Management**:
    *   `get_collection_stats`: Detailed Qdrant collection statistics.
    *   `get_available_sources`: Lists unique crawled source domains.
    *   `cluster_content`: K-means clustering of vectors with optional t-SNE visualization.
*   **🔌 MCP Integration**: All functionalities exposed as MCP tools.
*   **📦 Easy Deployment & Development**:
    *   `docker-compose.yml` for full stack orchestration (MCP server, TEI, Qdrant).
    *   Live reload for the MCP server in Docker.
*   **🛡️ Robust Tooling**: Fallback mechanisms for critical service initialization.

## 🛠️ Setup and Installation

### Prerequisites

*   Python 3.12+ (Dockerfile uses `python:3.12-slim`)
*   Docker and Docker Compose
*   NVIDIA GPU (recommended for TEI server)
*   (Optional) `HUGGING_FACE_HUB_TOKEN` if your TEI model needs it.

### 1. Clone the Repository 📂

   ```bash
   git clone https://github.com/jmagar/crawl4mcp.git # Or your fork
   cd crawl4mcp
   ```

### 2. Configure Environment Variables ⚙️

Create a `.env` file in the project root (you can copy `.env.example`):

   ```bash
   cp .env.example .env
   ```

Open and edit your `.env` file. Key variables include:

```env
# --- MCP Server --- 
HOST=0.0.0.0
PORT=9130
# PATH=/mcp # Default path prefix for MCP server if needed by client
LOG_LEVEL=DEBUG
LOG_FILENAME=crawl4mcp.log

# --- Embedding Server (TEI) --- 
EMBEDDING_SERVER_URL=http://crawl4mcp-embeddings:8080/embed # Your TEI server URL
EMBEDDING_SERVER_BATCH_SIZE=64
VECTOR_DIM=1024 # Dimension for BAAI/bge-large-en-v1.5

# --- Qdrant Vector Database --- 
QDRANT_URL=http://crawl4mcp-qdrant:6333 # Your Qdrant instance URL
QDRANT_API_KEY=your_qdrant_api_key_if_any # Optional
QDRANT_COLLECTION=crawl4mcp # Preferred collection name
QDRANT_UPSERT_BATCH_SIZE=512 # Batch size for Qdrant upserts

# --- Crawl4AI --- 
CHUNK_SIZE=500 # General text chunk size
CHUNK_OVERLAP=100 # Overlap for general text chunking
MARKDOWN_CHUNK_SIZE=750 # Chunk size for Markdown content from web pages
CRAWLER_VERBOSE=true # Verbosity for Crawl4AI's browser operations

# --- Docker --- 
# COMPOSE_BAKE=true # For Docker buildkit
# DOCKER_BUILDKIT=1
# COMPOSE_DOCKER_CLI_BUILD=1

# --- Hugging Face Hub Token (if TEI model requires it for download) --- 
# HUGGING_FACE_HUB_TOKEN=your_hf_token_here
```

**Key Points for `.env`:**

*   Ensure `EMBEDDING_SERVER_URL` and `QDRANT_URL` point to your running instances.
*   `VECTOR_DIM` must match your embedding model (1024 for `BAAI/bge-large-en-v1.5`).
*   The `docker-compose.yml` in this repository is primarily for the MCP application itself; it assumes TEI and Qdrant might be run as separate services or externally.

### 3. Build and Run with Docker Compose 🐳

```bash
docker compose up --build -d
```

This command will:

*   Build the MCP server image.
*   Start the MCP server.
*   Enable live reload for changes in `./src`.
*   The MCP server will be accessible based on `HOST` and `PORT` (e.g., `http://localhost:9130`).

To stop the service:
```bash
docker compose down
```

### 3. (Alternative) Local Python Environment Setup for MCP Server 💻

Briefly:
```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[visualization]"
crawl4ai-setup
# Start the server (ensure .env is configured)
python -m src.app
```
Ensure your `.env` correctly points to your TEI and Qdrant instances if you are running them externally (i.e., not using the example `docker-compose.yml` which includes them).

## 🛠️ MCP Tools Provided

Once the server is running, it exposes these tools (prefixes removed):

*   `crawl_single_page(url: str)`
    *   `url: str` - The URL of the single web page to crawl.
*   `smart_crawl_url(url: str, max_depth: Optional[int] = 3, chunk_size: Optional[int] = None, max_concurrent: Optional[int] = 30)`
    *   `url: str` - The starting URL to crawl (can be a webpage, sitemap.xml, or .txt file with URLs).
    *   `max_depth: Optional[int]` (default: 3) - Maximum depth for recursive crawls if the URL is a webpage.
    *   `chunk_size: Optional[int]` (default: `MARKDOWN_CHUNK_SIZE` from env, typically 750) - Max size of each markdown content chunk from web pages.
    *   `max_concurrent: Optional[int]` (default: 30) - Maximum concurrent requests for crawling multiple URLs (e.g., from sitemap or recursive crawl).
*   `crawl_repo(repo_url: str, branch: Optional[str] = None, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None, ignore_dirs: Optional[List[str]] = None, allowed_extensions: Optional[List[str]] = None)`
    *   `repo_url: str` - URL of the Git repository to crawl.
    *   `branch: Optional[str]` (default: repository's default branch) - Specific branch to clone.
    *   `chunk_size: Optional[int]` (default: `CHUNK_SIZE` from env, typically 500) - Size of each text chunk for processing repository files.
    *   `chunk_overlap: Optional[int]` (default: `CHUNK_OVERLAP` from env, typically 100 for repo files) - Overlap between text chunks for repository files.
    *   `ignore_dirs: Optional[List[str]]` (default: `[".git", "node_modules", "__pycache__", ...])` - List of directory names or path patterns to ignore.
    *   `allowed_extensions: Optional[List[str]]` (default: None, processes most common text/code files) - Specific file extensions to process (e.g., `[*.py*, *.js*]`). If None, a broad set of text-based files are considered.
*   `crawl_dir(dir_path: str, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None, ignore_patterns: Optional[List[str]] = None, allowed_extensions: Optional[List[str]] = None)`
    *   **Important:** When running the MCP server in Docker, the `dir_path` must be a path accessible *inside the container*. You need to mount the host directory you wish to crawl as a volume into the container in your `docker-compose.yml` file. For example, map host `./my_docs` to container `/crawl_target` and then use `/crawl_target` as the `dir_path`.
    *   `dir_path: str` - Absolute path to the local directory to crawl.
    *   `chunk_size: Optional[int]` - Size of each text chunk. Defaults to server config.
    *   `chunk_overlap: Optional[int]` - Overlap between chunks. Defaults to server config.
    *   `ignore_patterns: Optional[List[str]]` - Glob patterns to ignore. Defaults to a predefined list (e.g., `node_modules`, `*.log`).
    *   `allowed_extensions: Optional[List[str]]` - File extensions to process. Defaults to a predefined list.
*   `perform_rag_query(query: str, source: Optional[str] = None, match_count: int = 5)`
    *   `query: str` - The natural language query for semantic search.
    *   `source: Optional[str]` (default: None) - Optional source domain to filter results (e.g., 'example.com').
    *   `match_count: int` (default: 5) - Maximum number of results to return.
*   `get_available_sources()`
    *   No parameters.
*   `perform_hybrid_search(query: str, filter_text: Optional[str] = None, vector_weight: float = 0.7, keyword_weight: float = 0.3, source: Optional[str] = None, match_count: int = 5)`
    *   `query: str` - The query text for semantic vector search.
    *   `filter_text: Optional[str]` (default: None) - Optional keyword text for filtering.
    *   `vector_weight: float` (default: 0.7) - Weight for vector search results (0.0-1.0).
    *   `keyword_weight: float` (default: 0.3) - Weight for keyword search results (0.0-1.0). (Note: weights are normalized if they don't sum to 1.0).
    *   `source: Optional[str]` (default: None) - Optional source domain to filter results.
    *   `match_count: int` (default: 5) - Maximum number of results to return.
*   `get_collection_stats(collection_name: Optional[str] = None, include_segments: bool = False)`
    *   `collection_name: Optional[str]` (default: uses `QDRANT_COLLECTION` from env) - Specific collection to get stats for. If None, stats for the default server collection (or all if no default is set) are fetched.
    *   `include_segments: bool` (default: False) - Whether to include detailed segment information (can be verbose, currently placeholder in implementation).
*   `find_similar_content(content_text: str, filter_source: Optional[str] = None, match_count: int = 5)`
    *   `content_text: str` - Text content to find similar items for.
    *   `filter_source: Optional[str]` (default: None) - Optional source domain to filter results.
    *   `match_count: int` (default: 5) - Maximum number of similar items to return.
*   `get_similar_items(item_id: str, filter_source: Optional[str] = None, match_count: int = 5)`
    *   `item_id: str` - ID of the item in Qdrant to find recommendations for.
    *   `filter_source: Optional[str]` (default: None) - Optional source domain to filter results.
    *   `match_count: int` (default: 5) - Maximum number of similar items to return.
*   `fetch_item_by_id(item_id: str, ctx: Optional[Context] = None)`
    *   `item_id: str` - ID of the item to fetch from Qdrant.
    *   `ctx: Optional[Context]` - MCP Context object (usually injected by the server).
*   `cluster_content(ctx: Context, source_filter: Optional[str] = None, num_clusters: Optional[int] = None, sample_size: Optional[int] = None, include_visualization: bool = False)`
    *   `ctx: Context` - MCP Context object (required, injected by the server).
    *   `source_filter: Optional[str]` (default: None) - Optional source domain to filter vectors for clustering.
    *   `num_clusters: Optional[int]` (default: None, auto-determined) - Number of clusters to create. If None, an optimal number is attempted.
    *   `sample_size: Optional[int]` (default: None, uses 500 in util) - Maximum number of vectors to fetch for clustering.
    *   `include_visualization: bool` (default: False) - Whether to generate and include HTML for a t-SNE visualization of clusters.
*   `view_server_logs(num_lines: Optional[int] = None)`
    *   `num_lines: Optional[int]` (default: 100 lines as configured in `logging_utils.DEFAULT_LOG_LINES`) - Number of recent log lines to retrieve from `crawl4mcp.log`.
    *   `ctx: Optional[Context]` - MCP Context object (usually injected by the server).

## 🧑‍💻 Development

This project uses `uv` for Python environment and package management.

To set up a local development environment for the MCP server:
1. Ensure you have Python 3.12+ and `uv` installed.
2. Create and activate a virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate
   ```
3. Install dependencies (including optional visualization tools):
   ```bash
   uv pip install -e ".[visualization]"
   ```
4. If you haven't already, run the Crawl4AI setup for necessary resources:
   ```bash
   crawl4ai-setup
   ```
5. Configure your `.env` file as described in the Setup section, ensuring `EMBEDDING_SERVER_URL` and `QDRANT_URL` are correctly set if you're running these services externally.
6. Run the development server:
   ```bash
   python -m src.app
   ```
   Alternatively, you can use the `start-dev.sh` script which handles this (after you've set up the environment and `.env` file):
   ```bash
   ./start-dev.sh
   ```
   This script will start the MCP server. Changes in the `./src` directory will trigger a live reload if you are running via `docker compose up` with the provided `docker-compose.yml`.

## 🙌 Contributing

Contributions welcome! Please open an issue or PR.