<h1 align="center">🚀 Crawl4AI RAG MCP Server 🚀</h1>

<p align="center">
  <em>Enhance AI Agents & Coding Assistants with Web Crawling 🕸️ and RAG 🧠 Capabilities!</em>
</p>

A powerful implementation of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io) integrated with [Crawl4AI](https://crawl4ai.com) and [Qdrant](https://qdrant.tech/). This server empowers AI agents and coding assistants by providing them with advanced web crawling and Retrieval Augmented Generation (RAG) functionalities.

At its core, this server leverages a self-hosted [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) model for generating text embeddings and utilizes Qdrant for efficient vector storage and search. For an even richer experience, it can optionally integrate with an OpenAI model (like GPT-3.5-turbo or GPT-4o-mini) to generate contextual summaries, further boosting the relevance of RAG retrievals.

## ✨ Features

*   **🤖 Smart Web Crawling**: Powered by Crawl4AI, it intelligently crawls diverse web content:
    *   Single web pages
    *   Sitemaps (`sitemap.xml`)
    *   Text files listing URLs (`llms.txt`)
    *   Recursive website crawling up to a specified depth
*   **⚙️ Self-Hosted Embeddings**: Generates high-quality embeddings locally using the BAAI/bge-large-en-v1.5 model via the included Text Embeddings Inference (TEI) server.
*   **💾 Vector Storage**: Securely stores crawled content and their embeddings in the bundled Qdrant vector database.
*   **🔍 RAG Queries**: Enables semantic search over the stored content using Retrieval Augmented Generation techniques.
*   **🧩 Advanced Search Capabilities**:
    *   Hybrid search combining vector similarity with keyword filtering
    *   Item-to-item recommendations for content discovery
    *   Vector clustering for pattern identification and knowledge organization
*   **📊 Monitoring Tools**: Collection statistics dashboard for monitoring database usage and performance.
*   **📝 Contextual Summaries (Optional)**: When configured with an OpenAI API key and model, it can generate query-focused summaries of text chunks before embedding, significantly improving retrieval relevance.
*   **🎯 Source Filtering**: Allows RAG queries to be refined by specific source domains, giving you more control over your search.
*   **🔌 MCP Integration**: Seamlessly exposes all crawling and RAG functionalities as tools compatible with the Model Context Protocol.
*   **📦 Easy Deployment**: Comes with a `docker-compose.yml` file that orchestrates the entire stack: MCP server, TEI server, and Qdrant database for a hassle-free setup.

## 🛠️ Setup and Installation

### Prerequisites

*   Python 3.12+ (only if you plan to run the MCP server locally, outside of Docker)
*   Docker and Docker Compose
*   NVIDIA GPU with up-to-date drivers (recommended for the TEI server with GPU acceleration as configured in `docker-compose.yml`)
*   (Optional) An OpenAI API Key and a chosen model if you wish to enable the contextual summaries feature.
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

# Text Chunking Configuration
CHUNK_SIZE=500 # Size of each text chunk in characters
CHUNK_OVERLAP=200 # Overlap between text chunks in characters
MARKDOWN_CHUNK_SIZE=750 # Size of markdown chunks (larger to preserve structure)

# Hugging Face Hub Token (if your TEI model requires it for download)
# HUGGING_FACE_HUB_TOKEN=your_hf_token_here

# --- Optional: For contextual summaries using an OpenAI model ---
OPENAI_API_KEY= # Your OpenAI API Key (if enabling contextual summaries)
SUMMARIZATION_MODEL_CHOICE=gpt-3.5-turbo # e.g., gpt-4o-mini (if enabling contextual summaries)
```

**Key Points for `.env`:**

*   The `docker-compose.yml` provides defaults for `EMBEDDING_SERVER_URL` (to `http://localhost:8080`) and `QDRANT_URL` (to `http://qdrant:6333`) if they aren't set in your `.env`. These are tailored for the included services.
*   Contextual summaries are disabled if `OPENAI_API_KEY` or `SUMMARIZATION_MODEL_CHOICE` are omitted.
*   `VECTOR_DIM` must be `1024` for the `BAAI/bge-large-en-v1.5` model.
*   If the `BAAI/bge-large-en-v1.5` model is private or requires authentication from Hugging Face Hub, ensure your `HUGGING_FACE_HUB_TOKEN` is set in the `.env` file.

### 3. Build and Run with Docker Compose 🐳

This is the **recommended method** to run the entire application stack (MCP Server, TEI Embeddings Server, and Qdrant database).

```bash
docker-compose up --build -d
```

This single command will:

*   🛠️ Build the Docker image for the MCP server.
*   📥 Pull the necessary images for the TEI server and Qdrant database if they're not already on your system.
*   🚀 Start all three services, fully orchestrated.
*   🌐 The MCP server will be accessible based on your `HOST` and `PORT` settings (e.g., `http://localhost:8051` if `PORT=8051`).
*   💾 Qdrant data will be persisted in a Docker volume named `qdrant_data`.
*   🧠 TEI model weights will be saved to a local `./models` directory.

To stop all services:
```bash
docker-compose down
```

### 4. (Alternative) Local Python Environment Setup for MCP Server 💻

If you prefer to run only the MCP server locally (e.g., for development purposes) and connect to TEI and Qdrant instances that are managed separately:

```bash
# Create and activate a virtual environment (highly recommended)
uv venv
source .venv/bin/activate

# Install dependencies (include 'openai' optional group if using summaries)
uv pip install -e ".[openai]" # or just `uv pip install -e .`

# Set up Playwright browsers for Crawl4AI
crawl4ai-setup

# Important: Ensure your .env file correctly points to your externally running TEI and Qdrant instances.
# Then, run the MCP server:
python src/crawl4ai_mcp.py
```

## 🛠️ MCP Tools Provided

Once the server is up and running, it exposes the following tools through the Model Context Protocol:

### Crawling Tools

*   `mcp_crawl4ai_crawl_single_page(url: str)`: Crawls a single web page and stores its content.
*   `mcp_crawl4ai_smart_crawl_url(url: str, max_depth: Optional[int] = 3, max_concurrent: Optional[int] = 10, chunk_size: Optional[int] = 5000)`: Intelligently crawls a given URL. It can detect and process sitemaps, text files containing URL lists, or perform recursive crawls on regular web pages. Content is then stored.
*   `mcp_crawl4ai_crawl_repo(repo_url: str, branch: Optional[str] = None, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None)`: Clones a Git repository, processes all files, chunks their content, and stores them for RAG. Chunk size and overlap can be configured via parameters or environment variables.

### Search & Retrieval Tools

*   `mcp_crawl4ai_perform_rag_query(query: str, source: Optional[str] = None, match_count: Optional[int] = 5)`: Executes a Retrieval Augmented Generation query against the indexed content.
*   `mcp_crawl4ai_perform_hybrid_search(query: str, filter_text: Optional[str] = None, vector_weight: Optional[float] = 0.7, keyword_weight: Optional[float] = 0.3, source: Optional[str] = None, match_count: Optional[int] = 5)`: Combines vector similarity search with keyword/text-based filtering for more precise results.
*   `mcp_crawl4ai_item_recommendations(item_id: Optional[str] = None, content: Optional[str] = None, source: Optional[str] = None, match_count: Optional[int] = 5)`: Finds similar items based on vector similarity using either an existing item ID or provided content text.

### Analytics & Monitoring Tools

*   `mcp_crawl4ai_get_available_sources()`: Fetches a list of unique source domains that have been successfully crawled and stored.
*   `mcp_crawl4ai_get_collection_statistics(collection_name: Optional[str] = None, include_segments: Optional[bool] = False)`: Provides detailed statistics about Qdrant collections including size, point count, and configuration details.
*   `mcp_crawl4ai_vector_clustering(collection_name: Optional[str] = None, source: Optional[str] = None, num_clusters: Optional[int] = 5, sample_size: Optional[int] = 1000, generate_visualization: Optional[bool] = True, random_seed: Optional[int] = 42)`: Groups similar vectors to identify data patterns using K-means clustering with optional visualization.

## 🧑‍💻 Development

Ensure you have `uv` installed for managing the Python project environment.

```bash
# Install all dependencies, including optional ones for development
uv pip install -e ".[openai]"

# Example: To run linters/formatters (e.g., using Ruff)
# uv run ruff format .
# uv run ruff check .
```

## 📚 Advanced Usage Examples

### Hybrid Search

The hybrid search combines vector similarity with keyword filtering:

```python
result = await mcp_crawl4ai_perform_hybrid_search(
    query="machine learning optimization techniques",
    filter_text="gradient descent",  # Will prioritize content containing this text
    vector_weight=0.6,  # Give slightly more priority to semantic similarity
    keyword_weight=0.4,  # And slightly less to keyword matching
    match_count=10
)
```

### Item-to-Item Recommendations

Find content similar to an existing item:

```python
# Get recommendations based on an item ID
result = await mcp_crawl4ai_item_recommendations(
    item_id="01FGABCDEF123456",
    match_count=5
)

# Or get recommendations based on text content
result = await mcp_crawl4ai_item_recommendations(
    content="This is example content to find similar items for",
    match_count=5
)
```

### Vector Clustering

Identify patterns in your vector database:

```python
result = await mcp_crawl4ai_vector_clustering(
    source="example.com",  # Optional: filter by source
    num_clusters=7,  # Create 7 clusters
    sample_size=2000,  # Use up to 2000 vectors
    generate_visualization=True  # Include a visualization
)
```

## 🙌 Contributing

Contributions are highly welcome! Whether it's a bug report, feature request, or a pull request, please feel free to engage with the project.