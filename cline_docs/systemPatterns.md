# System Architecture and Patterns

## How the system is built
The system is an MCP (Model Context Protocol) server designed for advanced web crawling and Retrieval Augmented Generation (RAG). It leverages:
- **Crawl4AI:** For intelligent, LLM-friendly web scraping and content extraction (Markdown format).
- **Self-Hosted TEI Server:** Running `BAAI/bge-large-en-v1.5` (1024 dimensions) for generating text embeddings locally.
- **Qdrant:** An open-source vector database used for storing text chunks and their corresponding embeddings, enabling semantic search.
- **FastMCP Framework:** For implementing the MCP server and its tools.

It follows a modular architecture with distinct components for crawling, embedding, storage, and querying.

## Key technical decisions
- **Crawl4AI for Web Scraping:** Chosen for its robust features in handling various web content types, JavaScript rendering, and producing clean Markdown suitable for LLMs.
- **Local Embeddings via TEI:** `BAAI/bge-large-en-v1.5` is used for local embedding generation. This provides control over the embedding process and avoids reliance on third-party APIs for this core step.
- **Qdrant for Vector Storage:** Selected as a scalable and efficient open-source vector database for RAG.
- **Character-Based Chunking:** Content is chunked using `smart_chunk_markdown` with a target of 750 characters, aiming to balance semantic coherence with the TEI server's payload limits and the BGE model's token capacity.
- **Contextual Summarization (Currently Disabled):** The capability to use an OpenAI model (e.g., `gpt-4o`) for query-focused summaries before embedding exists but is currently disabled due to previous TEI payload issues. If re-enabled, strict output length control would be necessary.
- **HTTP for TEI Communication:** The `requests` library is used for synchronous communication with the TEI server's `/embed` endpoint.

## Architecture patterns
- **MCP Server:** Exposes crawling and RAG functionalities as tools callable via the Model Context Protocol.
- **Service Integration:** Interacts with the local TEI server and the Qdrant database instance via network calls.
- **Environment-Driven Configuration:** Critical parameters (Qdrant URL/API key, TEI server URL, collection names, vector dimensions, optional OpenAI keys) are managed via a `.env` file.
- **Dockerized Deployment:** The MCP server is packaged as a Docker container, and `docker-compose.yml` is provided to orchestrate it (though the TEI and Qdrant services are expected to be managed externally or via separate compose definitions based on current setup).
- **Asynchronous Operations:** Core server logic and tool implementations in `src/crawl4ai_mcp.py` and `src/utils.py` heavily utilize `async/await` for non-blocking I/O, especially for crawling and Qdrant interactions.
- **Lifespan Management (`asynccontextmanager`):** Used in `src/crawl4ai_mcp.py` to manage the lifecycle of resources like the Crawl4AI crawler and the Qdrant client.

## Data Processing Flow
1.  URL is received by a crawl tool (`crawl_single_page` or `smart_crawl_url`).
2.  Crawl4AI fetches and processes the web content into Markdown.
3.  The Markdown is chunked into segments (target 750 chars) by `smart_chunk_markdown`.
4.  (Summarization step is currently bypassed).
5.  Each chunk is sent to the TEI server (`BAAI/bge-large-en-v1.5`) via `create_embeddings_batch` in `utils.py` to get a 1024-dimension vector.
6.  The chunk text and its embedding are stored as a point in the specified Qdrant collection using `store_embeddings`.
7.  For RAG queries, the query text is embedded using the same TEI server, and Qdrant is searched for similar vectors.

## Dependency Injection
- Passes context between components via request context

## Async Processing
- Utilizes async/await for non-blocking operations

## Batch Processing
- Processes documents in batches for efficiency

## Smart Chunking
- Respects code blocks, paragraphs, and sentences when splitting content

## Metadata Extraction
- Generates metadata for each chunk to improve filtering and retrieval 