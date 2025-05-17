# System Architecture and Patterns

## How the system is built
The system is an MCP (Model Context Protocol) server designed for advanced web crawling and Retrieval Augmented Generation (RAG). It leverages:
- **Crawl4AI:** For intelligent, LLM-friendly web scraping and content extraction (Markdown format).
- **Self-Hosted TEI Server:** Running `BAAI/bge-large-en-v1.5` (1024 dimensions) for generating text embeddings locally.
- **Qdrant:** An open-source vector database used for storing text chunks and their corresponding embeddings, enabling semantic search.
- **FastMCP Framework:** For implementing the MCP server and its tools.
- **Scikit-learn (Optional):** For vector clustering, dimensionality reduction, and other analytical functions.
- **Plotly (Optional):** For interactive visualizations of vector spaces and clusters.

It follows a modular architecture with distinct components for crawling, embedding, storage, querying, and analytics.

## Key technical decisions
- **Crawl4AI for Web Scraping:** Chosen for its robust features in handling various web content types, JavaScript rendering, and producing clean Markdown suitable for LLMs.
- **Local Embeddings via TEI:** `BAAI/bge-large-en-v1.5` is used for local embedding generation. This provides control over the embedding process and avoids reliance on third-party APIs for this core step.
- **Qdrant for Vector Storage:** Selected as a scalable and efficient open-source vector database for RAG.
- **Character-Based Chunking:** Content is chunked using `smart_chunk_markdown` with a target of 750 characters, aiming to balance semantic coherence with the TEI server's payload limits and the BGE model's token capacity.
- **Contextual Summarization (Currently Disabled):** The capability to use an OpenAI model (e.g., `gpt-4o`) for query-focused summaries before embedding exists but is currently disabled due to previous TEI payload issues. If re-enabled, strict output length control would be necessary.
- **HTTP for TEI Communication:** The `requests` library is used for synchronous communication with the TEI server's `/embed` endpoint.
- **Hybrid Search Implementation:** Combines vector similarity with keyword/text-based filtering for more precise queries.
- **Optional Visualization Dependencies:** Scikit-learn, plotly, numpy, and nltk are included as optional dependencies for visualization and analytics.
- **Standardized Helper Functions:** Created reusable helpers for common operations like filter creation and result formatting.

## Architecture patterns
- **MCP Server:** Exposes crawling and RAG functionalities as tools callable via the Model Context Protocol.
- **Service Integration:** Interacts with the local TEI server and the Qdrant database instance via network calls.
- **Environment-Driven Configuration:** Critical parameters (Qdrant URL/API key, TEI server URL, collection names, vector dimensions, optional OpenAI keys) are managed via a `.env` file.
- **Dockerized Deployment:** The MCP server is packaged as a Docker container, and `docker-compose.yml` is provided to orchestrate it (though the TEI and Qdrant services are expected to be managed externally or via separate compose definitions based on current setup).
- **Asynchronous Operations:** Core server logic and tool implementations in `src/crawl4ai_mcp.py` and `src/utils.py` heavily utilize `async/await` for non-blocking I/O, especially for crawling and Qdrant interactions.
- **Lifespan Management (`asynccontextmanager`):** Used in `src/crawl4ai_mcp.py` to manage the lifecycle of resources like the Crawl4AI crawler and the Qdrant client.
- **Helper Function Pattern:** Common operations like filter creation and result formatting are abstracted into helper functions for code reuse and consistency.
- **Error Handling Pattern:** Standardized error handling through dedicated functions like `handle_search_error`.
- **Bivvy Climb Pattern:** Features are developed and tracked through "bivvy climbs" with descriptive JSON files recording the status of each "move".

## Data Processing Flow
1.  URL is received by a crawl tool (`crawl_single_page` or `smart_crawl_url`).
2.  Crawl4AI fetches and processes the web content into Markdown.
3.  The Markdown is chunked into segments (target 750 chars) by `smart_chunk_markdown`.
4.  (Summarization step is currently bypassed).
5.  Each chunk is sent to the TEI server (`BAAI/bge-large-en-v1.5`) via `create_embeddings_batch` in `utils.py` to get a 1024-dimension vector.
6.  The chunk text and its embedding are stored as a point in the specified Qdrant collection using `store_embeddings`.
7.  For RAG queries, the query text is embedded using the same TEI server, and Qdrant is searched for similar vectors.
8.  For hybrid search, both vector similarity and keyword matching are combined with configurable weights.
9.  For clustering, vectors are fetched from Qdrant and processed using KMeans and TSNE for dimensionality reduction and visualization.
10. For recommendations, similar items are found using Qdrant's recommendation API or by embedding and searching for similar content.

## Advanced Functionality
- **Create Qdrant Filter:** `create_qdrant_filter` in utils.py standardizes filter creation for Qdrant queries.
- **Format Search Results:** `format_search_result` processes raw Qdrant results into a structured, consistent format.
- **Handle Search Errors:** `handle_search_error` provides consistent error handling for search operations.
- **Enhance Payload Metadata:** `enhance_payload_metadata` adds additional metadata to improve result context.
- **Vector Clustering:** Using KMeans from scikit-learn to group similar vectors and identify patterns.
- **Dimensionality Reduction:** TSNE is used to reduce high-dimensional vectors for visualization.
- **Interactive Visualizations:** Plotly generates interactive visualizations for exploring vector space.

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