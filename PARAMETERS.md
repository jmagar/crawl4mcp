# MCP Tool Parameters

This document lists all available MCP tools and their respective parameters.

## Analytics Tools (`analytics_tools.py`)

### `cluster_content`
Clusters stored content into semantically similar groups.
-   `source_filter` (Optional[str]): Filter content by source.
-   `num_clusters` (Optional[int]): The number of clusters to create. If not provided, an optimal number will be determined.
-   `sample_size` (Optional[int]): The number of vectors to sample for clustering. Defaults to 500 if not provided.
-   `include_visualization` (bool): Whether to include data for visualization in the output. Defaults to `False`.

## Crawling Tools (`crawling_tools.py`)

### `crawl_single_page`
Crawls a single web page and stores its content in Qdrant.
-   `url` (str): URL of the web page to crawl.

### `crawl_repo`
Clones a Git repository, processes specified file types, and stores their content in Qdrant.
-   `repo_url` (str): URL of the Git repository to crawl.
-   `branch` (Optional[str]): Specific branch to clone. Defaults to the repository's default branch.
-   `chunk_size` (Optional[int]): Size of each text chunk in characters.
-   `chunk_overlap` (Optional[int]): Overlap between text chunks in characters.
-   `ignore_dirs` (Optional[List[str]]): List of directory names or path patterns to ignore.

### `smart_crawl_url`

### `crawl_dir`
Crawls a local directory, processes specified file types, and stores their content in Qdrant.
-   `dir_path` (str): Absolute path to the local directory to crawl.
-   `chunk_size` (Optional[int]): Size of each text chunk in characters. Defaults to `CHUNK_SIZE` from environment or crawling_utils.
-   `chunk_overlap` (Optional[int]): Overlap between text chunks in characters. Defaults to `CHUNK_OVERLAP` from environment or crawling_utils.
-   `ignore_patterns` (Optional[List[str]]): List of glob-style patterns for files/directories to ignore (e.g., `["*.log", "temp/*"]`). Defaults to a predefined list including common temporary/build/cache directories.
-   `allowed_extensions` (Optional[List[str]]): List of file extensions (e.g., `['.py', '.md']`) to process. Defaults to a predefined list of common text and code file types.

### `smart_crawl_url`
Intelligently crawls a URL based on its type and stores content in Qdrant.
-   `url` (str): URL to crawl.
-   `max_depth` (int): Maximum depth for recursive crawls (default: 3).
-   `max_concurrent` (int): Maximum concurrent requests (default: 30).
-   `chunk_size` (Optional[int]): Max size of each markdown content chunk.

## Management Tools (`management_tools.py`)

### `get_available_sources`
Gets all available sources based on unique source metadata values in the Qdrant collection.
-   No parameters.

### `get_collection_stats`
Gets statistics about a Qdrant collection or all collections.
-   `collection_name` (Optional[str]): The name of the collection to get stats for.
-   `include_segments` (bool): Whether to include detailed segment information. Defaults to `False`.

### `view_server_logs`
Retrieves the last N lines from the server's log file.
-   `num_lines` (int): The number of log lines to retrieve. Defaults to 150.

## Retrieval Tools (`retrieval_tools.py`)

### `perform_rag_query`
Performs a RAG (Retrieval Augmented Generation) query on the stored content.
-   `query` (str): The search query.
-   `source` (Optional[str]): Optional source domain to filter results.
-   `match_count` (int): Maximum number of results to return (default: 5).

### `perform_hybrid_search`
Performs a hybrid search combining vector similarity with keyword/text-based filtering.
-   `query` (str): The search query.
-   `filter_text` (Optional[str]): Text to use for keyword filtering.
-   `vector_weight` (float): Weight for vector similarity (0.0 to 1.0). Default: 0.7.
-   `keyword_weight` (float): Weight for keyword matching (0.0 to 1.0). Default: 0.3.
-   `source` (Optional[str]): Optional source to filter results.
-   `match_count` (int): Maximum number of results to return (default: 5).

### `get_similar_items`
Finds similar items based on vector similarity using Qdrant's recommendation API, given an existing item's ID.
-   `item_id` (str): The ID of the item to find similar items for.
-   `filter_source` (Optional[str]): Optional source to filter results.
-   `match_count` (int): Maximum number of results to return (default: 5).

### `fetch_item_by_id`
Fetches a specific item by its ID from the Qdrant collection.
-   `item_id` (str): The ID of the item to fetch.

### `find_similar_content`
Finds similar content based on a given text, not an existing item ID.
-   `content_text` (str): The text to find similar content for.
-   `filter_source` (Optional[str]): Optional source to filter results.
-   `match_count` (int): Maximum number of results to return (default: 5).