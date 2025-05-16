"""
MCP server for web crawling with Crawl4AI.

This server provides tools to crawl websites using Crawl4AI, automatically detecting
the appropriate crawl method based on URL type (sitemap, txt file, or regular webpage).
"""
from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, urldefrag
from xml.etree import ElementTree
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from pathlib import Path
import requests
import asyncio
import json
import os
import re
import tempfile
import subprocess

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher
from utils import (
    get_qdrant_client, 
    ensure_qdrant_collection_async as async_ensure_collection_exists,
    store_embeddings,
    query_qdrant,
    get_available_sources as get_available_sources_async,
    get_collection_stats
)

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)

# Chunking configuration from environment variables
# Regular text chunking (used for code files and general text)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
# Markdown chunking (specialized for preserving markdown structure like headers and code blocks)
MARKDOWN_CHUNK_SIZE = int(os.getenv("MARKDOWN_CHUNK_SIZE", "750"))

# Create a dataclass for our application context
@dataclass
class Crawl4AIContext:
    """Context for the Crawl4AI MCP server."""
    crawler: AsyncWebCrawler
    qdrant_client: QdrantClient
    collection_name: str
    
@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    """
    Manages the Crawl4AI client lifecycle.
    
    Args:
        server: The FastMCP server instance
        
    Yields:
        Crawl4AIContext: The context containing the Crawl4AI crawler and Qdrant client
    """
    # Create browser configuration
    browser_config = BrowserConfig(
        headless=True,
        verbose=False
    )
    
    # Initialize the crawler
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()
    
    # Initialize Qdrant client
    qdrant_client = get_qdrant_client()
    collection_name = os.getenv("QDRANT_COLLECTION", "crawled_pages")
    vector_dim = int(os.getenv("VECTOR_DIM", "1024")) # Default to 1024 if not set
    
    # Ensure collection exists
    await async_ensure_collection_exists(qdrant_client, collection_name, vector_dim)
    
    try:
        yield Crawl4AIContext(
            crawler=crawler,
            qdrant_client=qdrant_client,
            collection_name=collection_name
        )
    finally:
        # Clean up the crawler
        await crawler.__aexit__(None, None, None)

# Initialize FastMCP server
mcp = FastMCP(
    "mcp-crawl4ai-rag",
    description="MCP server for RAG and web crawling with Crawl4AI",
    lifespan=crawl4ai_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=os.getenv("PORT", "8051")
)

def is_sitemap(url: str) -> bool:
    """
    Check if a URL is a sitemap.
    
    Args:
        url: URL to check
        
    Returns:
        True if the URL is a sitemap, False otherwise
    """
    return url.endswith('sitemap.xml') or 'sitemap' in urlparse(url).path

def is_txt(url: str) -> bool:
    """
    Check if a URL is a text file.
    
    Args:
        url: URL to check
        
    Returns:
        True if the URL is a text file, False otherwise
    """
    return url.endswith('.txt')

def parse_sitemap(sitemap_url: str) -> List[str]:
    """
    Parse a sitemap and extract URLs.
    
    Args:
        sitemap_url: URL of the sitemap
        
    Returns:
        List of URLs found in the sitemap
    """
    resp = requests.get(sitemap_url)
    urls = []

    if resp.status_code == 200:
        try:
            tree = ElementTree.fromstring(resp.content)
            urls = [loc.text for loc in tree.findall('.//{*}loc')]
        except Exception as e:
            print(f"Error parsing sitemap XML: {e}")

    return urls

def smart_chunk_markdown(text: str, chunk_size: int = None) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    # Use environment variable or default if chunk_size not provided
    if chunk_size is None:
        chunk_size = MARKDOWN_CHUNK_SIZE

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = end

    return chunks

def extract_section_info(chunk: str) -> Dict[str, Any]:
    """
    Extracts headers and stats from a chunk.
    
    Args:
        chunk: Markdown chunk
        
    Returns:
        Dictionary with headers and stats
    """
    headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
    header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers]) if headers else ''

    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split())
    }

def simple_text_chunker(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
    """
    Splits text into chunks with overlap, suitable for general text or code.
    """
    # Use environment variables or defaults if not provided
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = CHUNK_OVERLAP

    chunks = []
    start = 0
    text_length = len(text)
    if text_length == 0:
        return []
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        
        if end == text_length:
            break
        
        start += (chunk_size - chunk_overlap)
        # Ensure progress if overlap is too large relative to chunk_size, or if start would not advance
        if start >= end: 
            start = end
            
    # Filter out empty or whitespace-only chunks that might result from chunking logic
    return [chunk for chunk in chunks if chunk and chunk.strip()]

@mcp.tool()
async def crawl_single_page(ctx: Context, url: str) -> str:
    """
    Crawl a single web page and store its content in Qdrant.
    
    This tool is ideal for quickly retrieving content from a specific URL without following links.
    The content is stored in Qdrant for later retrieval and querying.
    
    Args:
        ctx: The MCP server provided context
        url: URL of the web page to crawl
    
    Returns:
        Summary of the crawling operation and storage in Qdrant
    """
    try:
        # Get the crawler from the context
        crawler = ctx.request_context.lifespan_context.crawler
        qdrant_client = ctx.request_context.lifespan_context.qdrant_client
        collection_name = ctx.request_context.lifespan_context.collection_name
        
        # Configure the crawl
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        
        # Crawl the page
        result = await crawler.arun(url=url, config=run_config)
        
        if result.success and result.markdown:
            # Use the default markdown chunk size from environment variable
            chunks_text = smart_chunk_markdown(result.markdown)
            
            # Prepare chunk data for store_embeddings
            chunks_data_for_qdrant = []
            for i, chunk_content in enumerate(chunks_text):
                meta = extract_section_info(chunk_content)
                chunks_data_for_qdrant.append({
                    "text": chunk_content,
                    "headers": meta.get("headers", ""),
                })

            # Add to Qdrant using the new async store_embeddings
            successful_chunks, failed_chunks = await store_embeddings(
                client=qdrant_client,
                collection_name=collection_name,
                chunks=chunks_data_for_qdrant,
                source_url=url,
                crawl_type="single_page"
            )
            
            return json.dumps({
                "success": True,
                "url": url,
                "chunks_processed": len(chunks_text),
                "successful_chunks": successful_chunks,
                "failed_chunks": failed_chunks,
                "content_length": len(result.markdown),
                "links_count": {
                    "internal": len(result.links.get("internal", [])),
                    "external": len(result.links.get("external", []))
                }
            }, indent=2)
        else:
            return json.dumps({
                "success": False,
                "url": url,
                "error": result.error_message
            }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def crawl_repo(
    ctx: Context,
    repo_url: str,
    branch: Optional[str] = None,
    chunk_size: int = None,
    chunk_overlap: int = None
) -> str:
    """
    Clones a Git repository, processes specified file types, and stores their content in Qdrant.

    Args:
        ctx: The MCP server provided context.
        repo_url: URL of the Git repository to crawl.
        branch: Optional specific branch to clone. Defaults to the repository's default branch.
        chunk_size: Size of each text chunk in characters for processing.
        chunk_overlap: Overlap between text chunks in characters.

    Returns:
        JSON string with crawl summary and storage information.
    """
    # Use default values from environment variables if parameters are None
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = CHUNK_OVERLAP
    
    try:
        qdrant_client = ctx.request_context.lifespan_context.qdrant_client
        collection_name = ctx.request_context.lifespan_context.collection_name

        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir) / "cloned_repo"
            
            git_command = ["git", "clone", "--depth", "1"]
            if branch:
                git_command.extend(["--branch", branch])
            git_command.extend([repo_url, str(repo_path)])

            process = await asyncio.to_thread(
                subprocess.run, git_command, capture_output=True, text=True, check=False
            )

            if process.returncode != 0:
                error_message = process.stderr or process.stdout or "Unknown git clone error"
                return json.dumps({
                    "success": False,
                    "repo_url": repo_url,
                    "error": f"Git clone failed: {error_message.strip()}"
                }, indent=2)

            processed_files_count = 0
            total_successful_chunks = 0
            total_failed_chunks = 0
            files_with_errors = []

            # Loop over all files recursively
            for file_path_obj in repo_path.rglob("*"):
                if file_path_obj.is_file():
                    relative_path = file_path_obj.relative_to(repo_path)
                    try:
                        with open(file_path_obj, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        
                        if not content.strip():
                            continue
                        
                        # Use the new simple_text_chunker
                        current_file_chunks_text = simple_text_chunker(
                            content, 
                            chunk_size=chunk_size, 
                            chunk_overlap=chunk_overlap
                        )
                        
                        if not current_file_chunks_text:
                            continue

                        chunks_data_for_qdrant = []
                        for i, chunk_content in enumerate(current_file_chunks_text):
                            meta_info = f"File: {str(relative_path)} - Chunk: {i+1}/{len(current_file_chunks_text)}"
                            chunks_data_for_qdrant.append({
                                "text": chunk_content,
                                "headers": meta_info, 
                            })
                        
                        if chunks_data_for_qdrant:
                            successful, failed = await store_embeddings(
                                client=qdrant_client,
                                collection_name=collection_name,
                                chunks=chunks_data_for_qdrant,
                                source_url=f"{repo_url} (file: {str(relative_path)})",
                                crawl_type="repository"
                            )
                            total_successful_chunks += successful
                            total_failed_chunks += failed
                            processed_files_count += 1
                    
                    except Exception as e:
                        error_detail = f"Error processing file {str(relative_path)}: {str(e)}"
                        print(error_detail) # Log to server console
                        files_with_errors.append(str(relative_path))
            
            summary = {
                "success": True,
                "repo_url": repo_url,
                "branch_crawled": branch if branch else "default",
                "files_processed": processed_files_count,
                "total_successful_chunks": total_successful_chunks,
                "total_failed_chunks": total_failed_chunks
            }
            if files_with_errors:
                summary["files_with_errors"] = files_with_errors
                summary["error_count_in_files"] = len(files_with_errors)

            return json.dumps(summary, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "repo_url": repo_url,
            "error": f"An unexpected error occurred: {str(e)}"
        }, indent=2)

@mcp.tool()
async def smart_crawl_url(ctx: Context, url: str, max_depth: int = 3, max_concurrent: int = 10, chunk_size: int = None) -> str:
    """
    Intelligently crawl a URL based on its type and store content in Qdrant.
    
    This tool automatically detects the URL type and applies the appropriate crawling method:
    - For sitemaps: Extracts and crawls all URLs in parallel
    - For text files (llms.txt): Directly retrieves the content
    - For regular webpages: Recursively crawls internal links up to the specified depth
    
    All crawled content is chunked and stored in Qdrant for later retrieval and querying.
    
    Args:
        ctx: The MCP server provided context
        url: URL to crawl (can be a regular webpage, sitemap.xml, or .txt file)
        max_depth: Maximum recursion depth for regular URLs (default: 3)
        max_concurrent: Maximum number of concurrent browser sessions (default: 10)
        chunk_size: Maximum size of each content chunk in characters, defaults to MARKDOWN_CHUNK_SIZE

    Returns:
        JSON string with crawl summary and storage information
    """
    # Use markdown chunk size if not specified (since we're dealing with web content)
    if chunk_size is None:
        chunk_size = MARKDOWN_CHUNK_SIZE
    
    try:
        # Get the crawler and Qdrant client from the context
        crawler = ctx.request_context.lifespan_context.crawler
        qdrant_client = ctx.request_context.lifespan_context.qdrant_client
        collection_name = ctx.request_context.lifespan_context.collection_name
        
        crawl_results = []
        crawl_type = "webpage"
        
        # Detect URL type and use appropriate crawl method
        if is_txt(url):
            # For text files, use simple crawl
            crawl_results = await crawl_markdown_file(crawler, url)
            crawl_type = "text_file"
        elif is_sitemap(url):
            # For sitemaps, extract URLs and crawl in parallel
            sitemap_urls = parse_sitemap(url)
            if not sitemap_urls:
                return json.dumps({
                    "success": False,
                    "url": url,
                    "error": "No URLs found in sitemap"
                }, indent=2)
            crawl_results = await crawl_batch(crawler, sitemap_urls, max_concurrent=max_concurrent)
            crawl_type = "sitemap"
        else:
            # For regular URLs, use recursive crawl
            crawl_results = await crawl_recursive_internal_links(crawler, [url], max_depth=max_depth, max_concurrent=max_concurrent)
            crawl_type = "webpage"
        
        if not crawl_results:
            return json.dumps({
                "success": False,
                "url": url,
                "error": "No content found"
            }, indent=2)
        
        # Process results and store in Qdrant
        all_chunks_data_for_qdrant = []
        processed_urls = set()
        total_successful_chunks = 0
        total_failed_chunks = 0

        for doc_data in crawl_results:
            source_url = doc_data['url']
            markdown_content = doc_data['markdown']
            processed_urls.add(source_url)
            
            chunks_text = smart_chunk_markdown(markdown_content, chunk_size=chunk_size)
            
            current_page_chunks_data = []
            for i, chunk_content in enumerate(chunks_text):
                meta = extract_section_info(chunk_content)
                current_page_chunks_data.append({
                    "text": chunk_content,
                    "headers": meta.get("headers", ""),
                })
            
            # Store embeddings for the current page's chunks
            successful_chunks, failed_chunks = await store_embeddings(
                client=qdrant_client,
                collection_name=collection_name,
                chunks=current_page_chunks_data,
                source_url=source_url,
                crawl_type=crawl_type
            )
            total_successful_chunks += successful_chunks
            total_failed_chunks += failed_chunks
        
        return json.dumps({
            "success": True,
            "url": url,
            "crawl_type": crawl_type,
            "pages_crawled": len(processed_urls),
            "total_successful_chunks": total_successful_chunks,
            "total_failed_chunks": total_failed_chunks,
            "urls_crawled_sample": list(processed_urls)[:5] + (["..."] if len(processed_urls) > 5 else [])
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "url": url,
            "error": str(e)
        }, indent=2)

async def crawl_markdown_file(crawler: AsyncWebCrawler, url: str) -> List[Dict[str, Any]]:
    """
    Crawl a .txt or markdown file.
    
    Args:
        crawler: AsyncWebCrawler instance
        url: URL of the file
        
    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig()

    result = await crawler.arun(url=url, config=crawl_config)
    if result.success and result.markdown:
        return [{'url': url, 'markdown': result.markdown}]
    else:
        print(f"Failed to crawl {url}: {result.error_message}")
        return []

async def crawl_batch(crawler: AsyncWebCrawler, urls: List[str], max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """
    Batch crawl multiple URLs in parallel.
    
    Args:
        crawler: AsyncWebCrawler instance
        urls: List of URLs to crawl
        max_concurrent: Maximum number of concurrent browser sessions
        
    Returns:
        List of dictionaries with URL and markdown content
    """
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    results = await crawler.arun_many(urls=urls, config=crawl_config, dispatcher=dispatcher)
    return [{'url': r.url, 'markdown': r.markdown} for r in results if r.success and r.markdown]

async def crawl_recursive_internal_links(crawler: AsyncWebCrawler, start_urls: List[str], max_depth: int = 3, max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """
    Recursively crawl internal links from start URLs up to a maximum depth.
    
    Args:
        crawler: AsyncWebCrawler instance
        start_urls: List of starting URLs
        max_depth: Maximum recursion depth
        max_concurrent: Maximum number of concurrent browser sessions
        
    Returns:
        List of dictionaries with URL and markdown content
    """
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )

    visited = set()

    def normalize_url(url):
        return urldefrag(url)[0]

    current_urls = set([normalize_url(u) for u in start_urls])
    results_all = []

    for depth in range(max_depth):
        urls_to_crawl = [normalize_url(url) for url in current_urls if normalize_url(url) not in visited]
        if not urls_to_crawl:
            break

        results = await crawler.arun_many(urls=urls_to_crawl, config=run_config, dispatcher=dispatcher)
        next_level_urls = set()

        for result in results:
            norm_url = normalize_url(result.url)
            visited.add(norm_url)

            if result.success and result.markdown:
                results_all.append({'url': result.url, 'markdown': result.markdown})
                for link in result.links.get("internal", []):
                    next_url = normalize_url(link["href"])
                    if next_url not in visited:
                        next_level_urls.add(next_url)

        current_urls = next_level_urls

    return results_all

@mcp.tool()
async def get_available_sources(ctx: Context) -> str:
    """
    Get all available sources based on unique source metadata values.
    
    This tool returns a list of all unique sources (domains) that have been crawled and stored
    in the database. This is useful for discovering what content is available for querying.
    
    Args:
        ctx: The MCP server provided context
    
    Returns:
        JSON string with the list of available sources
    """
    try:
        # Get the Qdrant client from the context
        qdrant_client = ctx.request_context.lifespan_context.qdrant_client
        collection_name = ctx.request_context.lifespan_context.collection_name
        
        sources = await get_available_sources_async(qdrant_client, collection_name)
        
        return json.dumps({
            "success": True,
            "sources": sources,
            "count": len(sources)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def perform_rag_query(ctx: Context, query: str, source: str = None, match_count: int = 5) -> str:
    """
    Perform a RAG (Retrieval Augmented Generation) query on the stored content.
    
    This tool searches the vector database for content relevant to the query and returns
    the matching documents. Optionally filter by source domain.

    Use the tool to get source domains if the user is asking to use a specific tool or framework.
    
    Args:
        ctx: The MCP server provided context
        query: The search query
        source: Optional source domain to filter results (e.g., 'example.com')
        match_count: Maximum number of results to return (default: 5)
    
    Returns:
        JSON string with the search results
    """
    try:
        # Get the Qdrant client from the context
        qdrant_client = ctx.request_context.lifespan_context.qdrant_client
        collection_name = ctx.request_context.lifespan_context.collection_name
        
        # Perform the search using the new async query_qdrant
        results = await query_qdrant(
            client=qdrant_client,
            collection_name=collection_name,
            query_text=query,
            source_filter=source if source and source.strip() else None,
            match_count=match_count
        )
        
        # Results from query_qdrant are already well-formatted
        return json.dumps({
            "success": True,
            "query": query,
            "source_filter": source,
            "results": results,
            "count": len(results)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "query": query,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def perform_hybrid_search(
    ctx: Context, 
    query: str, 
    filter_text: str = None, 
    vector_weight: float = 0.7, 
    keyword_weight: float = 0.3, 
    source: str = None, 
    match_count: int = 5
) -> str:
    """
    Perform a hybrid search combining vector similarity with keyword/text-based filtering.
    
    This tool enhances search by combining semantic vector search with keyword filtering,
    allowing for more precise and flexible querying of the vector database.
    
    Args:
        ctx: The MCP server provided context
        query: The query text for semantic vector search
        filter_text: Optional keyword text for filtering (text search)
        vector_weight: Weight for vector results (0.0-1.0, default: 0.7)
        keyword_weight: Weight for keyword results (0.0-1.0, default: 0.3)
        source: Optional source domain to filter results (e.g., 'example.com')
        match_count: Maximum number of results to return (default: 5)
    
    Returns:
        JSON string with the hybrid search results
    """
    try:
        # Get the Qdrant client from the context
        qdrant_client = ctx.request_context.lifespan_context.qdrant_client
        collection_name = ctx.request_context.lifespan_context.collection_name
        
        # Validate weights
        if not (0.0 <= vector_weight <= 1.0) or not (0.0 <= keyword_weight <= 1.0):
            return json.dumps({
                "success": False,
                "query": query,
                "error": "Weights must be between 0.0 and 1.0"
            }, indent=2)
        
        # Perform the hybrid search
        results = await perform_hybrid_search(
            client=qdrant_client,
            collection_name=collection_name,
            query_text=query,
            filter_text=filter_text,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
            source_filter=source if source and source.strip() else None,
            match_count=match_count
        )
        
        return json.dumps({
            "success": True,
            "query": query,
            "filter_text": filter_text,
            "weights": {
                "vector": vector_weight,
                "keyword": keyword_weight
            },
            "source_filter": source,
            "results": results,
            "count": len(results)
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "query": query,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def get_collection_statistics(
    ctx: Context,
    collection_name: str = None,
    include_segments: bool = False
) -> str:
    """
    Get statistics about Qdrant collections and their usage.
    
    This tool provides a dashboard of collection metrics including size, point count,
    and configurations. It can target a specific collection or all collections.
    
    Args:
        ctx: The MCP server provided context
        collection_name: Optional name of a specific collection to analyze
                        (if None, all collections are analyzed)
        include_segments: Whether to include detailed segment-level information
    
    Returns:
        JSON string with collection statistics
    """
    try:
        # Get the Qdrant client from the context
        qdrant_client = ctx.request_context.lifespan_context.qdrant_client
        
        # Use default collection from context if none specified
        if collection_name is None or collection_name.strip() == "":
            default_collection = ctx.request_context.lifespan_context.collection_name
            print(f"No collection specified, using default: {default_collection}")
            
            # If 'all' is explicitly requested, set to None to get all collections
            if default_collection.lower() == "all":
                collection_name = None
            else:
                collection_name = default_collection
                
        # Get collection stats
        stats = await get_collection_stats(
            client=qdrant_client,
            collection_name=collection_name,
            include_segments=include_segments
        )
        
        # Format human-readable summary
        if stats.get("success", False):
            summary = {
                "success": True,
                "timestamp": stats["timestamp"],
                "summary": {
                    "total_collections": stats["total_collections"],
                    "total_vectors": stats["total_vectors"],
                    "collections_analyzed": [c["name"] for c in stats["collections"]]
                },
                "collections": stats["collections"]
            }
            
            # Add a human-readable section
            collection_summaries = []
            for coll in stats["collections"]:
                if "error" in coll:
                    collection_summaries.append(f"- {coll['name']}: ERROR - {coll['error']}")
                else:
                    collection_summaries.append(
                        f"- {coll['name']}: {coll.get('vectors_count', 'N/A')} vectors, "
                        f"status: {coll.get('status', 'N/A')}, "
                        f"{coll.get('cluster_info', {}).get('peer_count', 1)} peers, "
                        f"{coll.get('cluster_info', {}).get('shard_count', 1)} shards"
                    )
            
            collection_list = "\n".join(collection_summaries)
            summary["human_readable"] = f"""
Collection Statistics Summary:
- Total Collections: {stats['total_collections']}
- Total Vectors: {stats['total_vectors']}

Collections:
{collection_list}
            """.strip()
            
            return json.dumps(summary, indent=2)
        else:
            return json.dumps(stats, indent=2)
            
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)

@mcp.tool()
async def item_recommendations(
    ctx: Context,
    item_id: str = None,
    content: str = None, 
    source: str = None,
    match_count: int = 5
) -> str:
    """
    Find similar items based on vector similarity using Qdrant's recommendation API.
    
    This tool can find similar content in two ways:
    1. Based on an existing item ID in the database
    2. Based on a piece of text provided directly
    
    At least one of item_id or content must be provided.
    
    Args:
        ctx: The MCP server provided context
        item_id: Optional ID of an existing item to find recommendations for
        content: Optional text content to find similar items for
        source: Optional source domain to filter results (e.g., 'example.com')
        match_count: Maximum number of recommendations to return (default: 5)
    
    Returns:
        JSON string with recommendation results
    """
    try:
        # Get the Qdrant client from the context
        qdrant_client = ctx.request_context.lifespan_context.qdrant_client
        collection_name = ctx.request_context.lifespan_context.collection_name
        
        # Ensure at least one of item_id or content is provided
        if not item_id and not content:
            return json.dumps({
                "success": False,
                "error": "Either item_id or content must be provided"
            }, indent=2)
        
        # Prepare source filter if provided
        filter_condition = None
        if source and source.strip():
            filter_condition = {"source": source}
        
        # Recommendation based on item ID
        if item_id:
            # First fetch the original item for context
            original_item = await fetch_item_by_id(
                client=qdrant_client,
                collection_name=collection_name,
                item_id=item_id
            )
            
            if original_item is None:
                return json.dumps({
                    "success": False,
                    "error": f"No item found with ID: {item_id}"
                }, indent=2)
            
            # Get similar items
            similar_items = await get_similar_items(
                client=qdrant_client,
                collection_name=collection_name,
                item_id=item_id,
                filter_condition=filter_condition,
                match_count=match_count
            )
            
            return json.dumps({
                "success": True,
                "recommendation_type": "item-based",
                "original_item": original_item,
                "source_filter": source,
                "recommendations": similar_items,
                "count": len(similar_items)
            }, indent=2)
        
        # Recommendation based on content text
        else:
            # Get similar content
            similar_items = await find_similar_content(
                client=qdrant_client,
                collection_name=collection_name,
                content_text=content,
                filter_condition=filter_condition,
                match_count=match_count
            )
            
            # Calculate a preview of the query content (truncated)
            content_preview = content
            if len(content) > 150:
                content_preview = content[:150] + "..."
            
            return json.dumps({
                "success": True,
                "recommendation_type": "content-based",
                "query_content_preview": content_preview,
                "source_filter": source,
                "recommendations": similar_items,
                "count": len(similar_items)
            }, indent=2)
    
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)

async def main():
    transport = os.getenv("TRANSPORT", "sse")
    if transport == 'sse':
        # Run the MCP server with sse transport
        await mcp.run_sse_async()
    else:
        # Run the MCP server with stdio transport
        await mcp.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())