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

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher
from utils import (
    get_qdrant_client, 
    ensure_qdrant_collection_async as async_ensure_collection_exists,
    store_embeddings,
    query_qdrant,
    get_available_sources as get_available_sources_async
)

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)

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

def smart_chunk_markdown(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
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
            # Chunk the content
            chunks_text = smart_chunk_markdown(result.markdown, chunk_size=750)
            
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
async def smart_crawl_url(ctx: Context, url: str, max_depth: int = 3, max_concurrent: int = 10, chunk_size: int = 5000) -> str:
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
        chunk_size: Maximum size of each content chunk in characters (default: 5000)
    
    Returns:
        JSON string with crawl summary and storage information
    """
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
            
            chunks_text = smart_chunk_markdown(markdown_content, chunk_size=750)
            
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