"""
MCP Server for Crawl4AI RAG
Implements the approved tool set for web crawling and RAG.
Built with FastMCP 2.0 following best practices from gofastmcp.com
Transport: Streamable HTTP
"""

import os
import sys
import asyncio
import json
import re
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, AsyncIterator # AsyncIterator for potential future streaming tools

from dotenv import load_dotenv
from qdrant_client import QdrantClient
import requests # For parse_sitemap, consider making async if it becomes a bottleneck
from xml.etree import ElementTree

from fastmcp import FastMCP, Context
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher

# utils.py is expected to be in the same directory or Python path
import utils # Assuming utils.py is in src/

# --- Logging Setup ---
LOG_LEVEL_STR = os.getenv('LOG_LEVEL', 'INFO').upper()
NUMERIC_LOG_LEVEL = getattr(logging, LOG_LEVEL_STR, logging.INFO)
SCRIPT_DIR = Path(__file__).resolve().parent

# Define a base logger
logger = logging.getLogger("Crawl4AIMCPServer")
logger.setLevel(NUMERIC_LOG_LEVEL)
logger.propagate = False

# Console Handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(NUMERIC_LOG_LEVEL)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
if not logger.handlers: # Add handlers only if they haven't been added (e.g. during multiple imports/runs)
    logger.addHandler(console_handler)

    # File Handler with Rotation
    log_file_name = f"{os.getenv('CRAWL4AI_MCP_NAME', 'crawl4ai_mcp').lower()}.log"
    log_file_path = SCRIPT_DIR / log_file_name
    file_handler = RotatingFileHandler(log_file_path, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    file_handler.setLevel(NUMERIC_LOG_LEVEL)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(lineno)d - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

logger.info(f"Logging initialized (console and file: {log_file_path}). Level: {LOG_LEVEL_STR}")

# --- Environment Variable Loading & Validation ---
# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'
load_dotenv(dotenv_path, override=True)
logger.info(f"Attempted to load .env from {dotenv_path}")

# Log key loaded variables
logger.info(f"QDRANT_URL loaded: {os.getenv('QDRANT_URL', 'Not Found')[:30]}...")
logger.info(f"QDRANT_API_KEY loaded: {'****' if os.getenv('QDRANT_API_KEY') else 'Not Found'}")
logger.info(f"QDRANT_COLLECTION loaded: {os.getenv('QDRANT_COLLECTION', 'crawled_pages')}")
logger.info(f"EMBEDDING_SERVER_URL loaded: {os.getenv('EMBEDDING_SERVER_URL', 'Not Found')[:30]}...")
logger.info(f"EMBEDDING_SERVER_BATCH_SIZE loaded: {utils.EMBEDDING_SERVER_BATCH_SIZE}") # From utils
logger.info(f"VECTOR_DIM loaded: {os.getenv('VECTOR_DIM', '1024')}")
logger.info(f"MCP_HOST set to: {os.getenv('MCP_HOST', '127.0.0.1')}")
logger.info(f"MCP_PORT set to: {os.getenv('MCP_PORT', '8051')}")

# Critical check for essential API credentials/URL
if not os.getenv('QDRANT_URL') or not os.getenv('EMBEDDING_SERVER_URL'):
    logger.error("QDRANT_URL and EMBEDDING_SERVER_URL must be set in environment variables. Exiting.")
    sys.exit(1)

# --- Application Context & Lifespan ---
@dataclass
class Crawl4AIAppContext:
    """Context for the Crawl4AI MCP server, managed by lifespan."""
    crawler: AsyncWebCrawler
    qdrant_client: QdrantClient
    collection_name: str
    vector_dim: int

@asynccontextmanager
async def app_lifespan(app: FastMCP) -> AsyncIterator[Dict[str, Crawl4AIAppContext]]: # Adjusted for FastMCP 2.0 style if it uses app object
    """Manages the Crawl4AI client and Qdrant client lifecycle."""
    logger.info("Application lifespan startup: Initializing resources...")
    browser_config = BrowserConfig(headless=True, verbose=False) # Consider making verbose configurable
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()
    
    qdrant_client = utils.get_qdrant_client()
    collection_name = os.getenv("QDRANT_COLLECTION", "crawled_pages")
    vector_dim = int(os.getenv("VECTOR_DIM", "1024"))
    
    try:
        await utils.ensure_qdrant_collection_async(qdrant_client, collection_name, vector_dim)
        logger.info(f"Qdrant collection '{collection_name}' ensured with dimension {vector_dim}.")
        
        app_context = Crawl4AIAppContext(
            crawler=crawler,
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            vector_dim=vector_dim
        )
        # For FastMCP 2.0, lifespan yields a dict to be merged into app.state if app is Starlette-like
        # The yielded dictionary key is how tools will access it via ctx.app.state.YOUR_KEY
        # setattr(app, 'app_state', app_context) # REMOVE THIS LINE - app is FastMCP instance, not Starlette app here.
                                               # The yield below handles context for Starlette.
        yield {"crawl4ai_app_context": app_context} 
        logger.info("Application context yielded.")

    except Exception as e:
        logger.error(f"Error during application startup: {e}", exc_info=True)
        # Ensure cleanup happens even if startup fails mid-way
        await crawler.__aexit__(None, None, None)
        if hasattr(qdrant_client, 'close'):
             qdrant_client.close()
        raise # Re-raise the exception to prevent server from starting in a bad state
    finally:
        logger.info("Application lifespan shutdown: Cleaning up resources...")
        await crawler.__aexit__(None, None, None)
        if hasattr(qdrant_client, 'close'):
            logger.info("Closing Qdrant client.")
            qdrant_client.close() # Ensure Qdrant client is closed if it has a close method
        logger.info("Application shutdown complete.")

# --- Initialize FastMCP Server ---
mcp = FastMCP(
    name="Crawl4AI-RAG-MCP-Server",
    description="MCP server for RAG and web crawling with Crawl4AI. Uses FastMCP 2.0.",
    lifespan=app_lifespan # Correctly pass the async context manager
)

# --- Helper functions (adapted from existing crawl4ai_mcp.py) ---
# These are kept in the server file if they are tightly coupled with tool logic
# or could be moved to utils.py if more general.

def is_sitemap(url: str) -> bool:
    return url.endswith('sitemap.xml') or 'sitemap' in utils.urlparse(url).path # Use utils.urlparse

def is_txt(url: str) -> bool:
    return url.endswith('.txt')

def parse_sitemap(sitemap_url: str) -> List[str]:
    logger.debug(f"Parsing sitemap: {sitemap_url}")
    urls = []
    try:
        # Using requests synchronously for now. Consider async HTTP client for high-volume sitemap parsing.
        resp = requests.get(sitemap_url, timeout=10)
        resp.raise_for_status()
        tree = ElementTree.fromstring(resp.content)
        urls = [loc.text for loc in tree.findall('.//{*}loc') if loc.text]
        logger.info(f"Found {len(urls)} URLs in sitemap {sitemap_url}")
    except requests.RequestException as e:
        logger.error(f"Error fetching sitemap {sitemap_url}: {e}")
    except ElementTree.ParseError as e:
        logger.error(f"Error parsing sitemap XML for {sitemap_url}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error parsing sitemap {sitemap_url}: {e}", exc_info=True)
    return urls

def smart_chunk_markdown(text: str, chunk_size: int = 750) -> List[str]:
    # This is a direct copy from the original file, ensure it meets requirements
    # Consider placing in utils.py if it's a general utility
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        if end >= text_length:
            chunks.append(text[start:].strip())
            break
        chunk_text_to_search = text[start:end]
        # Try to break at a code block boundary first (```)
        code_block_idx = chunk_text_to_search.rfind('```')
        # Try to break at double newline (paragraph)
        paragraph_idx = chunk_text_to_search.rfind('\n\n')
        # Try to break at sentence
        sentence_idx = -1
        if '. ' in chunk_text_to_search:
            sentence_idx = chunk_text_to_search.rfind('. ') + 1 # Include the period and space
        
        # Prefer code block, then paragraph, then sentence, if they are reasonably sized
        if code_block_idx > chunk_size * 0.3: # prefer if past 30% of chunk
            end = start + code_block_idx
        elif paragraph_idx > chunk_size * 0.3:
            end = start + paragraph_idx
        elif sentence_idx > chunk_size * 0.3:
            end = start + sentence_idx
            
        final_chunk = text[start:end].strip()
        if final_chunk: # Avoid adding empty chunks
            chunks.append(final_chunk)
        start = end
    logger.debug(f"Chunked text of length {text_length} into {len(chunks)} chunks with target size {chunk_size}.")
    return chunks

def extract_section_info(chunk: str) -> Dict[str, Any]:
    # This is a direct copy from the original file
    # Consider placing in utils.py
    headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
    header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers]) if headers else ''
    return {"headers": header_str, "char_count": len(chunk), "word_count": len(chunk.split())}

# --- Core Tool Definitions ---

@mcp.tool()
async def crawl_single_page(ctx: Context, url: str) -> Dict[str, Any]:
    """
    Crawls a single web page, extracts content, chunks, embeds, and stores in Qdrant.
    Returns a JSON summary of the operation.
    """
    logger.info(f"Tool 'crawl_single_page' called with URL: {url}")
    app_s: Crawl4AIAppContext = ctx.app.state.crawl4ai_app_context
    
    try:
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False) # stream=False for page content
        logger.debug(f"Crawling {url} with config: {run_config}")
        result = await app_s.crawler.arun(url=url, config=run_config)
        
        if result.success and result.markdown:
            logger.info(f"Successfully crawled {url}, content length: {len(result.markdown)}")
            chunks_text = smart_chunk_markdown(result.markdown, chunk_size=750) # Using agreed 750
            logger.debug(f"Content from {url} split into {len(chunks_text)} chunks.")

            chunks_data_for_qdrant = []
            for i, chunk_content in enumerate(chunks_text):
                meta = extract_section_info(chunk_content)
                chunks_data_for_qdrant.append({"text": chunk_content, "headers": meta.get("headers", "")})

            successful_chunks, failed_chunks = await utils.store_embeddings(
                client=app_s.qdrant_client,
                collection_name=app_s.collection_name,
                chunks=chunks_data_for_qdrant,
                source_url=url,
                crawl_type="single_page"
            )
            logger.info(f"Stored embeddings for {url}: {successful_chunks} successful, {failed_chunks} failed.")
            
            return {
                "success": True,
                "url": url,
                "chunks_processed": len(chunks_text),
                "successful_chunks_stored": successful_chunks,
                "failed_chunks_stored": failed_chunks,
                "content_length": len(result.markdown),
                "links_found": {
                    "internal": len(result.links.get("internal", [])),
                    "external": len(result.links.get("external", []))
                }
            }
        else:
            logger.error(f"Failed to crawl {url}: {result.error_message}")
            return {"success": False, "url": url, "error": result.error_message or "Unknown crawl error"}
    except Exception as e:
        logger.error(f"Exception in 'crawl_single_page' for {url}: {e}", exc_info=True)
        return {"success": False, "url": url, "error": str(e)}

async def _crawl_markdown_file_content(app_context: Crawl4AIAppContext, url: str) -> List[Dict[str, Any]]:
    """Helper to crawl a single .txt or markdown file for smart_crawl_url."""
    logger.debug(f"Crawling text/markdown file: {url}")
    # page_timeout increased for potentially large text files.
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False, page_timeout=180000)
    result = await app_context.crawler.arun(url=url, config=crawl_config)
    if result.success and result.markdown:
        logger.info(f"Successfully retrieved content from text file {url}, length {len(result.markdown)}")
        return [{'url': url, 'markdown': result.markdown}]
    else:
        logger.warning(f"Failed to crawl text file {url}: {result.error_message}")
        return []

async def _crawl_batch_urls(app_context: Crawl4AIAppContext, urls: List[str], max_concurrent: int) -> List[Dict[str, Any]]:
    """Helper to batch crawl multiple URLs for smart_crawl_url (typically from sitemap)."""
    if not urls: return []
    logger.debug(f"Batch crawling {len(urls)} URLs with max_concurrent={max_concurrent}.")
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False, page_timeout=180000)
    # Using MemoryAdaptiveDispatcher as in the original code
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0, # Consider making configurable
        check_interval=1.0,
        max_session_permit=max_concurrent
    )
    results = await app_context.crawler.arun_many(urls=urls, config=crawl_config, dispatcher=dispatcher)
    valid_results = [{'url': r.url, 'markdown': r.markdown} for r in results if r.success and r.markdown]
    logger.info(f"Batch crawl yielded {len(valid_results)} successful pages from {len(urls)} URLs.")
    return valid_results

async def _crawl_recursive_internal(app_context: Crawl4AIAppContext, start_urls: List[str], max_depth: int, max_concurrent: int) -> List[Dict[str, Any]]:
    """Helper for recursive internal link crawling for smart_crawl_url."""
    logger.debug(f"Recursive crawl starting from {start_urls} with max_depth={max_depth}, max_concurrent={max_concurrent}.")
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False, page_timeout=180000)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0, check_interval=1.0, max_session_permit=max_concurrent
    )
    visited = set()
    all_results_data = []
    
    # Normalize URLs using urldefrag as in original code
    current_urls_to_crawl = set([utils.urldefrag(u)[0] for u in start_urls])

    for depth in range(max_depth):
        crawl_batch_list = [url for url in current_urls_to_crawl if url not in visited]
        if not crawl_batch_list:
            logger.debug(f"Recursive crawl: No new URLs to crawl at depth {depth + 1}.")
            break
        
        logger.info(f"Recursive crawl depth {depth + 1}: Attempting to crawl {len(crawl_batch_list)} URLs.")
        results = await app_context.crawler.arun_many(urls=crawl_batch_list, config=run_config, dispatcher=dispatcher)
        
        next_level_urls_to_visit = set()
        for result in results:
            norm_url = utils.urldefrag(result.url)[0]
            visited.add(norm_url)
            if result.success and result.markdown:
                all_results_data.append({'url': result.url, 'markdown': result.markdown})
                if result.links and "internal" in result.links:
                    for link_info in result.links["internal"]:
                        next_url_candidate = utils.urldefrag(link_info["href"])[0]
                        if next_url_candidate not in visited:
                            next_level_urls_to_visit.add(next_url_candidate)
        current_urls_to_crawl = next_level_urls_to_visit
        logger.debug(f"Recursive crawl depth {depth + 1}: Found {len(all_results_data)} total pages so far, {len(current_urls_to_crawl)} URLs for next depth.")

    logger.info(f"Recursive crawl finished. Total pages collected: {len(all_results_data)}.")
    return all_results_data

@mcp.tool()
async def smart_crawl_url(ctx: Context, url: str, max_depth: int = 3, max_concurrent: int = 10, chunk_size: int = 750) -> Dict[str, Any]:
    """
    Intelligently crawls a URL (webpage, sitemap, or .txt file), processes content, 
    and stores embeddings in Qdrant. Returns a JSON summary.
    Default chunk_size for smart_chunk_markdown is 750.
    """
    logger.info(f"Tool 'smart_crawl_url' called with URL: {url}, max_depth: {max_depth}, max_concurrent: {max_concurrent}, chunk_size: {chunk_size}")
    app_s: Crawl4AIAppContext = ctx.app.state.crawl4ai_app_context
    
    try:
        crawl_results_data = [] # List of {'url': ..., 'markdown': ...}
        crawl_type = "unknown"

        if is_txt(url):
            crawl_type = "text_file"
            crawl_results_data = await _crawl_markdown_file_content(app_s, url)
        elif is_sitemap(url):
            crawl_type = "sitemap"
            sitemap_urls = parse_sitemap(url)
            if not sitemap_urls:
                logger.warning(f"No URLs found in sitemap: {url}")
                return {"success": False, "url": url, "crawl_type": crawl_type, "error": "No URLs found in sitemap"}
            crawl_results_data = await _crawl_batch_urls(app_s, sitemap_urls, max_concurrent)
        else:
            crawl_type = "webpage"
            crawl_results_data = await _crawl_recursive_internal(app_s, [url], max_depth, max_concurrent)

        if not crawl_results_data:
            logger.warning(f"No content successfully crawled for {url} (type: {crawl_type})")
            return {"success": False, "url": url, "crawl_type": crawl_type, "error": "No content found or all crawl attempts failed"}

        total_successful_chunks_stored = 0
        total_failed_chunks_stored = 0
        processed_urls_set = set()

        for page_data in crawl_results_data:
            page_url = page_data['url']
            markdown_content = page_data['markdown']
            processed_urls_set.add(page_url)
            
            chunks_text = smart_chunk_markdown(markdown_content, chunk_size=chunk_size) # Use tool's chunk_size
            logger.debug(f"Page {page_url}: Split into {len(chunks_text)} chunks with target size {chunk_size}.")

            page_chunks_for_qdrant = [{"text": ct, "headers": extract_section_info(ct).get("headers","")} for ct in chunks_text]
            
            successful, failed = await utils.store_embeddings(
                client=app_s.qdrant_client,
                collection_name=app_s.collection_name,
                chunks=page_chunks_for_qdrant,
                source_url=page_url,
                crawl_type=crawl_type # Pass the determined crawl_type
            )
            total_successful_chunks_stored += successful
            total_failed_chunks_stored += failed
        
        logger.info(f"Smart crawl for {url} completed. Processed {len(processed_urls_set)} pages. Stored {total_successful_chunks_stored} chunks.")
        return {
            "success": True,
            "url": url,
            "crawl_type": crawl_type,
            "pages_crawled_count": len(processed_urls_set),
            "total_successful_chunks_stored": total_successful_chunks_stored,
            "total_failed_chunks_stored": total_failed_chunks_stored,
            "urls_crawled_sample": list(processed_urls_set)[:5] + (["..."] if len(processed_urls_set) > 5 else [])
        }

    except Exception as e:
        logger.error(f"Exception in 'smart_crawl_url' for {url}: {e}", exc_info=True)
        return {"success": False, "url": url, "error": str(e)}

@mcp.tool()
async def perform_rag_query(ctx: Context, query: str, source: Optional[str] = None, match_count: int = 5) -> Dict[str, Any]:
    """
    Performs a RAG query against the Qdrant database.
    Returns a list of matching documents in a single response.
    """
    logger.info(f"Tool 'perform_rag_query' called with query: '{query}', source_filter: {source}, match_count: {match_count}")
    app_s: Crawl4AIAppContext = ctx.app.state.crawl4ai_app_context
    
    try:
        # No need for progress reporting here for a non-streaming tool returning a single object
        results = await utils.query_qdrant(
            client=app_s.qdrant_client,
            collection_name=app_s.collection_name,
            query_text=query,
            source_filter=source if source and source.strip() else None,
            match_count=match_count
        )
        logger.debug(f"RAG query for '{query}' returned {len(results)} results from Qdrant.")

        return {
            "success": True,
            "query": query,
            "source_filter": source,
            "results": results, # results is already a list of dicts
            "count": len(results)
        }

    except Exception as e:
        logger.error(f"Exception in 'perform_rag_query' for query '{query}': {e}", exc_info=True)
        return {"success": False, "query": query, "error": str(e)}

@mcp.tool()
async def get_available_sources(ctx: Context) -> Dict[str, Any]:
    """
    Retrieves a list of unique source domains from the Qdrant database.
    """
    logger.info(f"Tool 'get_available_sources' called.")
    app_s: Crawl4AIAppContext = ctx.app.state.crawl4ai_app_context
    
    try:
        sources = await utils.get_available_sources_async(app_s.qdrant_client, app_s.collection_name)
        logger.debug(f"'get_available_sources' found {len(sources)} sources.")
        return {
            "success": True,
            "sources": sources,
            "count": len(sources)
        }
    except Exception as e:
        logger.error(f"Exception in 'get_available_sources': {e}", exc_info=True)
        return {"success": False, "error": str(e)}

# --- Server Execution ---
if __name__ == "__main__":
    host = os.getenv("MCP_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_PORT", "8051")) # Defaulting to 8051 as in original code
    
    logger.info(f"Starting Crawl4AI RAG MCP Server on {host}:{port} with Streamable HTTP transport (via Uvicorn directly).")
    
    # Using Uvicorn directly with the ASGI app from mcp.http_app()
    # This can sometimes provide more direct control over the ASGI server lifecycle.
    import uvicorn
    
    # The path argument in http_app defines the root path for MCP requests.
    # This is important if your client expects the MCP server at /mcp.
    asgi_app = mcp.http_app(path="/mcp") 
    
    uvicorn.run(
        asgi_app, 
        host=host,
        port=port,
        log_level=LOG_LEVEL_STR.lower()
    )

    # Commenting out the direct mcp.run() for now to test uvicorn direct run
    # mcp.run(
    #     transport="streamable-http", 
    #     host=host,
    #     port=port,
    #     log_level=LOG_LEVEL_STR.lower() 
    # )