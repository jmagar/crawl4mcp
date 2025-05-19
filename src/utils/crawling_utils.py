"""
Utility functions for web crawling, parsing, and chunking.
"""
import os
import re
import requests
from xml.etree import ElementTree
from urllib.parse import urlparse, urldefrag
from typing import List, Dict, Any, Optional
import asyncio # Ensure asyncio is imported

# Import logging utilities
from .logging_utils import get_logger

# Initialize logger
logger = get_logger(__name__)

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, MemoryAdaptiveDispatcher

# Chunking configuration from environment variables (or defaults)
# These were originally in crawl4mcp-server.py
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MARKDOWN_CHUNK_SIZE = int(os.getenv("MARKDOWN_CHUNK_SIZE", "750"))

def is_sitemap(url: str) -> bool:
    """
    Check if a URL is a sitemap.
    """
    return url.endswith('sitemap.xml') or 'sitemap' in urlparse(url).path

def is_txt(url: str) -> bool:
    """
    Check if a URL is a text file.
    """
    return url.endswith('.txt')

def parse_sitemap(sitemap_url: str) -> List[str]:
    """
    Parse a sitemap and extract URLs.
    """
    urls = []
    try:
        # Define a synchronous helper for requests.get
        def _fetch_sitemap():
            return requests.get(sitemap_url, timeout=10)

        # Run the synchronous call in a thread
        # Note: parse_sitemap itself is NOT async, so this await asyncio.to_thread
        # is only callable if parse_sitemap is called from an async context that can run it.
        # If parse_sitemap is called from purely synchronous code, this will not work as intended
        # without further changes to how it's called or by using a sync-to-async bridge.
        # For now, assuming it's called from an async context that can handle this.
        # A better long-term solution would be to make parse_sitemap async itself if it's primarily used by async code.
        loop = asyncio.get_event_loop()
        resp = loop.run_until_complete(asyncio.to_thread(_fetch_sitemap))
        
        resp.raise_for_status() # Raise an exception for HTTP errors
        tree = ElementTree.fromstring(resp.content)
        # Namespace-agnostic way to find <loc> tags
        urls = [loc.text for loc in tree.findall('.//{*}loc') if loc.text]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching sitemap {sitemap_url}: {e}")
    except ElementTree.ParseError as e:
        logger.error(f"Error parsing sitemap XML from {sitemap_url}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while parsing sitemap {sitemap_url}: {e}")
    return urls

def smart_chunk_markdown(text: str, chunk_size: Optional[int] = None) -> List[str]:
    """
    Split text into chunks, respecting code blocks and paragraphs.

    Args:
        text: The input text to be chunked.
        chunk_size: The target maximum size for each chunk. Defaults to MARKDOWN_CHUNK_SIZE.

    Returns:
        A list of text chunks.
    """
    if chunk_size is None:
        chunk_size = MARKDOWN_CHUNK_SIZE

    chunks = []
    current_pos = 0
    text_len = len(text)

    while current_pos < text_len:
        end_pos = min(current_pos + chunk_size, text_len)
        current_chunk_text = text[current_pos:end_pos]

        if end_pos == text_len: # Last chunk
            chunks.append(current_chunk_text.strip())
            break

        # Try to find a natural break point to avoid splitting mid-sentence/paragraph/code block
        # Prioritize full paragraph breaks, then code block fences, then sentence breaks.
        para_break = current_chunk_text.rfind('\n\n')
        code_fence_break = current_chunk_text.rfind('\n```') # Look for end of a code block or start of new one
        sentence_break = -1
        # Try to break at a sentence ending with. ! ? followed by space or newline
        sentence_end_match = list(re.finditer(r'[.!?](\s|\n)+', current_chunk_text))
        if sentence_end_match:
            sentence_break = sentence_end_match[-1].start() + 1 # after the punctuation
        
        # Determine the best split point
        split_at = -1
        if para_break > chunk_size * 0.5: # Prefer paragraph break if it's reasonably far in
            split_at = para_break + 2 # after the \n\n
        elif code_fence_break > chunk_size * 0.3: # Then code fence
            # if it's ```\n, we want to split *after* the newline
            if text[current_pos+code_fence_break:].startswith('```\n'):
                 split_at = code_fence_break + 4
            else: # just ```
                 split_at = code_fence_break + 3
        elif sentence_break > chunk_size * 0.4: # Then sentence break
            split_at = sentence_break
        
        if split_at != -1 and (current_pos + split_at) < end_pos : # Ensure we are making progress and split is valid
            final_chunk = text[current_pos : current_pos + split_at].strip()
            current_pos += split_at
        else: # No good break found, take the full chunk_size or what's left
            final_chunk = text[current_pos:end_pos].strip()
            current_pos = end_pos
        
        if final_chunk:
            chunks.append(final_chunk)

    return [c for c in chunks if c] # Filter out any empty chunks

def extract_section_info(chunk: str) -> Dict[str, Any]:
    """
    Extracts headers and stats from a text chunk.

    Args:
        chunk: The text chunk to analyze.

    Returns:
        A dictionary containing:
            'headers' (str): A semicolon-separated string of found headers (e.g., "# Header 1; ## Header 2").
            'char_count' (int): The number of characters in the chunk.
            'word_count' (int): The number of words in the chunk.
    """
    headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
    header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers]) if headers else ''
    return {
        "headers": header_str,
        "char_count": len(chunk),
        "word_count": len(chunk.split())
    }

def simple_text_chunker(text: str, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None) -> List[str]:
    """
    Splits text into chunks with overlap, suitable for general text or code.

    Args:
        text: The input text to be chunked.
        chunk_size: The target size for each chunk. Defaults to CHUNK_SIZE.
        chunk_overlap: The number of characters to overlap between chunks. Defaults to CHUNK_OVERLAP.

    Returns:
        A list of text chunks.
    """
    if chunk_size is None:
        chunk_size = CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = CHUNK_OVERLAP

    if chunk_overlap >= chunk_size:
        logger.warning(f"Chunk overlap ({chunk_overlap}) is >= chunk size ({chunk_size}). Setting overlap to {chunk_size // 3}")
        chunk_overlap = chunk_size // 3

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
        if start >= end: # Ensure progress if overlap is too large relative to chunk_size
            start = end # Move to the end of the current chunk to guarantee progress
            
    return [chunk for chunk in chunks if chunk and chunk.strip()]

async def crawl_markdown_file(crawler: AsyncWebCrawler, url: str) -> List[Dict[str, Any]]:
    """
    Crawl a .txt or markdown file using AsyncWebCrawler.

    Args:
        crawler: An instance of AsyncWebCrawler.
        url: The URL of the .txt or markdown file to crawl.

    Returns:
        A list containing a single dictionary with 'url' and 'markdown' content
        if successful, e.g., [{'url': 'http://example.com/file.md', 'markdown': '...content...'}].
        Returns an empty list if crawling fails.
    """
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False) # Bypass cache for single file fetches
    result = await crawler.arun(url=url, config=crawl_config)
    if result.success and result.markdown:
        return [{'url': url, 'markdown': result.markdown}]
    else:
        logger.error(f"Failed to crawl {url}: {result.error_message}")
        return []

async def crawl_batch(crawler: AsyncWebCrawler, urls: List[str], max_concurrent: int = 10) -> List[Dict[str, Any]]:
    """
    Batch crawl multiple URLs in parallel using AsyncWebCrawler.

    Args:
        crawler: An instance of AsyncWebCrawler.
        urls: A list of URLs to crawl.
        max_concurrent: The maximum number of concurrent crawling tasks. Defaults to 10.

    Returns:
        A list of dictionaries, where each dictionary contains 'url' and 'markdown'
        for successfully crawled pages, e.g., [{'url': '...', 'markdown': '...'}, ...].
    """
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    # Consider making dispatcher configurable or passed in if needed
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent
    )
    results = await crawler.arun_many(urls=urls, config=crawl_config, dispatcher=dispatcher)
    return [{'url': r.url, 'markdown': r.markdown} for r in results if r.success and r.markdown]

async def crawl_recursive_internal_links(
    crawler: AsyncWebCrawler, 
    start_urls: List[str], 
    max_depth: int = 3, 
    max_concurrent: int = 10,
    target_domain: Optional[str] = None # Optional: to ensure only links from this domain are followed
) -> List[Dict[str, Any]]:
    """
    Recursively crawl internal links from start URLs up to a maximum depth.
    Optionally restrict to a target domain.

    Args:
        crawler: An instance of AsyncWebCrawler.
        start_urls: A list of starting URLs for the crawl.
        max_depth: The maximum depth for recursive crawling. Defaults to 3.
        max_concurrent: The maximum number of concurrent crawling tasks. Defaults to 10.
        target_domain: Optional. If provided, only links from this domain will be followed
                       and included. If None, it's inferred from the first start_url.
    
    Returns:
        A list of dictionaries, where each dictionary contains 'url' and 'markdown'
        for successfully crawled pages, e.g., [{'url': '...', 'markdown': '...'}, ...].
    """
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0, 
        check_interval=1.0, 
        max_session_permit=max_concurrent
    )

    visited_normalized_urls = set()
    results_all_pages = []
    # Normalize start_urls immediately
    current_level_urls_to_crawl = {urldefrag(u)[0].rstrip('/') for u in start_urls}

    # If target_domain is not provided, try to infer from the first start_url
    if not target_domain and start_urls:
        parsed_first_url = urlparse(start_urls[0])
        target_domain = parsed_first_url.netloc
        logger.debug(f"Target domain inferred as: {target_domain}")

    for depth in range(max_depth):
        if not current_level_urls_to_crawl:
            break
        
        # Filter out already visited URLs before batch crawling
        urls_for_this_batch = list(current_level_urls_to_crawl - visited_normalized_urls)
        if not urls_for_this_batch:
            break
            
        logger.debug(f"Depth {depth + 1}, crawling {len(urls_for_this_batch)} URLs: {urls_for_this_batch[:3]}...")
        visited_normalized_urls.update(urls_for_this_batch)
        
        crawl_results_this_batch = await crawler.arun_many(urls=urls_for_this_batch, config=run_config, dispatcher=dispatcher)
        
        next_level_urls_to_visit = set()
        for result in crawl_results_this_batch:
            if result.success and result.markdown:
                results_all_pages.append({'url': result.url, 'markdown': result.markdown})
                
                # Process internal links for next depth
                for link_info in result.links.get("internal", []):
                    raw_next_url = link_info["href"]
                    normalized_next_url = urldefrag(raw_next_url)[0].rstrip('/')
                    
                    # Check if it's on the same domain (if target_domain is set)
                    if target_domain:
                        parsed_link_domain = urlparse(normalized_next_url).netloc
                        if parsed_link_domain != target_domain:
                            continue # Skip if not same domain
                            
                    if normalized_next_url not in visited_normalized_urls:
                        next_level_urls_to_visit.add(normalized_next_url)
        
        current_level_urls_to_crawl = next_level_urls_to_visit

    logger.info(f"Recursive crawl finished. Total pages: {len(results_all_pages)}")
    return results_all_pages 