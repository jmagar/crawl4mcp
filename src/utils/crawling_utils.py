"""
Utility functions for web crawling, parsing, and local directory scanning.
"""
import os
import re
import requests
import asyncio
import uuid
from pathlib import Path
from xml.etree import ElementTree
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from src.utils.logging_utils import get_logger
from src.utils.text_processing import simple_text_chunker, smart_chunk_markdown
from src.config import settings

logger = get_logger(__name__)

# --- Constants ---
CHUNK_SIZE = settings.CHUNK_SIZE
CHUNK_OVERLAP = settings.CHUNK_OVERLAP
MARKDOWN_CHUNK_SIZE = settings.MARKDOWN_CHUNK_SIZE

DEFAULT_LOCAL_IGNORE_PATTERNS = [
    ".git",
    ".vscode",
    "__pycache__",
    "node_modules",
    ".pnpm-store",
    "dist",
    "build",
    "*.pyc",
    "*.egg-info",
    ".env",
]
DEFAULT_ALLOWED_EXTENSIONS = [
    ".py", ".md", ".json", ".txt", ".html", ".js", ".ts", ".jsx", ".tsx",
    ".css", ".scss", ".less", ".sh", ".bash", ".zsh", ".c", ".h", ".cpp",
    ".java", ".go", ".php", ".rb", ".rs", ".swift", ".kt", ".kts", ".scala",
    ".yml", ".yaml", ".toml", ".ini", ".cfg", ".sql"
]


def is_sitemap(url: str) -> bool:
    """Check if a URL is a sitemap."""
    return url.endswith('sitemap.xml') or 'sitemap' in urlparse(url).path


def is_txt(url: str) -> bool:
    """Check if a URL is a text file."""
    return url.endswith('.txt')


async def parse_sitemap(sitemap_url: str) -> List[str]:
    """Asynchronously parse a sitemap and extract URLs."""
    urls = []
    try:
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            None, 
            lambda: requests.get(sitemap_url, timeout=10)
        )
        resp.raise_for_status()
        tree = ElementTree.fromstring(resp.content)
        urls = [loc.text for loc in tree.findall('.//{*}loc') if loc.text]
    except Exception as e:
        logger.error(f"Error processing sitemap {sitemap_url}: {e}")
    return urls


def crawl_directory(
    dir_path: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    ignore_patterns: Optional[List[str]] = None,
    allowed_extensions: Optional[List[str]] = None,
    document_buffer: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Crawls a local directory, processes files, and adds chunks to a buffer.
    Correctly handles ignoring specified directory and file patterns.
    """
    if document_buffer is None:
        raise ValueError("A document_buffer list must be provided.")

    final_ignore_patterns = set(ignore_patterns or DEFAULT_LOCAL_IGNORE_PATTERNS)
    final_allowed_extensions = set(allowed_extensions or DEFAULT_ALLOWED_EXTENSIONS)
    
    effective_chunk_size = chunk_size or CHUNK_SIZE
    effective_chunk_overlap = chunk_overlap or CHUNK_OVERLAP

    logger.info(f"Starting directory crawl at: {dir_path}")
    logger.info(f"Ignoring patterns: {final_ignore_patterns}")

    stats = {
        "source_directory": dir_path, "total_files_scanned": 0, "total_dirs_scanned": 0,
        "total_files_processed": 0, "total_dirs_ignored": 0,
        "total_files_ignored_by_extension": 0, "total_chunks_added": 0,
    }

    dir_path_obj = Path(dir_path)

    for root, dirs, files in os.walk(dir_path_obj, topdown=True):
        stats["total_dirs_scanned"] += 1
        
        # Filter directories in-place to prevent os.walk from traversing them
        original_dir_count = len(dirs)
        dirs[:] = [d for d in dirs if d not in final_ignore_patterns]
        stats["total_dirs_ignored"] += (original_dir_count - len(dirs))

        for filename in files:
            stats["total_files_scanned"] += 1
            file_path_obj = Path(root) / filename

            if file_path_obj.suffix.lower() not in final_allowed_extensions:
                stats["total_files_ignored_by_extension"] += 1
                continue

            try:
                with open(file_path_obj, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                
                if not content.strip():
                    continue
                
                stats["total_files_processed"] += 1
                
                chunks = smart_chunk_markdown(content, chunk_size=effective_chunk_size) \
                    if file_path_obj.suffix.lower() in ['.md', '.markdown'] \
                    else simple_text_chunker(content, effective_chunk_size, effective_chunk_overlap)

                for i, chunk_text in enumerate(chunks):
                    document_buffer.append({
                        "id": f"{file_path_obj}-{i}", "text": chunk_text, "source": "local_directory",
                        "source_path": str(file_path_obj),
                        "metadata": {
                            "filename": filename, "path": str(file_path_obj.relative_to(dir_path_obj)),
                            "chunk_index": i, "total_chunks": len(chunks),
                        }
                    })
                stats["total_chunks_added"] += len(chunks)

            except Exception as e:
                logger.error(f"Failed to read or process file {file_path_obj}: {e}")

    logger.info("Directory crawl finished.")
    stats["message"] = (
        f"Directory processing complete. {stats['total_chunks_added']} chunks from "
        f"{stats['total_files_processed']} files added to buffer."
    )
    stats["buffer_size"] = len(document_buffer)
    
    return stats


async def crawl_and_chunk_urls(
    urls: List[str],
    chunk_size: int = MARKDOWN_CHUNK_SIZE,
    document_buffer: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    Crawls a list of URLs, chunks the content, and adds to the document buffer.
    """
    if document_buffer is None:
        raise ValueError("A document_buffer list must be provided.")

    crawler = AsyncWebCrawler()
    config = CrawlerRunConfig(cache_mode=CacheMode.ENABLED, stream=False)

    async def process_url(url):
        result = await crawler.arun(url=url, config=config)
        if result.success and result.markdown:
            chunks = smart_chunk_markdown(result.markdown, chunk_size)
            for i, chunk in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                document_buffer.append({
                    "id": chunk_id,
                    "text": chunk,
                    "source": "web_url",
                    "source_path": url,
                    "metadata": {
                        "chunk_index": i, 
                        "total_chunks": len(chunks),
                        "original_url": url
                    }
                })
            logger.info(f"Successfully crawled and chunked {url}, added {len(chunks)} chunks.")
        else:
            logger.error(f"Failed to crawl {url}: {result.error_message}")

    tasks = [process_url(url) for url in urls]
    await asyncio.gather(*tasks) 