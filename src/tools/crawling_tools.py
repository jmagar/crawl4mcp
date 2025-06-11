"""
MCP Tools for crawling web pages, repositories, and smart URL handling.
"""
import json
import asyncio
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
import os
import fnmatch
import uuid
import hashlib

from mcp.server.fastmcp.exceptions import ToolError # Correct import path

# Import the centralized mcp instance
from ..mcp_setup import mcp
# Import utility functions
from ..utils.qdrant.setup import get_qdrant_client
from ..utils.qdrant.ingestion import store_embeddings
from qdrant_client import QdrantClient
from ..utils.crawling_utils import (
    is_sitemap,
    is_txt,
    parse_sitemap,
    smart_chunk_markdown,
    extract_section_info,
    simple_text_chunker,
    crawl_markdown_file,
    crawl_batch,
    crawl_recursive_internal_links,
    CHUNK_SIZE,             # For crawl_repo default
    CHUNK_OVERLAP,          # For crawl_repo default
    MARKDOWN_CHUNK_SIZE     # For smart_crawl_url default
)
# Import logging utilities
from ..utils.logging_utils import get_logger
from ..config import settings
from ..utils.text_processing import extract_code_blocks

# Default ignore patterns for local directory crawling (glob-style)
DEFAULT_LOCAL_IGNORE_PATTERNS = [
    # Common version control and environment stuff
    ".git", ".svn", ".hg", "CVS",
    "venv", ".venv", "env", ".env",
    # Common build/cache/temp artifacts
    "__pycache__", "*.pyc", "*.pyo", "*.so", "*.o",
    "node_modules", ".next", ".cache", ".turbo", "dist", "build", "target", "out",
    "logs", "*.log", "*.tmp", "*.temp", "*.swp", "*.swo", "*.swn",
    # OS-specific
    ".DS_Store", "._*", ".Spotlight-V100", ".Trashes", "Thumbs.db",
    # Archives and compiled executables (less likely to want to embed raw)
    "*.zip", "*.tar.gz", "*.rar", "*.7z", "*.exe", "*.dll", "*.pdb", "*.obj",
    # Common data files that might be too large or not text-based for direct embedding
    "*.db", "*.sqlite", "*.sqlite3", "*.parquet", "*.arrow"
]

# Default allowed extensions (reused from crawl_repo logic, can be refined)
DEFAULT_ALLOWED_EXTENSIONS = [
    # Common code files
    '.py', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.hpp', '.cs', '.go', '.rs', '.swift',
    '.kt', '.scala', '.rb', '.php', '.pl', '.pm', '.sh', '.bash', '.ps1', '.bat',
    # Common text/markup/config files
    '.txt', '.md', '.rst', '.json', '.yaml', '.yml', '.xml', '.toml', '.ini', '.cfg',
    '.html', '.htm', '.css', '.scss', '.less',
    # Data files
    '.csv', '.tsv'
]

# Initialize logger
logger = get_logger(__name__)

from crawl4ai import CrawlerRunConfig, CacheMode, AsyncWebCrawler, BrowserConfig

async def _process_and_store_single_repo_file(
    file_path_obj: Path,
    repo_root_path: Path,
    qdrant_client_instance: QdrantClient, # Type hint for clarity
    collection_name_str: str,
    repo_url: str,
    effective_chunk_size: int,
    effective_chunk_overlap: int
) -> Dict[str, Any]:
    """
    Reads, chunks, and stores embeddings for a single file from the repository.
    Returns a dict with stats for this file (path, successful_chunks, failed_chunks, error).
    """
    relative_path = file_path_obj.relative_to(repo_root_path)
    file_summary = {
        "path": str(relative_path),
        "successful_chunks": 0,
        "failed_chunks": 0,
        "error": None,
        "content_read": False # New flag
    }
    chunks_data_for_qdrant_list: List[Dict[str, Any]] = [] # Ensure it's defined for the error case
    code_chunks_data_for_qdrant_list: List[Dict[str, Any]] = []

    try:
        logger.debug(f"Reading file: {relative_path}")
        content = await asyncio.to_thread(file_path_obj.read_text, encoding="utf-8", errors="ignore")
        file_summary["content_read"] = True
        if not content.strip():
            logger.debug(f"File is empty: {relative_path}")
            return file_summary

        # Agentic RAG: Extract and prepare code blocks if enabled
        if settings.USE_AGENTIC_RAG:
            logger.debug(f"Agentic RAG enabled. Extracting code blocks from repo file: {relative_path}")
            extracted_codes = extract_code_blocks(content, settings.CODE_BLOCK_MIN_LENGTH)
            for i, code_block_info in enumerate(extracted_codes):
                code_content = code_block_info['code']
                code_lang = code_block_info.get('language', 'unknown')
                code_block_hash = hashlib.md5(code_content.encode('utf-8')).hexdigest()
                # Use repo_url and relative_path for a more globally unique ID namespace
                code_block_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{repo_url}::{str(relative_path)}::{code_block_hash}"))
                
                code_chunks_data_for_qdrant_list.append({
                    "id": code_block_id,
                    "text": code_content,
                    "headers": f"Code Block from: {repo_url} (file: {str(relative_path)}) - Language: {code_lang}",
                    "payload_override": {
                        "source": f"{repo_url} (file: {str(relative_path)} - code_block)",
                        "crawl_type": "code_example",
                        "language": code_lang,
                        "original_file_path": str(relative_path),
                        "repo_url": repo_url,
                        "code_hash": code_block_hash
                    }
                })
            logger.info(f"Extracted {len(code_chunks_data_for_qdrant_list)} code blocks from {relative_path} for Agentic RAG.")

        # Store extracted code blocks if any
        if code_chunks_data_for_qdrant_list:
            logger.debug(f"Storing {len(code_chunks_data_for_qdrant_list)} code blocks for repo file: {relative_path}")
            successful_code, failed_code = await store_embeddings(
                client=qdrant_client_instance,
                collection_name=collection_name_str,
                chunks=code_chunks_data_for_qdrant_list,
                source_url=f"{repo_url} (file: {str(relative_path)} - code_blocks)",
                crawl_type="code_example_batch"
            )
            file_summary["successful_chunks"] += successful_code
            file_summary["failed_chunks"] += failed_code
            logger.debug(f"Stored code_example embeddings for repo file {relative_path}: {successful_code} successful, {failed_code} failed")

        logger.debug(f"Chunking general file content: {relative_path}") # Renamed log message for clarity
        current_file_chunks_text = simple_text_chunker(
            content,
            chunk_size=effective_chunk_size,
            chunk_overlap=effective_chunk_overlap
        )
        if not current_file_chunks_text:
            logger.debug(f"No chunks generated for file: {relative_path}")
            return file_summary

        for i, chunk_content_item in enumerate(current_file_chunks_text):
            meta_info = f"File: {str(relative_path)} - Chunk: {i+1}/{len(current_file_chunks_text)}"
            chunks_data_for_qdrant_list.append({"text": chunk_content_item, "headers": meta_info})
        
        # Process and store general file content (original logic)
        if chunks_data_for_qdrant_list: # This list should now ONLY contain general content chunks
            logger.debug(f"Storing {len(chunks_data_for_qdrant_list)} general content chunks for repo file: {relative_path}")
            successful, failed = await store_embeddings(
                client=qdrant_client_instance,
                collection_name=collection_name_str,
                chunks=chunks_data_for_qdrant_list,
                source_url=f"{repo_url} (file: {str(relative_path)})",
                crawl_type="repository"
            )
            file_summary["successful_chunks"] = successful
            file_summary["failed_chunks"] = failed
            logger.debug(f"Stored embeddings for {relative_path}: {successful} successful, {failed} failed")
            
    except Exception as e_file:
        logger.error(f"Error processing file {relative_path}: {e_file}")
        file_summary["error"] = str(e_file)
        # If error occurred after chunking, estimate all potential chunks as failed.
        # If error was during read, chunks_data_for_qdrant_list would be empty.
        file_summary["failed_chunks"] = len(chunks_data_for_qdrant_list) if chunks_data_for_qdrant_list else 0
        if not file_summary["content_read"]: # If read itself failed
             file_summary["failed_chunks"] = 1 # Mark as at least one conceptual failure for the file

    return file_summary

async def _process_and_store_single_local_file(
    file_path_obj: Path,
    base_dir_path: Path, 
    qdrant_client_instance: QdrantClient,
    collection_name_str: str,
    source_identifier: str, 
    effective_chunk_size: int,
    effective_chunk_overlap: int
) -> Dict[str, Any]:
    """
    Reads, chunks, and stores embeddings for a single file from the local directory.
    """
    relative_path = file_path_obj.relative_to(base_dir_path)
    file_summary = {
        "path": str(relative_path),
        "successful_chunks": 0,
        "failed_chunks": 0,
        "error": None,
        "content_read": False
    }
    chunks_data_for_qdrant_list: List[Dict[str, Any]] = []
    code_chunks_data_for_qdrant_list: List[Dict[str, Any]] = []

    try:
        logger.debug(f"Reading local file: {relative_path}")
        content = await asyncio.to_thread(file_path_obj.read_text, encoding="utf-8", errors="ignore")
        file_summary["content_read"] = True
        if not content.strip():
            logger.debug(f"Local file is empty: {relative_path}")
            return file_summary

        # Agentic RAG: Extract and prepare code blocks if enabled
        if settings.USE_AGENTIC_RAG:
            logger.debug(f"Agentic RAG enabled. Extracting code blocks from: {relative_path}")
            extracted_codes = extract_code_blocks(content, settings.CODE_BLOCK_MIN_LENGTH)
            for i, code_block_info in enumerate(extracted_codes):
                code_content = code_block_info['code']
                code_lang = code_block_info.get('language', 'unknown')
                # Generate a stable ID for the code block
                code_block_hash = hashlib.md5(code_content.encode('utf-8')).hexdigest()
                code_block_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{str(relative_path)}::{code_block_hash}"))
                
                # For code blocks, the 'text' is the code itself.
                # Metadata includes language and original file path.
                code_chunks_data_for_qdrant_list.append({  # Corrected list name here
                    "id": code_block_id,
                    "text": code_content,
                    "headers": f"Code Block from: {str(relative_path)} - Language: {code_lang}",
                    "payload_override": { # This allows direct payload setting in store_embeddings
                        "source": f"local_dir: {source_identifier} (file: {str(relative_path)} - code_block)",
                        "crawl_type": "code_example",
                        "language": code_lang,
                        "original_file_path": str(relative_path),
                        "code_hash": code_block_hash
                    }
                })
            logger.info(f"Extracted {len(code_chunks_data_for_qdrant_list)} code blocks from {relative_path} for Agentic RAG.")

        # Store extracted code blocks if any
        if code_chunks_data_for_qdrant_list:
            logger.debug(f"Storing {len(code_chunks_data_for_qdrant_list)} code blocks for local file: {relative_path}")
            successful_code, failed_code = await store_embeddings(
                client=qdrant_client_instance,
                collection_name=collection_name_str,
                chunks=code_chunks_data_for_qdrant_list,
                source_url=f"local_dir: {source_identifier} (file: {str(relative_path)} - code_blocks)",
                crawl_type="code_example_batch"
            )
            file_summary["successful_chunks"] += successful_code
            file_summary["failed_chunks"] += failed_code
            logger.debug(f"Stored code_example embeddings for local {relative_path}: {successful_code} successful, {failed_code} failed")

        # Process and store general file content (original logic)
        current_file_chunks_text = simple_text_chunker(
            content,
            chunk_size=effective_chunk_size,
            chunk_overlap=effective_chunk_overlap
        )
        if not current_file_chunks_text:
            logger.debug(f"No chunks generated for local file: {relative_path}")
            return file_summary

        for i, chunk_content_item in enumerate(current_file_chunks_text):
            meta_info = f"File: {str(relative_path)} - Chunk: {i+1}/{len(current_file_chunks_text)}"
            chunks_data_for_qdrant_list.append({"text": chunk_content_item, "headers": meta_info})
        
        # This is the existing block for general content, ensure it's correctly placed and uses the right list
        if chunks_data_for_qdrant_list: # This list should now ONLY contain general content chunks
            successful, failed = await store_embeddings(
                client=qdrant_client_instance,
                collection_name=collection_name_str,
                chunks=chunks_data_for_qdrant_list, # This should be the general content chunks
                source_url=f"local_dir: {source_identifier} (file: {str(relative_path)})", # Source for general content
                crawl_type="local_directory"
            )
            file_summary["successful_chunks"] = successful
            file_summary["failed_chunks"] = failed
            logger.debug(f"Stored embeddings for local {relative_path}: {successful} successful, {failed} failed")
            
    except Exception as e_file:
        logger.error(f"Error processing local file {relative_path}: {e_file}")
        file_summary["error"] = str(e_file)
        file_summary["failed_chunks"] = len(chunks_data_for_qdrant_list) if chunks_data_for_qdrant_list else 0
        if not file_summary["content_read"]:
             file_summary["failed_chunks"] = 1

    return file_summary

@mcp.tool()
async def crawl_single_page(url: str) -> str:
    """
    Crawl a single web page and store its content in Qdrant.
    
    This tool is ideal for quickly retrieving content from a specific URL without following links.
    The content is stored in Qdrant for later retrieval and querying.
    
    Args:
        url: URL of the web page to crawl
    
    Returns:
        Summary of the crawling operation and storage in Qdrant
    """
    logger.info(f"Starting crawl_single_page for URL: {url}")
    crawler_instance = None
    crawler_initialized_here = False

    try:
        # Initialize components from environment
        qdrant_client_instance = get_qdrant_client()
        collection_name_str = os.getenv("QDRANT_COLLECTION")
        if not collection_name_str:
            logger.error("QDRANT_COLLECTION environment variable must be set")
            raise ToolError(f"QDRANT_COLLECTION environment variable must be set.", "CONFIG_ERROR")

        # Initialize AsyncWebCrawler
        browser_config = BrowserConfig(
            headless=True,
            verbose=os.getenv("CRAWLER_VERBOSE", "False").lower() == "true"
        )
        crawler_instance = AsyncWebCrawler(config=browser_config)
        await crawler_instance.__aenter__()
        crawler_initialized_here = True
        logger.debug("Initialized standalone crawler")

        logger.info(f"Crawling page: {url}")
        # Configure crawler run for single page
        run_config = CrawlerRunConfig(
            page_timeout=60000,  # Increased timeout to 60 seconds
            cache_mode=CacheMode.BYPASS  # Bypass cache for direct calls
        )
        
        # Get the raw HTML or content from the web page
        results_list = await crawler_instance.arun(url, config=run_config)
        
        response = None
        if results_list and isinstance(results_list, list) and len(results_list) > 0:
            response = results_list[0] # Take the first result
            logger.debug(f"Response type from arun (first element): {type(response)}")
        elif results_list: # This block should catch the single CrawlResultContainer
            response = results_list 
            logger.debug(f"Response type from arun (direct object): {type(response)}")
        
        # Check for markdown attribute for content
        if not response or not hasattr(response, 'markdown') or not response.markdown:
            logger.warning(f"No markdown content retrieved from URL: {url}")
            raise ToolError(f"No markdown content retrieved from URL: {url}", "NO_CONTENT")
        else:
            logger.debug(f"Successfully retrieved markdown content from {url} ({len(response.markdown)} bytes)")
            
            # Convert to chunks with metadata
            raw_chunks = smart_chunk_markdown(
                text=response.markdown,
                chunk_size=MARKDOWN_CHUNK_SIZE,
            )
            
            if not raw_chunks:
                logger.warning(f"No content chunks generated for URL: {url}")
                raise ToolError(f"No content chunks generated for URL: {url}", "NO_CHUNKS")
            else:
                # Transform List[str] to List[Dict[str, Any]] for store_embeddings
                chunks_data_for_qdrant = [{"text": chunk_str} for chunk_str in raw_chunks]

                # Store embeddings in Qdrant
                logger.info(f"Storing {len(chunks_data_for_qdrant)} chunks from {url}")
                successful, failed = await store_embeddings(
                    client=qdrant_client_instance,
                    collection_name=collection_name_str,
                    chunks=chunks_data_for_qdrant,
                    source_url=url,
                    crawl_type="single_page"
                )
                
                # Format the result
                result = {
                    "success": True,
                    "url": url,
                    "total_chunks": len(chunks_data_for_qdrant),
                    "successful_chunks": successful,
                    "failed_chunks": failed
                }
                logger.info(f"Crawl complete for {url}: {successful} chunks stored successfully, {failed} failed")
                
    except Exception as e:
        logger.error(f"Error during crawl_single_page for {url}: {str(e)}")
        raise ToolError(f"Error during crawl_single_page for {url}: {str(e)}", "CRAWL_ERROR", {"url": url, "original_exception": str(e)})
    finally:
        # Only clean up crawler if we initialized it in this call
        if crawler_initialized_here and crawler_instance:
            logger.debug("Cleaning up standalone crawler")
            await crawler_instance.__aexit__(None, None, None)
            
    # Return formatted result as JSON string
    return json.dumps(result, indent=2)

@mcp.tool()
async def crawl_repo(
    repo_url: str,
    branch: Optional[str] = None,
    chunk_size: Optional[int] = None,      # User can override
    chunk_overlap: Optional[int] = None,   # User can override
    ignore_dirs: Optional[List[str]] = None # New: Directories/patterns to ignore
) -> str:
    """
    Clones a Git repository, processes specified file types (based on filters), and stores their content in Qdrant.

    Args:
        repo_url: URL of the Git repository to crawl.
        branch: Optional specific branch to clone. Defaults to the repository's default branch.
        chunk_size: Size of each text chunk in characters for processing. Defaults to CHUNK_SIZE from crawling_utils.
        chunk_overlap: Overlap between text chunks in characters. Defaults to CHUNK_OVERLAP from crawling_utils.
        ignore_dirs: Optional list of directory names or path patterns to ignore (e.g., [".git", "node_modules", "dist/"]).

    Returns:
        JSON string with crawl summary and storage information.
    """
    effective_chunk_size = chunk_size if chunk_size is not None else CHUNK_SIZE
    effective_chunk_overlap = chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP
    
    # Default ignored directories/patterns if none provided
    default_ignore_dirs = [".git", "node_modules", "__pycache__", ".venv", "venv", "env", "dist", "build", "target", ".DS_Store"]
    final_ignore_dirs = ignore_dirs if ignore_dirs is not None else default_ignore_dirs

    try:
        # Initialize components from environment
        qdrant_client_instance = get_qdrant_client()
        collection_name_str = os.getenv("QDRANT_COLLECTION")
        if not collection_name_str:
            raise ValueError("QDRANT_COLLECTION environment variable must be set.")

        logger.info(f"Starting repository crawl for {repo_url}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir) / "cloned_repo"
            git_command = ["/usr/bin/git", "clone", "--depth", "1"]
            if branch:
                git_command.extend(["--branch", branch])
            git_command.extend([repo_url, str(repo_path)])

            process = await asyncio.to_thread(
                subprocess.run, git_command, capture_output=True, text=True, check=False
            )

            if process.returncode != 0:
                error_message = process.stderr or process.stdout or "Unknown git clone error"
                raise ToolError(f"Git clone failed: {error_message.strip()}", "GIT_CLONE_ERROR", {"repo_url": repo_url, "stderr": process.stderr, "stdout": process.stdout})

            all_file_paths_to_process = []
            logger.info(f"Identifying files to process...")
            
            for fp_obj in repo_path.rglob("*"):
                if fp_obj.is_file():
                    # Apply filtering
                    try:
                        relative_path_str = str(fp_obj.relative_to(repo_path))
                        path_parts = set(Path(relative_path_str).parts)
                        
                        # Check ignored directories
                        skip_due_to_ignore = False
                        for ignored_pattern in final_ignore_dirs:
                            if ignored_pattern in path_parts:
                                skip_due_to_ignore = True
                                break
                            if relative_path_str.startswith(ignored_pattern + os.path.sep) or relative_path_str == ignored_pattern:
                                skip_due_to_ignore = True
                                break
                        if skip_due_to_ignore:
                            continue
                        
                        all_file_paths_to_process.append(fp_obj)
                    except Exception as path_err:
                        logger.warning(f"Skipping file {fp_obj} due to path processing error: {path_err}")
                        continue
            
            logger.info(f"Found {len(all_file_paths_to_process)} files to process. Starting concurrent processing.")
            tasks = []
            for fp_obj in all_file_paths_to_process:
                tasks.append(
                    _process_and_store_single_repo_file(
                        file_path_obj=fp_obj,
                        repo_root_path=repo_path,
                        qdrant_client_instance=qdrant_client_instance,
                        collection_name_str=collection_name_str,
                        repo_url=repo_url,
                        effective_chunk_size=effective_chunk_size,
                        effective_chunk_overlap=effective_chunk_overlap
                    )
                )
            
            # Using asyncio.gather to run file processing tasks concurrently
            results_of_tasks = await asyncio.gather(*tasks, return_exceptions=True)

            logger.info("Aggregating results from file processing...")
            # Aggregate results
            files_processed_successfully_count = 0
            total_successful_chunks_stored = 0
            total_failed_chunks_during_storage = 0
            files_with_processing_errors_list = []
            processed_file_paths_list = []

            for result_item in results_of_tasks:
                if isinstance(result_item, Exception):
                    files_with_processing_errors_list.append(f"A file processing task failed: {str(result_item)}")
                    continue

                file_path_str = result_item["path"]
                
                if result_item["error"]:
                    files_with_processing_errors_list.append(f"{file_path_str}: {result_item['error']}")
                elif result_item["content_read"] and result_item["successful_chunks"] > 0 :
                    processed_file_paths_list.append(file_path_str)
                    files_processed_successfully_count += 1
                elif result_item["content_read"] and result_item["successful_chunks"] == 0 and result_item["failed_chunks"] == 0 and not result_item["error"]:
                    processed_file_paths_list.append(file_path_str)
                    files_processed_successfully_count +=1

                total_successful_chunks_stored += result_item["successful_chunks"]
                total_failed_chunks_during_storage += result_item["failed_chunks"]
            
            summary = {
                "success": True,
                "repo_url": repo_url,
                "branch_crawled": branch if branch else "default",
                "files_attempted": len(all_file_paths_to_process),
                "files_processed_successfully": files_processed_successfully_count,
                "processed_file_paths": processed_file_paths_list,
                "total_successful_chunks_stored": total_successful_chunks_stored,
                "total_failed_chunks_during_storage": total_failed_chunks_during_storage,
                "files_with_processing_errors": files_with_processing_errors_list
            }
            logger.info(f"Repository crawl for {repo_url} complete.")
            return json.dumps(summary, indent=4)
    except Exception as e:
        logger.error(f"An unexpected error occurred in crawl_repo: {str(e)}", exc_info=True)
        raise ToolError(f"An unexpected error occurred in crawl_repo: {str(e)}", "REPO_CRAWL_UNEXPECTED_ERROR", {"repo_url": repo_url, "original_exception": str(e)})

@mcp.tool()
async def smart_crawl_url(url: str, max_depth: int = 3, max_concurrent: int = 30, chunk_size: Optional[int] = None) -> str:
    """
    Intelligently crawl a URL based on its type and store content in Qdrant.
    Args:
        url: URL to crawl
        max_depth: Maximum depth for recursive crawls (default: 3)
        max_concurrent: Maximum concurrent requests (default: 30)
        chunk_size: Max size of each markdown content chunk. Defaults to MARKDOWN_CHUNK_SIZE from crawling_utils.
    """
    effective_markdown_chunk_size = chunk_size if chunk_size is not None else MARKDOWN_CHUNK_SIZE
    
    crawler_instance = None
    crawler_initialized_here = False

    try:
        # Initialize components from environment
        qdrant_client_instance = get_qdrant_client()
        collection_name_str = os.getenv("QDRANT_COLLECTION")
        if not collection_name_str:
            raise ValueError("QDRANT_COLLECTION environment variable must be set.")

        browser_config = BrowserConfig(
            headless=True,
            verbose=os.getenv("CRAWLER_VERBOSE", "False").lower() == "true"
        )
        crawler_instance = AsyncWebCrawler(config=browser_config)
        await crawler_instance.__aenter__()
        crawler_initialized_here = True

        crawl_results = []
        crawl_type = "webpage"
        
        logger.info(f"Smart crawling URL: {url} (max_depth={max_depth}, max_concurrent={max_concurrent})")

        if is_txt(url):
            logger.info(f"URL recognized as text/markdown file: {url}")
            crawl_results = await crawl_markdown_file(crawler_instance, url)
            crawl_type = "text_file"
        elif is_sitemap(url):
            logger.info(f"URL recognized as sitemap: {url}")
            sitemap_urls = parse_sitemap(url)
            if not sitemap_urls:
                raise ToolError(f"No URLs found in sitemap: {url}", "SITEMAP_EMPTY")
            logger.info(f"Found {len(sitemap_urls)} URLs in sitemap. Starting batch crawl.")
            crawl_results = await crawl_batch(crawler_instance, sitemap_urls, max_concurrent=max_concurrent)
            crawl_type = "sitemap"
        else:
            logger.info(f"URL recognized as standard webpage. Starting recursive crawl: {url}")
            crawl_results = await crawl_recursive_internal_links(crawler_instance, [url], max_depth=max_depth, max_concurrent=max_concurrent)
            crawl_type = "webpage"
        
        if not crawl_results:
            raise ToolError(f"No content found for URL {url} with type {crawl_type}", "NO_CONTENT_FOUND")
        
        processed_urls = set()
        total_successful_chunks = 0
        total_failed_chunks = 0

        for i, doc_data in enumerate(crawl_results):
            source_url = doc_data['url']
            markdown_content = doc_data['markdown']
            processed_urls.add(source_url)
            
            logger.info(f"Processing content from {source_url} ({i+1}/{len(crawl_results)})")

            chunks_text = smart_chunk_markdown(markdown_content, chunk_size=effective_markdown_chunk_size)
            
            current_page_chunks_data = []
            for i, chunk_content_item in enumerate(chunks_text):
                meta = extract_section_info(chunk_content_item)
                current_page_chunks_data.append({"text": chunk_content_item, "headers": meta.get("headers", "")})
            
            successful_chunks, failed_chunks = await store_embeddings(
                client=qdrant_client_instance, collection_name=collection_name_str,
                chunks=current_page_chunks_data, source_url=source_url, crawl_type=crawl_type
            )
            total_successful_chunks += successful_chunks
            total_failed_chunks += failed_chunks
        
        logger.info(f"Smart crawl for {url} complete. Processed {len(processed_urls)} URLs, stored {total_successful_chunks} chunks.")
        return json.dumps({
            "success": True, "url": url, "crawl_type": crawl_type,
            "pages_crawled": len(processed_urls),
            "total_successful_chunks": total_successful_chunks,
            "total_failed_chunks": total_failed_chunks,
            "urls_crawled_sample": list(processed_urls)[:5] + (["..."] if len(processed_urls) > 5 else [])
        }, indent=2)
    except Exception as e:
        logger.error(f"Error during smart_crawl_url for {url}: {str(e)}", exc_info=True)
        raise ToolError(f"Error during smart_crawl_url for {url}: {str(e)}", "SMART_CRAWL_ERROR", {"url": url, "original_exception": str(e)})
    finally:
        if crawler_initialized_here and crawler_instance:
            await crawler_instance.__aexit__(None, None, None)

@mcp.tool()
async def crawl_dir(
    dir_path: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    ignore_patterns: Optional[List[str]] = None,
    allowed_extensions: Optional[List[str]] = None
) -> str:
    """
    Crawls a local directory, processes specified file types, and stores their content in Qdrant.

    Args:
        dir_path: Absolute path to the local directory to crawl.
        chunk_size: Size of each text chunk in characters. Defaults to CHUNK_SIZE from crawling_utils.
        chunk_overlap: Overlap between text chunks in characters. Defaults to CHUNK_OVERLAP from crawling_utils.
        ignore_patterns: Optional list of glob-style patterns for files/directories to ignore.
                         Defaults to DEFAULT_LOCAL_IGNORE_PATTERNS.
        allowed_extensions: Optional list of file extensions (e.g., ['.py', '.md']) to process.
                            Defaults to DEFAULT_ALLOWED_EXTENSIONS.

    Returns:
        JSON string with crawl summary and storage information.
    """
    logger.info(f"Starting crawl_dir for directory: {dir_path}")

    base_dir_path = Path(dir_path)
    if not base_dir_path.is_dir():
        logger.error(f"Provided path is not a directory or does not exist: {dir_path}")
        raise ToolError(f"Path is not a directory or does not exist: {dir_path}", "PATH_NOT_DIR")

    # Resolve effective settings
    effective_chunk_size = chunk_size if chunk_size is not None else CHUNK_SIZE
    effective_chunk_overlap = chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP
    effective_ignore_patterns = ignore_patterns if ignore_patterns is not None else DEFAULT_LOCAL_IGNORE_PATTERNS
    effective_allowed_extensions = allowed_extensions if allowed_extensions is not None else DEFAULT_ALLOWED_EXTENSIONS
    # Ensure extensions have a leading dot for consistent matching
    effective_allowed_extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in effective_allowed_extensions]

    try:
        qdrant_client_instance = get_qdrant_client()
        collection_name_str = os.getenv("QDRANT_COLLECTION")
        if not collection_name_str:
            logger.error("QDRANT_COLLECTION environment variable must be set")
            raise ToolError("QDRANT_COLLECTION environment variable must be set.", "CONFIG_ERROR")
    except Exception as e_setup:
        logger.error(f"Error during Qdrant setup for crawl_dir: {e_setup}")
        raise ToolError(f"Qdrant setup failed: {e_setup}", "QDRANT_SETUP_ERROR")

    processed_files_summary: List[Dict[str, Any]] = []
    total_files_scanned = 0
    total_files_processed = 0
    total_files_ignored_pattern = 0
    total_files_ignored_extension = 0
    total_successful_chunks = 0
    total_failed_chunks = 0

    logger.info(f"Scanning directory: {base_dir_path}")
    logger.debug(f"Effective ignore patterns: {effective_ignore_patterns}")
    logger.debug(f"Effective allowed extensions: {effective_allowed_extensions}")

    for file_path_obj in base_dir_path.rglob('*'):
        total_files_scanned += 1
        if not file_path_obj.is_file():
            continue

        relative_path_str = str(file_path_obj.relative_to(base_dir_path))
        
        # Check against ignore patterns
        ignored_by_pattern = False
        for pattern in effective_ignore_patterns:
            # Match against the full relative path or parts of it
            # fnmatch is good for simple glob on names, but for paths, Path.match might be tricky
            # We'll check against the full string representation of the relative path, and also individual parts
            if fnmatch.fnmatch(relative_path_str, pattern) or \
               fnmatch.fnmatch(file_path_obj.name, pattern) or \
               any(fnmatch.fnmatch(part, pattern) for part in file_path_obj.parts):
                logger.debug(f"Ignoring file '{relative_path_str}' due to pattern: {pattern}")
                total_files_ignored_pattern +=1
                ignored_by_pattern = True
                break
        if ignored_by_pattern:
            continue

        # Check extension
        if file_path_obj.suffix.lower() not in effective_allowed_extensions:
            logger.debug(f"Ignoring file '{relative_path_str}' due to disallowed extension: {file_path_obj.suffix}")
            total_files_ignored_extension += 1
            continue
        
        total_files_processed += 1
        logger.info(f"Processing local file: {relative_path_str}")
        file_stat = await _process_and_store_single_local_file(
            file_path_obj=file_path_obj,
            base_dir_path=base_dir_path,
            qdrant_client_instance=qdrant_client_instance,
            collection_name_str=collection_name_str,
            source_identifier=dir_path, # Use original dir_path as source id
            effective_chunk_size=effective_chunk_size,
            effective_chunk_overlap=effective_chunk_overlap
        )
        processed_files_summary.append(file_stat)
        total_successful_chunks += file_stat.get("successful_chunks", 0)
        total_failed_chunks += file_stat.get("failed_chunks", 0)

    final_summary = {
        "source_directory": dir_path,
        "total_files_scanned": total_files_scanned,
        "total_files_eligible_for_processing": total_files_processed, # after ignore/extension checks
        "total_files_ignored_by_pattern": total_files_ignored_pattern,
        "total_files_ignored_by_extension": total_files_ignored_extension,
        "files_processed_details": processed_files_summary,
        "total_successful_chunks_stored": total_successful_chunks,
        "total_failed_chunks_stored": total_failed_chunks
    }

    logger.info(f"Finished crawl_dir for {dir_path}. Processed {total_files_processed} files. Stored {total_successful_chunks} chunks.")
    try:
        return json.dumps(final_summary, indent=2)
    except TypeError as e:
        logger.error(f"Error serializing final summary to JSON: {e}")
        # Fallback to a string representation if JSON fails for some reason
        return str(final_summary)


# Ensure the file ends with a newline for linters 