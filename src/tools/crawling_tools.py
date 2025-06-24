"""
MCP Tools for crawling web pages, repositories, and smart URL handling.
"""
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
import os
import fnmatch
import uuid
import hashlib
import tempfile
import subprocess

from mcp.server.fastmcp.exceptions import ToolError
from qdrant_client import AsyncQdrantClient
from crawl4ai import CrawlerRunConfig, CacheMode, AsyncWebCrawler, BrowserConfig

# Import the centralized mcp instance
from src.mcp_setup import mcp
# Import utility functions
from src.utils.qdrant.setup import get_qdrant_client
from src.utils.qdrant.ingestion import store_embeddings
from src.utils.crawling_utils import (
    is_sitemap,
    is_txt,
    parse_sitemap,
    crawl_and_chunk_urls,
    CHUNK_SIZE,             # For crawl_repo default
    CHUNK_OVERLAP,          # For crawl_repo default
    MARKDOWN_CHUNK_SIZE     # For smart_crawl_url default
)
# Import logging utilities
from src.utils.logging_utils import get_logger
from src.config import settings
from src.utils.text_processing import extract_code_blocks, simple_text_chunker, smart_chunk_markdown

# --- Document Buffer for Batch Processing ---
# This buffer will hold documents from various crawling tools before they are processed and stored in a single batch.
document_buffer: List[Dict[str, Any]] = []
# -------------------------------------------

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

@mcp.tool()
async def add_to_buffer(documents: List[Dict[str, Any]]):
    """
    Adds a list of documents to the central document buffer.
    A document should be a dictionary, e.g., {'text': '...', 'source': '...'}.
    """
    if not isinstance(documents, list):
        raise ToolError("Invalid input: 'documents' must be a list of dictionaries.", "VALIDATION_ERROR")
    
    document_buffer.extend(documents)
    logger.info(f"Added {len(documents)} documents to the buffer. Current buffer size: {len(document_buffer)}")
    return {"success": True, "message": f"Added {len(documents)} documents. Buffer size: {len(document_buffer)}"}

@mcp.tool()
async def clear_buffer():
    """Clears all documents from the central buffer."""
    buffer_size = len(document_buffer)
    document_buffer.clear()
    logger.info(f"Cleared {buffer_size} documents from the buffer.")
    return {"success": True, "message": f"Cleared {buffer_size} documents."}

@mcp.tool()
async def view_buffer(limit: int = 10) -> List[Dict[str, Any]]:
    """Shows a preview of the documents currently in the buffer."""
    preview = document_buffer[:limit]
    return preview

@mcp.tool()
async def process_and_store_buffer() -> str:
    """
    Processes and stores all documents currently in the buffer into Qdrant.
    This performs embedding and batch upserting. The buffer will be cleared after processing.
    """
    if not document_buffer:
        return json.dumps({"success": True, "message": "Buffer is empty. Nothing to process."})

    buffer_size = len(document_buffer)
    logger.info(f"Processing {buffer_size} documents from the buffer.")
    
    total_successful_chunks = 0
    total_failed_chunks = 0
    
    try:
        qdrant_client_instance = get_qdrant_client()
        collection_name_str = os.getenv("QDRANT_COLLECTION")
        if not all([qdrant_client_instance, collection_name_str]):
            raise ToolError("Qdrant client or collection name is not configured.", "CONFIG_ERROR")
        assert collection_name_str is not None, "QDRANT_COLLECTION environment variable not set."

        batch_size = settings.EMBEDDING_SERVER_BATCH_SIZE
        total_batches = (buffer_size + batch_size - 1) // batch_size
        
        for i in range(0, buffer_size, batch_size):
            batch_num = (i // batch_size) + 1
            batch_docs = document_buffer[i:i + batch_size]
            
            logger.info(f"--- Processing Batch {batch_num}/{total_batches} (Size: {len(batch_docs)}) ---")
            
            try:
                successful, failed = await store_embeddings(
                    client=qdrant_client_instance,
                    collection_name=collection_name_str,
                    documents=batch_docs,
                )
                total_successful_chunks += successful
                total_failed_chunks += failed
                logger.info(f"--- Batch {batch_num}/{total_batches} completed. Successful: {successful}, Failed: {failed} ---")

            except Exception as e:
                logger.error(f"!!! Critical error processing batch {batch_num}/{total_batches}: {e} !!!")
                logger.error(f"Skipping this batch. {len(batch_docs)} documents will be lost.")
                total_failed_chunks += len(batch_docs)

        # Clear buffer after processing all batches
        document_buffer.clear()

        final_message = (
            f"Buffer processing complete. "
            f"Total Successful: {total_successful_chunks}, Total Failed: {total_failed_chunks}."
        )
        logger.info(final_message)
        return json.dumps({
            "success": True, 
            "message": final_message,
            "successful_chunks": total_successful_chunks,
            "failed_chunks": total_failed_chunks
        })

    except Exception as e:
        logger.critical(f"An unexpected error occurred during buffer processing: {e}", exc_info=True)
        # We don't clear the buffer on critical failure so it can be inspected
        return json.dumps({
            "success": False, 
            "message": f"Critical error during buffer processing: {e}"
        })

async def _process_and_store_single_repo_file(
    file_path_obj: Path,
    repo_root_path: Path,
    qdrant_client_instance: AsyncQdrantClient, # Type hint for clarity
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
                documents=code_chunks_data_for_qdrant_list
            )
            file_summary["successful_chunks"] += successful_code
            file_summary["failed_chunks"] += failed_code
            logger.debug(f"Stored code_example embeddings for repo file {relative_path}: {successful_code} successful, {failed_code} failed")

        logger.debug(f"Chunking general file content: {relative_path}")
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
            logger.debug(f"Adding {len(chunks_data_for_qdrant_list)} general content chunks to buffer from repo file: {relative_path}")
            # Add source information to each document before buffering
            for doc in chunks_data_for_qdrant_list:
                doc['source'] = f"{repo_url} (file: {str(relative_path)})"
            document_buffer.extend(chunks_data_for_qdrant_list)
            file_summary["successful_chunks"] = len(chunks_data_for_qdrant_list)
            
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
    qdrant_client_instance: AsyncQdrantClient,
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
                documents=code_chunks_data_for_qdrant_list
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
            logger.debug(f"Adding {len(chunks_data_for_qdrant_list)} general content chunks to buffer from local file: {relative_path}")
            # Add source information to each document before buffering
            for doc in chunks_data_for_qdrant_list:
                doc['source'] = f"local_dir: {source_identifier} (file: {str(relative_path)})"
            document_buffer.extend(chunks_data_for_qdrant_list)
            file_summary["successful_chunks"] += len(chunks_data_for_qdrant_list)
            
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
    Crawls a single web page and adds its content to the document buffer.
    """
    logger.info(f"Crawling single page: {url}")
    try:
        # Configuration for the crawler
        crawler = AsyncWebCrawler(
            url,
            run_config=CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                verbose=settings.CRAWLER_VERBOSE
            )
        )
        # Execute the crawl
        result = await crawler.run()
        
        if not result or not result.markdown:
            logger.warning(f"No content found for URL: {url}")
            return json.dumps({"success": False, "message": f"No content found for URL: {url}"})

        logger.info(f"Successfully crawled {url}. Preparing to add to buffer.")
        # Chunk the markdown content
        chunks = smart_chunk_markdown(result.markdown, chunk_size=MARKDOWN_CHUNK_SIZE)
        
        # Prepare documents for the buffer
        docs_to_add = [{"text": chunk, "source": url} for chunk in chunks]
        
        # Add to buffer
        document_buffer.extend(docs_to_add)

        return json.dumps({
            "success": True,
            "message": f"Successfully added {len(docs_to_add)} chunks from {url} to the buffer.",
            "source_url": url,
            "buffer_size": len(document_buffer)
        })
    except Exception as e:
        logger.error(f"Error crawling single page {url}: {e}")
        raise ToolError(f"Error crawling {url}: {e}", "CRAWL_ERROR")

@mcp.tool()
async def crawl_repo(
    repo_url: str,
    branch: Optional[str] = None,
    chunk_size: Optional[int] = None,      # User can override
    chunk_overlap: Optional[int] = None,   # User can override
    ignore_dirs: Optional[List[str]] = None # New: Directories/patterns to ignore
) -> str:
    """
    Clones a Git repository, processes specified file types (based on filters), and adds their content to the document buffer.

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
            # After processing all files, the document_buffer will be populated.
            # The user should then call process_and_store_buffer.
            final_summary = {
                "success": True,
                "message": f"Repository processing complete. {total_successful_chunks_stored} chunks added to buffer. Call process_and_store_buffer() to save to Qdrant.",
                "buffer_size": len(document_buffer)
            }
            logger.info(f"Finished processing repo {repo_url}. Total chunks added to buffer: {total_successful_chunks_stored}")
            return json.dumps(final_summary, indent=4)
    except Exception as e:
        logger.error(f"An unexpected error occurred in crawl_repo: {str(e)}", exc_info=True)
        raise ToolError(f"Failed to crawl repository {repo_url} due to an unexpected error: {str(e)}", "REPO_CRAWL_UNEXPECTED_ERROR")

@mcp.tool()
async def smart_crawl_url(url: str, max_depth: int = 3, max_concurrent: int = settings.MAX_CONCURRENT_REQUESTS, chunk_size: Optional[int] = None) -> str:
    """
    Intelligently crawl a URL based on its type and add its content to the document buffer.
    If it's a sitemap, it crawls all URLs in the sitemap.
    If it's a text file, it crawls just that file.
    Otherwise, it performs a recursive crawl up to max_depth.
    """
    logger.info(f"Smart crawling URL: {url} with max_depth={max_depth}")
    
    chunk_size_to_use = chunk_size or MARKDOWN_CHUNK_SIZE

    if is_sitemap(url):
        logger.info("Sitemap detected. Parsing and crawling all URLs found.")
        urls_to_crawl = await parse_sitemap(url)
        if not urls_to_crawl:
            return json.dumps({"success": False, "message": "Sitemap was empty or failed to parse."})
        await crawl_and_chunk_urls(urls_to_crawl, chunk_size_to_use, document_buffer)

    elif is_txt(url):
        logger.info("Plain text file detected. Crawling single file.")
        await crawl_and_chunk_urls([url], chunk_size_to_use, document_buffer)
        
    else:
        logger.info(f"Regular web page detected. Starting recursive crawl with max_depth={max_depth}")
        await recursive_crawl_url(url, max_depth, max_concurrent, chunk_size_to_use, document_buffer)

    return json.dumps({
        "success": True, 
        "message": f"Smart crawl completed for {url}. Added content to buffer.",
        "buffer_size": len(document_buffer)
    })

async def recursive_crawl_url(
    start_url: str, 
    max_depth: int, 
    max_concurrent: int, 
    chunk_size: int,
    document_buffer: List[Dict[str, Any]]
) -> None:
    """
    Recursively crawl a website starting from start_url up to max_depth.
    """
    from urllib.parse import urljoin, urlparse
    import asyncio
    
    visited_urls = set()
    urls_to_process = [(start_url, 0)]  # (url, depth)
    base_domain = urlparse(start_url).netloc
    
    crawler = AsyncWebCrawler()
    config = CrawlerRunConfig(cache_mode=CacheMode.ENABLED, stream=False)
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_url_with_depth(url: str, depth: int):
        if url in visited_urls or depth > max_depth:
            return []
            
        async with semaphore:
            visited_urls.add(url)
            logger.info(f"Crawling (depth {depth}): {url}")
            
            try:
                result = await crawler.arun(url=url, config=config)
                
                if not result.success or not result.markdown:
                    logger.warning(f"Failed to crawl {url}: {result.error_message}")
                    return []
                
                # Chunk and add to buffer
                chunks = smart_chunk_markdown(result.markdown, chunk_size)
                for i, chunk in enumerate(chunks):
                    # Generate a proper UUID for Qdrant
                    chunk_id = str(uuid.uuid4())
                    document_buffer.append({
                        "id": chunk_id,
                        "text": chunk,
                        "source": "recursive_crawl",
                        "source_path": url,
                        "metadata": {
                            "chunk_index": i, 
                            "total_chunks": len(chunks),
                            "crawl_depth": depth,
                            "original_url": url
                        }
                    })
                
                logger.info(f"Successfully crawled {url} (depth {depth}), added {len(chunks)} chunks.")
                
                # Extract links for next depth level
                if depth < max_depth and result.links:
                    new_urls = []
                    for link in result.links.get('internal', []):
                        link_url = link.get('href', '')
                        if link_url and link_url not in visited_urls:
                            # Resolve relative URLs
                            full_url = urljoin(url, link_url)
                            parsed_url = urlparse(full_url)
                            
                            # Only follow links on the same domain
                            if parsed_url.netloc == base_domain:
                                new_urls.append((full_url, depth + 1))
                    
                    return new_urls
                
                return []
                
            except Exception as e:
                logger.error(f"Error crawling {url}: {e}")
                return []
    
    # Process URLs level by level (breadth-first)
    current_level_urls = [(start_url, 0)]
    
    while current_level_urls and any(depth <= max_depth for _, depth in current_level_urls):
        # Process current level
        tasks = [process_url_with_depth(url, depth) for url, depth in current_level_urls if depth <= max_depth]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect URLs for next level
        next_level_urls = []
        for result in results:
            if isinstance(result, list):
                next_level_urls.extend(result)
        
        # Remove duplicates and already visited URLs
        unique_next_urls = []
        for url, depth in next_level_urls:
            if url not in visited_urls:
                unique_next_urls.append((url, depth))
        
        current_level_urls = unique_next_urls
        logger.info(f"Found {len(current_level_urls)} new URLs for next level")
    
    logger.info(f"Recursive crawl completed. Visited {len(visited_urls)} URLs.")

@mcp.tool()
async def crawl_dir(
    dir_path: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    ignore_patterns: Optional[List[str]] = None,
    allowed_extensions: Optional[List[str]] = None,
) -> str:
    """
    Crawls a local directory, processes specified file types, and adds their content to the document buffer.

    Args:
        dir_path: Absolute path to the local directory to crawl.
        chunk_size: Size of each text chunk in characters.
        chunk_overlap: Overlap between text chunks in characters.
        ignore_patterns: Optional list of glob-style patterns for files/directories to ignore.
        allowed_extensions: Optional list of file extensions to process.

    Returns:
        JSON string with crawl summary and storage information.
    """
    from src.utils.crawling_utils import crawl_directory as util_crawl_directory
    
    logger.info(f"Initiating directory crawl via tool: {dir_path}")

    # The utility function appends to the global document_buffer directly
    try:
        stats = util_crawl_directory(
            dir_path=dir_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            ignore_patterns=ignore_patterns,
            allowed_extensions=allowed_extensions,
            document_buffer=document_buffer # Pass the global buffer
        )
        logger.info(f"Crawl directory tool finished. Stats: {stats}")
        return json.dumps(stats)
    except Exception as e:
        logger.error(f"Error calling crawl_directory utility: {e}", exc_info=True)
        raise ToolError(f"Failed to crawl directory: {e}", "CRAWL_ERROR")


# Ensure the file ends with a newline for linters 