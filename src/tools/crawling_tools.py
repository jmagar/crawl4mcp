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

from mcp.server.fastmcp import Context # MCP Context for tool arguments
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
    chunks_data_for_qdrant_list: List[Dict[str, str]] = [] # Ensure it's defined for the error case

    try:
        logger.debug(f"Reading file: {relative_path}")
        content = await asyncio.to_thread(file_path_obj.read_text, encoding="utf-8", errors="ignore")
        file_summary["content_read"] = True
        if not content.strip():
            logger.debug(f"File is empty: {relative_path}")
            return file_summary

        logger.debug(f"Chunking file content: {relative_path}")
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
        
        if chunks_data_for_qdrant_list:
            logger.debug(f"Storing {len(chunks_data_for_qdrant_list)} chunks for file: {relative_path}")
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

@mcp.tool(
    annotations={
        "title": "Crawl Single Web Page",
        "readOnlyHint": False,
    }
)
async def crawl_single_page(url: str, ctx: Optional[Context] = None) -> str:
    """
    Crawl a single web page and store its content in Qdrant.
    
    This tool is ideal for quickly retrieving content from a specific URL without following links.
    The content is stored in Qdrant for later retrieval and querying.
    
    Args:
        url: URL of the web page to crawl
        ctx: The MCP server provided context (optional)
    
    Returns:
        Summary of the crawling operation and storage in Qdrant
    """
    logger.info(f"Starting crawl_single_page for URL: {url}")
    crawler_instance = None
    qdrant_client_instance = None
    collection_name_str = None
    # Flag to indicate if we initialized the crawler in this tool call
    crawler_initialized_here = False

    # Create a dummy ctx.log and ctx.report_progress if ctx is None
    class DummyLogger:
        def info(self, message):
            logger.info(message)
        def debug(self, message):
            logger.debug(message)
        def warning(self, message):
            logger.warning(message)
        def error(self, message):
            logger.error(message)
    
    class DummyContext:
        def __init__(self):
            self.log = DummyLogger()
        
        def report_progress(self, progress, total, message=None, parent_step=None, total_parent_steps=None):
            logger.info(f"Progress: {progress}/{total} - {message if message else ''}")
    
    # If ctx is None, create a dummy context
    if ctx is None:
        ctx = DummyContext()
        logger.warning("Context not available for crawl_single_page. Using dummy context.")

    # Handle case when ctx is empty or doesn't have request_context
    if not hasattr(ctx, 'request_context') or ctx.request_context is None:
        logger.warning("No request_context available. Initializing components from environment.")
        try:
            # Initialize AsyncWebCrawler
            # Create browser configuration for standalone crawler
            browser_config = BrowserConfig(
                headless=True,
                verbose=os.getenv("CRAWLER_VERBOSE", "False").lower() == "true"
            )
            crawler_instance = AsyncWebCrawler(config=browser_config)
            await crawler_instance.__aenter__() # Manually enter context
            crawler_initialized_here = True
            logger.debug("Initialized standalone crawler")

            qdrant_client_instance = get_qdrant_client()
            collection_name_str = os.getenv("QDRANT_COLLECTION")
            if not collection_name_str:
                logger.error("QDRANT_COLLECTION environment variable must be set")
                # Changed error handling
                raise ToolError(f"QDRANT_COLLECTION environment variable must be set when context is not available.", "CONFIG_ERROR")
            logger.debug(f"Using Qdrant client from environment with collection {collection_name_str}")
        except Exception as e_init:
            logger.error(f"Failed to initialize components: {e_init}")
            if crawler_initialized_here and crawler_instance:
                await crawler_instance.__aexit__(None, None, None) # Ensure cleanup if init fails mid-way
            # Changed error handling
            raise ToolError(f"Failed to initialize components: {str(e_init)}", "INITIALIZATION_ERROR", {"original_exception": str(e_init)})
    else:
        try:
            # Try to get instances from context
            crawler_instance = ctx.request_context.lifespan_context.crawler
            qdrant_client_instance = ctx.request_context.lifespan_context.qdrant_client
            collection_name_str = ctx.request_context.lifespan_context.collection_name
            logger.debug("Using components from request context lifespan")
        except (AttributeError, ValueError) as e: # Catch both expected errors
            logger.warning(f"Context access failed for crawl_single_page ({type(e).__name__}: {e}). Initializing components from environment.")
            try:
                # Initialize AsyncWebCrawler
                # Create browser configuration for standalone crawler
                browser_config = BrowserConfig(
                    headless=True,
                    verbose=os.getenv("CRAWLER_VERBOSE", "False").lower() == "true"
                )
                crawler_instance = AsyncWebCrawler(config=browser_config)
                await crawler_instance.__aenter__() # Manually enter context
                crawler_initialized_here = True
                logger.debug("Initialized standalone crawler")

                qdrant_client_instance = get_qdrant_client()
                collection_name_str = os.getenv("QDRANT_COLLECTION")
                if not collection_name_str:
                    logger.error("QDRANT_COLLECTION environment variable must be set")
                    # Changed error handling
                    raise ToolError(f"QDRANT_COLLECTION environment variable must be set when context is not available.", "CONFIG_ERROR")
                logger.debug(f"Using Qdrant client from environment with collection {collection_name_str}")
            except Exception as e_init:
                logger.error(f"Failed to initialize components: {e_init}")
                if crawler_initialized_here and crawler_instance:
                    await crawler_instance.__aexit__(None, None, None) # Ensure cleanup if init fails mid-way
                # Changed error handling
                raise ToolError(f"Failed to initialize components: {str(e_init)}", "INITIALIZATION_ERROR", {"original_exception": str(e_init)})

    if not all([crawler_instance, qdrant_client_instance, collection_name_str]):
        logger.error("One or more critical components (crawler, qdrant client, collection name) could not be initialized")
        if crawler_initialized_here and crawler_instance: # Ensure cleanup if already entered
            await crawler_instance.__aexit__(None, None, None)
        # Changed error handling
        raise ToolError(f"One or more critical components (crawler, qdrant client, collection name) could not be initialized.", "MISSING_DEPENDENCY")

    try:
        ctx.log.info(f"Crawling page: {url}")
        ctx.report_progress(1, 3, f"Fetching content from {url}")
        logger.info(f"Crawling page: {url}")
        # Configure crawler run for single page
        run_config = CrawlerRunConfig(
            page_timeout=60000,  # Increased timeout to 60 seconds (was 60)
            cache_mode=CacheMode.BYPASS  # Bypass cache for direct calls
        )
        
        # Get the raw HTML or content from the web page
        # Assuming arun might return a list of results, even for a single URL
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
            logger.warning(f"No markdown content retrieved from URL: {url}. Response status: {response.status_code if hasattr(response, 'status_code') else 'N/A'}, Error: {response.error_message if hasattr(response, 'error_message') else 'N/A'}")
            raise ToolError(f"No markdown content retrieved from URL: {url}", "NO_CONTENT")
        else:
            # Use response.markdown for further processing
            logger.debug(f"Successfully retrieved markdown content from {url} ({len(response.markdown)} bytes)")
            ctx.report_progress(2, 3, f"Processing content from {url}")
            
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
                ctx.log.info(f"Storing {len(chunks_data_for_qdrant)} chunks from {url}")
                ctx.report_progress(3, 3, f"Storing {len(chunks_data_for_qdrant)} chunks from {url}")
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
        # Changed error handling
        raise ToolError(f"Error during crawl_single_page for {url}: {str(e)}", "CRAWL_ERROR", {"url": url, "original_exception": str(e)})
    finally:
        # Only clean up crawler if we initialized it in this call
        if crawler_initialized_here and crawler_instance:
            logger.debug("Cleaning up standalone crawler")
            await crawler_instance.__aexit__(None, None, None)
            
    # Return formatted result as JSON string
    return json.dumps(result, indent=2)

@mcp.tool(
    annotations={
        "title": "Crawl Git Repository",
        "readOnlyHint": False,
    }
)
async def crawl_repo(
    repo_url: str,
    branch: Optional[str] = None,
    chunk_size: Optional[int] = None,      # User can override
    chunk_overlap: Optional[int] = None,   # User can override
    ignore_dirs: Optional[List[str]] = None, # New: Directories/patterns to ignore
    ctx: Optional[Context] = None
) -> str:
    """
    Clones a Git repository, processes specified file types (based on filters), and stores their content in Qdrant.

    Args:
        repo_url: URL of the Git repository to crawl.
        branch: Optional specific branch to clone. Defaults to the repository's default branch.
        chunk_size: Size of each text chunk in characters for processing. Defaults to CHUNK_SIZE from crawling_utils.
        chunk_overlap: Overlap between text chunks in characters. Defaults to CHUNK_OVERLAP from crawling_utils.
        ignore_dirs: Optional list of directory names or path patterns to ignore (e.g., [".git", "node_modules", "dist/"]).
        ctx: The MCP server provided context (optional).

    Returns:
        JSON string with crawl summary and storage information.
    """
    # Create a dummy ctx.log and ctx.report_progress if ctx is None
    class DummyLogger:
        def info(self, message):
            logger.info(message)
        def debug(self, message):
            logger.debug(message)
        def warning(self, message):
            logger.warning(message)
        def error(self, message):
            logger.error(message)
    
    class DummyContext:
        def __init__(self):
            self.log = DummyLogger()
        
        def report_progress(self, progress, total, message=None, parent_step=None, total_parent_steps=None):
            logger.info(f"Progress: {progress}/{total} - {message if message else ''}")
    
    # If ctx is None, create a dummy context
    if ctx is None:
        ctx = DummyContext()
        logger.warning("Context not available for crawl_repo. Using dummy context.")

    effective_chunk_size = chunk_size if chunk_size is not None else CHUNK_SIZE
    effective_chunk_overlap = chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP
    
    # Default ignored directories/patterns if none provided
    default_ignore_dirs = [".git", "node_modules", "__pycache__", ".venv", "venv", "env", "dist", "build", "target", ".DS_Store"]
    final_ignore_dirs = ignore_dirs if ignore_dirs is not None else default_ignore_dirs

    qdrant_client_instance = None
    collection_name_str = None

    try:
        # Try to get Qdrant client and collection name from context
        if hasattr(ctx, 'request_context') and ctx.request_context is not None:
            qdrant_client_instance = ctx.request_context.lifespan_context.qdrant_client
            collection_name_str = ctx.request_context.lifespan_context.collection_name
        else:
            raise AttributeError("request_context not available on ctx object for crawl_repo")

    except (AttributeError, ValueError) as e: # Catch both expected errors
        logger.warning(f"Context access failed for crawl_repo ({type(e).__name__}: {e}). Initializing Qdrant client from environment.")
        try:
            qdrant_client_instance = get_qdrant_client()
            collection_name_str = os.getenv("QDRANT_COLLECTION")
            if not collection_name_str:
                raise ValueError("QDRANT_COLLECTION environment variable must be set when context is not available.")
        except Exception as e_init:
            # Changed error handling
            raise ToolError(f"Failed to initialize Qdrant components: {str(e_init)}", "INITIALIZATION_ERROR", {"original_exception": str(e_init)})

    if not all([qdrant_client_instance, collection_name_str]):
        # Changed error handling
        raise ToolError(f"Qdrant client or collection name could not be initialized.", "MISSING_DEPENDENCY")

    try:
        ctx.log.info(f"Starting repository crawl for {repo_url}")
        ctx.report_progress(1, 4, f"Cloning repository {repo_url}")
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
                # Changed error handling
                raise ToolError(f"Git clone failed: {error_message.strip()}", "GIT_CLONE_ERROR", {"repo_url": repo_url, "stderr": process.stderr, "stdout": process.stdout})

            all_file_paths_to_process = []
            ctx.log.info(f"Identifying files to process...")
            ctx.report_progress(2, 4, "Identifying files")
            for fp_obj in repo_path.rglob("*"):
                if fp_obj.is_file():
                    # Apply filtering
                    try:
                        relative_path_str = str(fp_obj.relative_to(repo_path))
                        path_parts = set(Path(relative_path_str).parts) # Use set for efficient "in" check
                        
                        # Check ignored directories
                        skip_due_to_ignore = False
                        for ignored_pattern in final_ignore_dirs:
                            # Simple check: if any part of the path is the ignored_pattern or starts with it (for top-level folders)
                            if ignored_pattern in path_parts:
                                skip_due_to_ignore = True
                                break
                            # More complex pattern matching could be added here if needed (e.g. fnmatch)
                            # For now, we check if any directory component *is* the ignored pattern,
                            # or if the relative path *starts with* the ignored pattern (e.g. "dist/" for "dist/somefile.js")
                            if relative_path_str.startswith(ignored_pattern + os.path.sep) or relative_path_str == ignored_pattern:
                                skip_due_to_ignore = True
                                break
                        if skip_due_to_ignore:
                            continue
                        
                        all_file_paths_to_process.append(fp_obj)
                    except Exception as path_err:
                        # Handle cases where relative_to might fail or other path issues
                        logger.warning(f"Skipping file {fp_obj} due to path processing error: {path_err}")
                        continue
            
            ctx.log.info(f"Found {len(all_file_paths_to_process)} files to process. Starting concurrent processing.")
            ctx.report_progress(3, 4, f"Processing {len(all_file_paths_to_process)} files")
            tasks = []
            for idx, fp_obj in enumerate(all_file_paths_to_process):
                # Report progress for file processing initiation if there are many files
                if len(all_file_paths_to_process) > 10 and idx % (len(all_file_paths_to_process) // 10) == 0:
                    ctx.report_progress(idx, len(all_file_paths_to_process), f"Queueing file {idx+1}/{len(all_file_paths_to_process)} for processing", parent_step=3, total_parent_steps=4)

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
            # return_exceptions=True ensures that if one task fails, others can complete,
            # and the exception is returned as a result for that task.
            results_of_tasks = await asyncio.gather(*tasks, return_exceptions=True)

            ctx.log.info("Aggregating results from file processing...")
            ctx.report_progress(4, 4, "Aggregating results")
            # Aggregate results
            files_processed_successfully_count = 0
            total_successful_chunks_stored = 0
            total_failed_chunks_during_storage = 0
            files_with_processing_errors_list = []
            processed_file_paths_list = []

            for result_item in results_of_tasks:
                if isinstance(result_item, Exception):
                    # This means the _process_and_store_single_repo_file task itself had an unhandled exception
                    # or was cancelled. We need a way to associate this with a file path if possible,
                    # but it's hard if the task failed before even getting the path.
                    # For now, log a generic task failure.
                    files_with_processing_errors_list.append(f"A file processing task failed: {str(result_item)}")
                    continue

                # result_item is the dict from _process_and_store_single_repo_file
                file_path_str = result_item["path"]
                
                if result_item["error"]:
                    files_with_processing_errors_list.append(f"{file_path_str}: {result_item['error']}")
                # Check if content was read and there were no errors before counting as "processed successfully"
                # And if successful chunks were stored.
                elif result_item["content_read"] and result_item["successful_chunks"] > 0 :
                    processed_file_paths_list.append(file_path_str)
                    files_processed_successfully_count += 1
                elif result_item["content_read"] and result_item["successful_chunks"] == 0 and result_item["failed_chunks"] == 0 and not result_item["error"]:
                    # This means the file was read, but it was empty or yielded no chunks, and no errors. Still counts as "processed".
                    processed_file_paths_list.append(file_path_str) # Add to processed paths
                    files_processed_successfully_count +=1 # Count as processed

                total_successful_chunks_stored += result_item["successful_chunks"]
                total_failed_chunks_during_storage += result_item["failed_chunks"]
            
            summary = {
                "success": True, # Overall success, individual file errors are listed
                "repo_url": repo_url,
                "branch_crawled": branch if branch else "default",
                "files_attempted": len(all_file_paths_to_process),
                "files_processed_successfully": files_processed_successfully_count,
                "processed_file_paths": processed_file_paths_list,
                "total_successful_chunks_stored": total_successful_chunks_stored,
                "total_failed_chunks_during_storage": total_failed_chunks_during_storage,
                "files_with_processing_errors": files_with_processing_errors_list
            }
            ctx.log.info(f"Repository crawl for {repo_url} complete.")
            return json.dumps(summary, indent=4)
    except Exception as e:
        # Changed error handling
        logger.error(f"An unexpected error occurred in crawl_repo: {str(e)}", exc_info=True)
        raise ToolError(f"An unexpected error occurred in crawl_repo: {str(e)}", "REPO_CRAWL_UNEXPECTED_ERROR", {"repo_url": repo_url, "original_exception": str(e)})

@mcp.tool(
    annotations={
        "title": "Smart Crawl URL (Webpage, Sitemap, Feed, etc.)",
        "readOnlyHint": False,
    }
)
async def smart_crawl_url(url: str, max_depth: int = 3, max_concurrent: int = 30, chunk_size: Optional[int] = None, ctx: Optional[Context] = None) -> str:
    """
    Intelligently crawl a URL based on its type and store content in Qdrant.
    Args:
        url: URL to crawl
        max_depth: Maximum depth for recursive crawls (default: 3)
        max_concurrent: Maximum concurrent requests (default: 30)
        chunk_size: Max size of each markdown content chunk. Defaults to MARKDOWN_CHUNK_SIZE from crawling_utils.
        ctx: The MCP server provided context (optional)
    """
    effective_markdown_chunk_size = chunk_size if chunk_size is not None else MARKDOWN_CHUNK_SIZE
    
    crawler_instance = None
    qdrant_client_instance = None
    collection_name_str = None
    crawler_initialized_here = False # Flag for standalone crawler cleanup

    # Create a dummy ctx.log and ctx.report_progress if ctx is None
    class DummyLogger:
        def info(self, message):
            logger.info(message)
        def debug(self, message):
            logger.debug(message)
        def warning(self, message):
            logger.warning(message)
        def error(self, message):
            logger.error(message)
    
    class DummyContext:
        def __init__(self):
            self.log = DummyLogger()
        
        def report_progress(self, progress, total, message=None, parent_step=None, total_parent_steps=None):
            logger.info(f"Progress: {progress}/{total} - {message if message else ''}")
    
    # If ctx is None, create a dummy context
    if ctx is None:
        ctx = DummyContext()
        logger.warning("Context not available for smart_crawl_url. Using dummy context.")

    # Handle case when ctx is empty or doesn't have request_context
    if not hasattr(ctx, 'request_context') or ctx.request_context is None:
        logger.warning("No request_context available. Initializing components from environment.")
        try:
            browser_config = BrowserConfig(
                headless=True,
                verbose=os.getenv("CRAWLER_VERBOSE", "False").lower() == "true"
            )
            crawler_instance = AsyncWebCrawler(config=browser_config)
            await crawler_instance.__aenter__()
            crawler_initialized_here = True

            qdrant_client_instance = get_qdrant_client()
            collection_name_str = os.getenv("QDRANT_COLLECTION")
            if not collection_name_str:
                raise ValueError("QDRANT_COLLECTION environment variable must be set when context is not available.")
        except Exception as e_init:
            if crawler_initialized_here and crawler_instance:
                await crawler_instance.__aexit__(None, None, None)
            # Changed error handling
            raise ToolError(f"Failed to initialize components: {str(e_init)}", "INITIALIZATION_ERROR", {"original_exception": str(e_init)})
    else:
        try:
            # Try to get instances from context
            crawler_instance = ctx.request_context.lifespan_context.crawler
            qdrant_client_instance = ctx.request_context.lifespan_context.qdrant_client
            collection_name_str = ctx.request_context.lifespan_context.collection_name
            logger.debug("Using components from request context lifespan")
        except (AttributeError, ValueError) as e: # Catch both expected errors
            logger.warning(f"Context access failed for smart_crawl_url ({type(e).__name__}: {e}). Initializing components from environment.")
            try:
                browser_config = BrowserConfig(
                    headless=True,
                    verbose=os.getenv("CRAWLER_VERBOSE", "False").lower() == "true"
                )
                crawler_instance = AsyncWebCrawler(config=browser_config)
                await crawler_instance.__aenter__()
                crawler_initialized_here = True

                qdrant_client_instance = get_qdrant_client()
                collection_name_str = os.getenv("QDRANT_COLLECTION")
                if not collection_name_str:
                    raise ValueError("QDRANT_COLLECTION environment variable must be set when context is not available.")
            except Exception as e_init:
                if crawler_initialized_here and crawler_instance:
                    await crawler_instance.__aexit__(None, None, None)
                # Changed error handling
                raise ToolError(f"Failed to initialize components: {str(e_init)}", "INITIALIZATION_ERROR", {"original_exception": str(e_init)})

    if not all([crawler_instance, qdrant_client_instance, collection_name_str]):
        if crawler_initialized_here and crawler_instance:
            await crawler_instance.__aexit__(None, None, None)
        # Changed error handling
        raise ToolError(f"One or more critical components (crawler, qdrant client, collection name) could not be initialized.", "MISSING_DEPENDENCY")

    try:
        crawl_results = []
        crawl_type = "webpage"
        
        ctx.log.info(f"Smart crawling URL: {url} (max_depth={max_depth}, max_concurrent={max_concurrent})")
        ctx.report_progress(1, 4, f"Determining crawl type for {url}")

        if is_txt(url):
            ctx.log.info(f"URL recognized as text/markdown file: {url}")
            ctx.report_progress(2, 4, f"Crawling text file: {url}")
            crawl_results = await crawl_markdown_file(crawler_instance, url)
            crawl_type = "text_file"
        elif is_sitemap(url):
            ctx.log.info(f"URL recognized as sitemap: {url}")
            ctx.report_progress(2, 4, f"Parsing sitemap: {url}")
            sitemap_urls = parse_sitemap(url)
            if not sitemap_urls:
                # Changed error handling
                raise ToolError(f"No URLs found in sitemap: {url}", "SITEMAP_EMPTY")
            ctx.log.info(f"Found {len(sitemap_urls)} URLs in sitemap. Starting batch crawl.")
            ctx.report_progress(2, 4, f"Batch crawling {len(sitemap_urls)} URLs from sitemap")
            crawl_results = await crawl_batch(crawler_instance, sitemap_urls, max_concurrent=max_concurrent)
            crawl_type = "sitemap"
        else:
            ctx.log.info(f"URL recognized as standard webpage. Starting recursive crawl: {url}")
            ctx.report_progress(2, 4, f"Recursively crawling from {url} (max_depth={max_depth})")
            crawl_results = await crawl_recursive_internal_links(crawler_instance, [url], max_depth=max_depth, max_concurrent=max_concurrent)
            # target_domain could be passed to crawl_recursive_internal_links if needed
            crawl_type = "webpage"
        
        ctx.report_progress(3, 4, "Processing crawled content")
        if not crawl_results:
            # Changed error handling
            raise ToolError(f"No content found for URL {url} with type {crawl_type}", "NO_CONTENT_FOUND")
        
        processed_urls = set()
        total_successful_chunks = 0
        total_failed_chunks = 0

        for i, doc_data in enumerate(crawl_results):
            source_url = doc_data['url']
            markdown_content = doc_data['markdown']
            processed_urls.add(source_url)
            
            ctx.log.info(f"Processing content from {source_url} ({i+1}/{len(crawl_results)})")
            if len(crawl_results) > 1:
                 ctx.report_progress(i, len(crawl_results), f"Processing page {i+1}/{len(crawl_results)}: {source_url}", parent_step=3, total_parent_steps=4)

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
        
        ctx.report_progress(4, 4, "Crawl and processing complete")
        ctx.log.info(f"Smart crawl for {url} complete. Processed {len(processed_urls)} URLs, stored {total_successful_chunks} chunks.")
        return json.dumps({
            "success": True, "url": url, "crawl_type": crawl_type,
            "pages_crawled": len(processed_urls),
            "total_successful_chunks": total_successful_chunks,
            "total_failed_chunks": total_failed_chunks,
            "urls_crawled_sample": list(processed_urls)[:5] + (["..."] if len(processed_urls) > 5 else [])
        }, indent=2)
    except Exception as e:
        # Changed error handling
        logger.error(f"Error during smart_crawl_url for {url}: {str(e)}", exc_info=True)
        raise ToolError(f"Error during smart_crawl_url for {url}: {str(e)}", "SMART_CRAWL_ERROR", {"url": url, "original_exception": str(e)})
    finally:
        if crawler_initialized_here and crawler_instance:
            await crawler_instance.__aexit__(None, None, None)

# Ensure the file ends with a newline for linters 