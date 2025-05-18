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

# Import the centralized mcp instance
from ..mcp_setup import mcp
# Import utility functions
from ..utils.qdrant_utils import store_embeddings, get_qdrant_client, QdrantClient
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
        content = await asyncio.to_thread(file_path_obj.read_text, encoding="utf-8", errors="ignore")
        file_summary["content_read"] = True
        if not content.strip():
            return file_summary

        current_file_chunks_text = simple_text_chunker(
            content,
            chunk_size=effective_chunk_size,
            chunk_overlap=effective_chunk_overlap
        )
        if not current_file_chunks_text:
            return file_summary

        for i, chunk_content_item in enumerate(current_file_chunks_text):
            meta_info = f"File: {str(relative_path)} - Chunk: {i+1}/{len(current_file_chunks_text)}"
            chunks_data_for_qdrant_list.append({"text": chunk_content_item, "headers": meta_info})
        
        if chunks_data_for_qdrant_list:
            successful, failed = await store_embeddings(
                client=qdrant_client_instance,
                collection_name=collection_name_str,
                chunks=chunks_data_for_qdrant_list,
                source_url=f"{repo_url} (file: {str(relative_path)})",
                crawl_type="repository"
            )
            file_summary["successful_chunks"] = successful
            file_summary["failed_chunks"] = failed
            
    except Exception as e_file:
        file_summary["error"] = str(e_file)
        # If error occurred after chunking, estimate all potential chunks as failed.
        # If error was during read, chunks_data_for_qdrant_list would be empty.
        file_summary["failed_chunks"] = len(chunks_data_for_qdrant_list) if chunks_data_for_qdrant_list else 0
        if not file_summary["content_read"]: # If read itself failed
             file_summary["failed_chunks"] = 1 # Mark as at least one conceptual failure for the file

    return file_summary

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
    crawler_instance = None
    qdrant_client_instance = None
    collection_name_str = None
    # Flag to indicate if we initialized the crawler in this tool call
    crawler_initialized_here = False

    try:
        # Try to get instances from context
        # This might raise ValueError if ctx.request_context is accessed inappropriately,
        # or AttributeError if subsequent attributes are missing.
        if hasattr(ctx, 'request_context') and ctx.request_context is not None: # Check first
            crawler_instance = ctx.request_context.lifespan_context.crawler
            qdrant_client_instance = ctx.request_context.lifespan_context.qdrant_client
            collection_name_str = ctx.request_context.lifespan_context.collection_name
        else:
            # Force fallback if request_context itself is not usable
            raise AttributeError("request_context not available on ctx object for crawl_single_page")
            
    except (AttributeError, ValueError) as e: # Catch both expected errors
        print(f"Context access failed for crawl_single_page ({type(e).__name__}: {e}). Initializing components from environment.")
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

            qdrant_client_instance = get_qdrant_client()
            collection_name_str = os.getenv("QDRANT_COLLECTION")
            if not collection_name_str:
                raise ValueError("QDRANT_COLLECTION environment variable must be set when context is not available.")
        except Exception as e_init:
            if crawler_initialized_here and crawler_instance:
                await crawler_instance.__aexit__(None, None, None) # Ensure cleanup if init fails mid-way
            return json.dumps({"success": False, "url": url, "error": f"Failed to initialize components: {str(e_init)}"}, indent=2)

    if not all([crawler_instance, qdrant_client_instance, collection_name_str]):
        if crawler_initialized_here and crawler_instance: # Ensure cleanup if already entered
            await crawler_instance.__aexit__(None, None, None)
        return json.dumps({"success": False, "url": url, "error": "One or more critical components (crawler, qdrant client, collection name) could not be initialized."}, indent=2)

    try:
        run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=False)
        result = await crawler_instance.arun(url=url, config=run_config)
        
        if result.success and result.markdown:
            chunks_text = smart_chunk_markdown(result.markdown) # Uses its internal default for chunk_size
            
            chunks_data_for_qdrant = []
            for i, chunk_content in enumerate(chunks_text):
                meta = extract_section_info(chunk_content)
                chunks_data_for_qdrant.append({
                    "text": chunk_content,
                    "headers": meta.get("headers", ""),
                })

            successful_chunks, failed_chunks = await store_embeddings(
                client=qdrant_client_instance,
                collection_name=collection_name_str,
                chunks=chunks_data_for_qdrant,
                source_url=url,
                crawl_type="single_page"
            )
            
            return json.dumps({
                "success": True, "url": url, "chunks_processed": len(chunks_text),
                "successful_chunks": successful_chunks, "failed_chunks": failed_chunks,
                "content_length": len(result.markdown),
                "links_count": {
                    "internal": len(result.links.get("internal", [])),
                    "external": len(result.links.get("external", []))
                }
            }, indent=2)
        else:
            return json.dumps({"success": False, "url": url, "error": result.error_message}, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "url": url, "error": str(e)}, indent=2)
    finally:
        if crawler_initialized_here and crawler_instance:
            await crawler_instance.__aexit__(None, None, None)

@mcp.tool()
async def crawl_repo(
    ctx: Context,
    repo_url: str,
    branch: Optional[str] = None,
    chunk_size: Optional[int] = None,      # User can override
    chunk_overlap: Optional[int] = None,   # User can override
    ignore_dirs: Optional[List[str]] = None, # New: Directories/patterns to ignore
    allowed_extensions: Optional[List[str]] = None # New: Specific file extensions to process
) -> str:
    """
    Clones a Git repository, processes specified file types (based on filters), and stores their content in Qdrant.

    Args:
        ctx: The MCP server provided context.
        repo_url: URL of the Git repository to crawl.
        branch: Optional specific branch to clone. Defaults to the repository's default branch.
        chunk_size: Size of each text chunk in characters for processing. Defaults to CHUNK_SIZE from crawling_utils.
        chunk_overlap: Overlap between text chunks in characters. Defaults to CHUNK_OVERLAP from crawling_utils.
        ignore_dirs: Optional list of directory names or path patterns to ignore (e.g., [".git", "node_modules", "dist/"]).
        allowed_extensions: Optional list of file extensions to process (e.g., [".py", ".js", ".md"]).
                          If None or empty, all files not in ignored_dirs will be considered (after extension check).

    Returns:
        JSON string with crawl summary and storage information.
    """
    effective_chunk_size = chunk_size if chunk_size is not None else CHUNK_SIZE
    effective_chunk_overlap = chunk_overlap if chunk_overlap is not None else CHUNK_OVERLAP
    
    # Default ignored directories/patterns if none provided
    default_ignore_dirs = [".git", "node_modules", "__pycache__", ".venv", "venv", "env", "dist", "build", "target", ".DS_Store"]
    final_ignore_dirs = ignore_dirs if ignore_dirs is not None else default_ignore_dirs
    
    # Normalize allowed extensions (e.g., ensure they start with a dot)
    final_allowed_extensions = []
    if allowed_extensions:
        for ext in allowed_extensions:
            final_allowed_extensions.append(ext if ext.startswith('.') else '.' + ext)

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
        print(f"Context access failed for crawl_repo ({type(e).__name__}: {e}). Initializing Qdrant client from environment.")
        try:
            qdrant_client_instance = get_qdrant_client()
            collection_name_str = os.getenv("QDRANT_COLLECTION")
            if not collection_name_str:
                raise ValueError("QDRANT_COLLECTION environment variable must be set when context is not available.")
        except Exception as e_init:
            return json.dumps({"success": False, "repo_url": repo_url, "error": f"Failed to initialize Qdrant components: {str(e_init)}"}, indent=2)

    if not all([qdrant_client_instance, collection_name_str]):
        return json.dumps({"success": False, "repo_url": repo_url, "error": "Qdrant client or collection name could not be initialized."}, indent=2)

    try:
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
                return json.dumps({"success": False, "repo_url": repo_url, "error": f"Git clone failed: {error_message.strip()}"}, indent=2)

            all_file_paths_to_process = []
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

                        # Check allowed extensions (if any are specified)
                        if final_allowed_extensions: # Only apply extension filter if list is not empty
                            if fp_obj.suffix.lower() not in final_allowed_extensions:
                                continue
                        
                        all_file_paths_to_process.append(fp_obj)
                    except Exception as path_err:
                        # Handle cases where relative_to might fail or other path issues
                        print(f"Skipping file {fp_obj} due to path processing error: {path_err}")
                        continue
            
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
            # return_exceptions=True ensures that if one task fails, others can complete,
            # and the exception is returned as a result for that task.
            results_of_tasks = await asyncio.gather(*tasks, return_exceptions=True)

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
            return json.dumps(summary, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "repo_url": repo_url, "error": f"An unexpected error occurred in crawl_repo: {str(e)}"}, indent=2)

@mcp.tool()
async def smart_crawl_url(ctx: Context, url: str, max_depth: int = 3, max_concurrent: int = 30, chunk_size: Optional[int] = None) -> str:
    """
    Intelligently crawl a URL based on its type and store content in Qdrant.
    Args:
        chunk_size: Max size of each markdown content chunk. Defaults to MARKDOWN_CHUNK_SIZE from crawling_utils.
    """
    effective_markdown_chunk_size = chunk_size if chunk_size is not None else MARKDOWN_CHUNK_SIZE
    
    crawler_instance = None
    qdrant_client_instance = None
    collection_name_str = None
    crawler_initialized_here = False # Flag for standalone crawler cleanup

    try:
        # Try to get instances from context
        if hasattr(ctx, 'request_context') and ctx.request_context is not None:
            crawler_instance = ctx.request_context.lifespan_context.crawler
            qdrant_client_instance = ctx.request_context.lifespan_context.qdrant_client
            collection_name_str = ctx.request_context.lifespan_context.collection_name
        else:
            raise AttributeError("request_context not available on ctx object for smart_crawl_url")
            
    except (AttributeError, ValueError) as e: # Catch both expected errors
        print(f"Context access failed for smart_crawl_url ({type(e).__name__}: {e}). Initializing components from environment.")
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
            return json.dumps({"success": False, "url": url, "error": f"Failed to initialize components: {str(e_init)}"}, indent=2)

    if not all([crawler_instance, qdrant_client_instance, collection_name_str]):
        if crawler_initialized_here and crawler_instance:
            await crawler_instance.__aexit__(None, None, None)
        return json.dumps({"success": False, "url": url, "error": "One or more critical components (crawler, qdrant client, collection name) could not be initialized."}, indent=2)

    try:
        crawl_results = []
        crawl_type = "webpage"
        
        if is_txt(url):
            crawl_results = await crawl_markdown_file(crawler_instance, url)
            crawl_type = "text_file"
        elif is_sitemap(url):
            sitemap_urls = parse_sitemap(url)
            if not sitemap_urls:
                return json.dumps({"success": False, "url": url, "error": "No URLs found in sitemap"}, indent=2)
            crawl_results = await crawl_batch(crawler_instance, sitemap_urls, max_concurrent=max_concurrent)
            crawl_type = "sitemap"
        else:
            crawl_results = await crawl_recursive_internal_links(crawler_instance, [url], max_depth=max_depth, max_concurrent=max_concurrent)
            # target_domain could be passed to crawl_recursive_internal_links if needed
            crawl_type = "webpage"
        
        if not crawl_results:
            return json.dumps({"success": False, "url": url, "error": "No content found"}, indent=2)
        
        processed_urls = set()
        total_successful_chunks = 0
        total_failed_chunks = 0

        for doc_data in crawl_results:
            source_url = doc_data['url']
            markdown_content = doc_data['markdown']
            processed_urls.add(source_url)
            
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
        
        return json.dumps({
            "success": True, "url": url, "crawl_type": crawl_type,
            "pages_crawled": len(processed_urls),
            "total_successful_chunks": total_successful_chunks,
            "total_failed_chunks": total_failed_chunks,
            "urls_crawled_sample": list(processed_urls)[:5] + (["..."] if len(processed_urls) > 5 else [])
        }, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "url": url, "error": str(e)}, indent=2)
    finally:
        if crawler_initialized_here and crawler_instance:
            await crawler_instance.__aexit__(None, None, None)

# Ensure the file ends with a newline for linters 