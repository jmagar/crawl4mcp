# /mnt/user/compose/crawl4mcp/src/mcp_setup.py

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional
import os
import torch

# USER: Please ensure 'settings' is imported correctly for your project.
# Common patterns include:
# from . import settings
# from .config import settings  # This is used as a placeholder below
# Verify this line matches your project structure:
from src.config import settings

from fastapi import FastAPI
from fastmcp.server import FastMCP
from qdrant_client import AsyncQdrantClient
from sentence_transformers import CrossEncoder

# Assuming these module paths are correct relative to this file
from crawl4ai import AsyncWebCrawler  # Import from installed package
from src.utils.logging_utils import LogAccessor  # Import from local utils

# --- Logging Setup ---
# Configure logging using the settings object.
log_level = getattr(settings, 'LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

# --- Resource Management ---
class AppResources:
    qdrant_client: Optional[AsyncQdrantClient] = None
    web_crawler: Optional[AsyncWebCrawler] = None
    log_accessor: Optional[LogAccessor] = None
    reranker: Optional[CrossEncoder] = None

resources = AppResources()

@asynccontextmanager
async def mcp_resource_lifespan(app: Optional[FastAPI] = None) -> AsyncIterator[None]: # app arg for FastAPI compatibility if needed, but not used by FastMCP directly
    """
    Async context manager for initializing and cleaning up MCP resources.
    Intended to be used with FastMCP's lifespan argument.
    """
    logger.info("MCP server starting up (FastMCP lifespan)...Initializing resources...")
    try:
        qdrant_url = settings.QDRANT_URL
        qdrant_api_key = getattr(settings, 'QDRANT_API_KEY', None)
        crawler_verbose = getattr(settings, 'CRAWLER_VERBOSE', False)
        reranker_model_name = getattr(settings, 'RERANKER_MODEL_NAME', "cross-encoder/ms-marco-MiniLM-L-6-v2")
        use_reranking = getattr(settings, 'USE_RERANKING', True)
        log_dir = getattr(settings, 'LOG_DIR', '/app/logs')

        resources.qdrant_client = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        logger.info(f"Qdrant client initialized: {qdrant_url}")

        resources.web_crawler = AsyncWebCrawler(verbose=crawler_verbose)
        logger.info(f"AsyncWebCrawler initialized (verbose: {crawler_verbose})")

        import inspect
        logger.info(f"DEBUG: LogAccessor type: {type(LogAccessor)}")
        logger.info(f"DEBUG: LogAccessor module: {LogAccessor.__module__}")
        try:
            logger.info(f"DEBUG: LogAccessor.__init__ signature: {inspect.signature(LogAccessor.__init__)}")
        except Exception as e_inspect:
            logger.info(f"DEBUG: Could not inspect LogAccessor.__init__: {e_inspect}")
        resources.log_accessor = LogAccessor(log_directory=log_dir)
        logger.info(f"LogAccessor initialized (log_dir: {log_dir})")

        resources.reranker = None
        if use_reranking and reranker_model_name:
            logger.info(f"Reranking enabled. Loading model: {reranker_model_name}")
            try:
                # Detect GPU availability and set device
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"CrossEncoder device selected: {device}")
                if device == "cuda":
                    logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
                
                resources.reranker = await asyncio.to_thread(
                    CrossEncoder, reranker_model_name, max_length=512, device=device
                )
                logger.info(f"Reranker model '{reranker_model_name}' loaded successfully on {device}.")
            except Exception as e_model_load:
                logger.error(f"Failed to load reranker model '{reranker_model_name}': {e_model_load}. Reranking disabled.", exc_info=True)
                resources.reranker = None
        elif not use_reranking:
            logger.info("Reranking is disabled in settings.")
        else:
            logger.warning("Reranking is enabled, but no RERANKER_MODEL_NAME is set in settings. Reranking disabled.")

        logger.info("All core resources initialized.")
        yield # Lifespan is active here

    except AttributeError as e_attr:
        logger.critical(f"A required setting ('{e_attr.name}') is missing for initialization. Server cannot start.", exc_info=True)
        # No yield, so server won't start if this fails
        raise RuntimeError(f"Configuration error: Missing setting '{e_attr.name}'") from e_attr
    except Exception as e_init:
        logger.critical(f"Fatal error during core resource initialization: {e_init}", exc_info=True)
        # No yield, so server won't start if this fails
        raise RuntimeError(f"Core resource initialization failed: {e_init}") from e_init
    finally:
        logger.info("MCP server shutting down (FastMCP lifespan). Cleaning up resources...")
        if resources.qdrant_client:
            await resources.qdrant_client.close()
            logger.info("Qdrant client closed.")
        if resources.web_crawler and hasattr(resources.web_crawler, 'close'):
            try:
                await resources.web_crawler.close()
                logger.info("AsyncWebCrawler closed successfully.")
            except Exception as e_cleanup:
                logger.error(f"Error closing AsyncWebCrawler during shutdown: {e_cleanup}", exc_info=True)
        logger.info("Resource cleanup complete. Shutdown finished (FastMCP lifespan).")

# --- MCP Server Instance and FastAPI App Setup ---

# 1. Initialize FastMCP server instance, now with its own lifespan manager
# This 'mcp' object is what tools will be decorated on (e.g., in src/tools/)
mcp = FastMCP(
    name=settings.MCP_SERVER_NAME, # Use 'name' as per FastMCP 2.x examples
    instructions=getattr(settings, 'MCP_SERVER_DESCRIPTION', "MCP server for RAG and web crawling with Crawl4AI"), # Use 'instructions'
    lifespan=mcp_resource_lifespan # Pass the lifespan manager here
    # timeout is not a constructor argument here; managed by transport or FastAPI app
    # host and port are for standalone FastMCP.run(), not needed when mounting into FastAPI
)

# Old lifecycle hooks removed, as lifespan manager handles this now.

# 3. Create the MCP-specific ASGI app (using SSE transport as implied by original error context)
# The path here is internal to the mcp_asgi_app
mcp_internal_base_path = getattr(settings, 'MCP_INTERNAL_BASE_PATH', '/mcp')
mcp_asgi_app = mcp.sse_app(path=mcp_internal_base_path)

# 4. Create the main FastAPI application
# Its lifespan is now managed by the mcp_asgi_app's lifespan
fastapi_app = FastAPI(lifespan=mcp_asgi_app.lifespan)

# 5. Mount the MCP ASGI app into the main FastAPI application
# This is the public-facing path for the MCP server
mcp_mount_path = getattr(settings, 'MCP_MOUNT_PATH', '/mcp-server')
fastapi_app.mount(mcp_mount_path, mcp_asgi_app)

# --- Resource Accessor Functions (using module-level 'resources') ---

def get_qdrant_client() -> AsyncQdrantClient:
    """Retrieves the Qdrant client from module-level resources."""
    if resources.qdrant_client is None:
        raise RuntimeError("Qdrant client not initialized.")
    return resources.qdrant_client

def get_web_crawler() -> AsyncWebCrawler:
    """Retrieves the AsyncWebCrawler from module-level resources."""
    if resources.web_crawler is None:
        raise RuntimeError("AsyncWebCrawler not initialized.")
    return resources.web_crawler

def get_log_accessor() -> LogAccessor:
    """Retrieves the LogAccessor from module-level resources."""
    if resources.log_accessor is None:
        raise RuntimeError("LogAccessor not initialized.")
    return resources.log_accessor

def get_reranker() -> Optional[CrossEncoder]:
    """Retrieves the CrossEncoder reranker from module-level resources. Returns None if not available."""
    return resources.reranker
