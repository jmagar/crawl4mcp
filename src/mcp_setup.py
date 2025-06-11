"""
Centralized setup for the FastMCP server instance, context, and lifespan manager.
"""
import os
import asyncio # Required for asyncio.to_thread in main app, but good to have for context manager if needed
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
import logging # Import logging module

from fastmcp import FastMCP # Changed from mcp.server.fastmcp
from qdrant_client import QdrantClient
from crawl4ai import AsyncWebCrawler, BrowserConfig

# Assuming these utilities are needed for lifespan setup
from .utils.qdrant.setup import get_qdrant_client, ensure_qdrant_collection_async
# Import logging utilities
from .utils.logging_utils import setup_logging, get_logger, LogAccessor # Import LogAccessor
from .config import settings # Import centralized settings
from sentence_transformers import CrossEncoder # For reranking
from typing import Optional # For Optional type hint for reranking_model

# Load environment variables from .env file (if it exists)
dotenv_path = Path(".env")
if dotenv_path.exists():
    load_dotenv(dotenv_path)
else:
    # If not in project root, try looking for .env in parent directory
    parent_dotenv_path = Path("../.env")
    if parent_dotenv_path.exists():
        load_dotenv(parent_dotenv_path)

# Setup logging early
load_dotenv()  # Ensure environment vars like LOG_LEVEL are available
setup_logging()  # Initialize logging configuration
logger = get_logger(__name__)  # Get logger for this module
app_logger = get_logger("crawl4mcp_server") # Specific application logger



# Create a dataclass for our application context
@dataclass
class LifespanContext:
    """
    Shared context available during the lifespan of the application.
    Stores critical components that should be shared across tool calls.
    """
    qdrant_client: QdrantClient
    collection_name: str
    crawler: AsyncWebCrawler
    app_logger: logging.Logger # Add the application logger
    log_accessor: LogAccessor   # Add the LogAccessor instance
    reranking_model: Optional[CrossEncoder] = None # For optional reranking

# Global lock and flags to ensure single initialization
_initialization_lock = asyncio.Lock()
_initialized = False
_lifespan_context_instance: Optional[LifespanContext] = None # To store the created context

@asynccontextmanager
async def lifespan_context_manager(app) -> AsyncIterator[LifespanContext]:
    global _initialized, _lifespan_context_instance

    async with _initialization_lock:
        if not _initialized:
            logger.info("Starting core initialization of FastMCP application and resources...")
            # Initialize to None before try block for broader scope in cleanup
            qdrant_client_instance_local = None
            crawler_instance_local = None
            log_accessor_instance_local = None
            reranking_model_instance_local = None
            try:
                # Initialize Qdrant Client using settings
                logger.info(f"Attempting to connect to Qdrant at {settings.QDRANT_URL}")
                qdrant_client_instance_local = get_qdrant_client(qdrant_url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
                logger.info("Successfully connected to Qdrant.")
                logger.info(f"Ensuring Qdrant collection '{settings.QDRANT_COLLECTION}' with vector dimension {settings.QDRANT_VECTOR_DIM} exists.")
                await ensure_qdrant_collection_async(qdrant_client_instance_local, settings.QDRANT_COLLECTION, settings.QDRANT_VECTOR_DIM)
                logger.info(f"Qdrant collection '{settings.QDRANT_COLLECTION}' is ready.")

                # Initialize AsyncWebCrawler
                logger.info("Initializing AsyncWebCrawler.")
                # Initialize without explicit browser_config to avoid parameter duplication
                crawler_instance_local = AsyncWebCrawler()
                logger.info("AsyncWebCrawler initialized.")

                # Initialize LogAccessor using settings
                logger.info("Initializing LogAccessor.")
                log_accessor_instance_local = LogAccessor(default_log_filename=settings.LOG_FILENAME)
                logger.info("LogAccessor initialized.")

                # Conditionally initialize CrossEncoder for reranking
                if settings.USE_RERANKING:
                    logger.info(f"Reranking is enabled. Loading cross-encoder model: {settings.CROSS_ENCODER_MODEL_NAME}")
                    try:
                        reranking_model_instance_local = CrossEncoder(settings.CROSS_ENCODER_MODEL_NAME)
                        logger.info(f"Successfully loaded cross-encoder model: {settings.CROSS_ENCODER_MODEL_NAME}")
                    except Exception as e:
                        logger.error(f"Failed to load cross-encoder model '{settings.CROSS_ENCODER_MODEL_NAME}': {e}. Reranking will be disabled.", exc_info=True)
                        reranking_model_instance_local = None # Ensure it's None if loading failed
                else:
                    logger.info("Reranking is disabled via settings.")

                _lifespan_context_instance = LifespanContext(
                    qdrant_client=qdrant_client_instance_local,
                    collection_name=settings.QDRANT_COLLECTION,
                    crawler=crawler_instance_local,
                    app_logger=app_logger, # Pass the app_logger instance
                    log_accessor=log_accessor_instance_local, # Pass the log_accessor instance
                    reranking_model=reranking_model_instance_local # Pass the reranker instance (can be None)
                )
                _initialized = True
                logger.info("Core initialization complete. Context created and stored.")
            except Exception as e:
                logger.error(f"Error during FastMCP application core startup: {e}", exc_info=True)
                if crawler_instance_local: # Use local var for cleanup
                    await crawler_instance_local.close()
                # Add other partial cleanup if necessary for qdrant_client_local, etc.
                _initialized = False # Ensure it's marked as not initialized on error
                _lifespan_context_instance = None # Ensure context is None on error
                raise # Re-raise the exception to signal startup failure
        else:
            logger.info("Core initialization already performed. Re-using existing context.")

    if not _lifespan_context_instance:
        # This should only be reached if the first initialization attempt failed and raised an exception,
        # and somehow the server is still trying to proceed (which Uvicorn might prevent).
        # Or if the lock logic has an issue.
        logger.critical("Lifespan context is None after initialization phase. Server cannot operate correctly.")
        raise RuntimeError("Critical error: Lifespan context is not available.")

    yield _lifespan_context_instance

    # Cleanup phase (runs when the server shuts down after the yield)
    # This code executes when the 'with' block using this context manager exits.
    # Ensure this block is correctly indented at the same level as the 'async with _initialization_lock:' above.
    logger.info("Shutting down FastMCP application and cleaning up resources.")
    
    if _lifespan_context_instance: # Check if context was ever successfully created
        if _lifespan_context_instance.crawler:
            try:
                await _lifespan_context_instance.crawler.close()
                logger.info("AsyncWebCrawler closed successfully.")
            except Exception as e_cleanup: # Use a different variable name for clarity
                logger.error(f"Error closing AsyncWebCrawler during shutdown: {e_cleanup}", exc_info=True)
        
        # Placeholder for Qdrant client cleanup if it had an explicit close method and was part of the context.
        # For example, if qdrant_client had a .close() method:
        # if hasattr(_lifespan_context_instance, 'qdrant_client') and _lifespan_context_instance.qdrant_client and \
        #    hasattr(_lifespan_context_instance.qdrant_client, 'close') and callable(getattr(_lifespan_context_instance.qdrant_client, 'close')):
        #     try:
        #         # await _lifespan_context_instance.qdrant_client.close() # if async
        #         # _lifespan_context_instance.qdrant_client.close() # if sync
        #         logger.info("Qdrant client resources released.")
        #     except Exception as e_cleanup:
        #         logger.error(f"Error closing Qdrant client during shutdown: {e_cleanup}", exc_info=True)

        logger.info("Other resources (Reranker, LogAccessor) checked for cleanup (if applicable).")

        # Optional: Reset global state if the server process itself isn't fully restarting.
        # This is generally not needed if each container run is fresh and the Python process exits.
        # async with _initialization_lock: # Ensure thread-safety if resetting globals
        #     _lifespan_context_instance = None
        #     _initialized = False
        #     logger.info("Global initialization state reset for potential future use.")
            
    logger.info("FastMCP application shutdown complete.")

# Initialize FastMCP server instance
# This `mcp` instance will be imported by tool modules to register their tools.
mcp = FastMCP(
    settings.MCP_SERVER_NAME,
    description="MCP server for RAG and web crawling with Crawl4AI",
    lifespan=lifespan_context_manager,
    host=settings.MCP_HOST,
    port=settings.MCP_PORT,
    timeout=settings.MCP_TIMEOUT
)

logger.debug(f"MCP server configuration - host: {settings.MCP_HOST}, port: {settings.MCP_PORT}, name: {settings.MCP_SERVER_NAME}")

logger.info(f"FastMCP instance created: {settings.MCP_SERVER_NAME} on {settings.MCP_HOST}:{settings.MCP_PORT}") 