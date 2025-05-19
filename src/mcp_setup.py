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

# Default config values (can be overridden in .env)
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9130
DEFAULT_PATH_PREFIX = "/mcp"

mcp_host = os.getenv("HOST", DEFAULT_HOST)
mcp_port = int(os.getenv("PORT", DEFAULT_PORT))
mcp_path = os.getenv("PATH", DEFAULT_PATH_PREFIX)

logger.debug(f"MCP server configuration - host: {mcp_host}, port: {mcp_port}, path prefix: {mcp_path}")

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

@asynccontextmanager
async def lifespan_context_manager(app) -> AsyncIterator[LifespanContext]:
    """
    Context manager for FastMCP lifespan to create and clean up resources.
    
    Args:
        app: The FastMCP server instance
    """
    logger.info("Initializing FastMCP application and resources")
    qdrant_client_instance = None
    crawler_instance = None
    log_accessor_instance = None # Initialize to None

    try:
        # Initialize Qdrant Client - get_qdrant_client() uses env vars internally
        # and doesn't take explicit parameters
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        logger.info(f"Attempting to connect to Qdrant at {qdrant_url}")
        qdrant_client_instance = get_qdrant_client()
        logger.info("Successfully connected to Qdrant.")

        # Ensure Qdrant Collection Exists
        collection_name = os.getenv("QDRANT_COLLECTION", "crawl4ai_mcp")
        vector_dim = int(os.getenv("VECTOR_DIM", "1024"))
        logger.info(f"Ensuring Qdrant collection '{collection_name}' with vector dimension {vector_dim} exists.")
        await ensure_qdrant_collection_async(qdrant_client_instance, collection_name, vector_dim)
        logger.info(f"Qdrant collection '{collection_name}' is ready.")

        # Initialize AsyncWebCrawler
        logger.info("Initializing AsyncWebCrawler.")
        # Initialize without explicit browser_config to avoid parameter duplication
        crawler_instance = AsyncWebCrawler()
        logger.info("AsyncWebCrawler initialized.")

        # Initialize LogAccessor
        # The default_log_filename for LogAccessor will be derived from LOG_FILENAME env var or its internal default.
        logger.info("Initializing LogAccessor.")
        log_accessor_instance = LogAccessor(default_log_filename=os.getenv("LOG_FILENAME", "crawl4mcp.log"))
        logger.info("LogAccessor initialized.")

        # Yield the context object with all initialized resources
        # Make sure app_logger (the general server logger) is also passed
        yield LifespanContext(
            qdrant_client=qdrant_client_instance,
            collection_name=collection_name,
            crawler=crawler_instance,
            app_logger=app_logger, # Pass the app_logger instance
            log_accessor=log_accessor_instance # Pass the log_accessor instance
        )
    
    except Exception as e:
        logger.error(f"Error during FastMCP application startup: {e}", exc_info=True)
        # Re-raise the exception to ensure Uvicorn/FastAPI handles it correctly and stops startup
        raise
    finally:
        logger.info("Shutting down FastMCP application and cleaning up resources.")
        if qdrant_client_instance:
            try:
                # QdrantClient doesn't have an explicit async close in the typical sense,
                # but if it did, it would be called here.
                # For now, ensure connections are closed if it manages a pool internally.
                # client.close() is synchronous, if it exists and is needed.
                if hasattr(qdrant_client_instance, 'close'):
                    qdrant_client_instance.close() # type: ignore
                logger.info("Qdrant client resources (if any) released.")
            except Exception as e:
                logger.error(f"Error closing Qdrant client: {e}", exc_info=True)
        
        if crawler_instance:
            try:
                await crawler_instance.close()
                logger.info("AsyncWebCrawler closed successfully.")
            except Exception as e:
                logger.error(f"Error closing AsyncWebCrawler: {e}", exc_info=True)
        logger.info("FastMCP application shutdown complete.")

# Initialize FastMCP server instance
# This `mcp` instance will be imported by tool modules to register their tools.
mcp = FastMCP(
    os.getenv("MCP_SERVER_NAME", "crawl4mcp"), # Server name from env or default
    description="MCP server for RAG and web crawling with Crawl4AI", # Hardcoded description
    lifespan=lifespan_context_manager,
    host=mcp_host,
    port=mcp_port,
    timeout=int(os.getenv("MCP_TIMEOUT", "1200")) # Ensure timeout is int
)

server_name_for_print = os.getenv("MCP_SERVER_NAME", "crawl4mcp")
host_for_print = mcp_host
port_for_print = mcp_port # For printing, string is fine
logger.info(f"FastMCP instance created: {server_name_for_print} on {host_for_print}:{port_for_print}") 