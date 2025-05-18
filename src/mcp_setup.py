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

from fastmcp import FastMCP # Changed from mcp.server.fastmcp
from qdrant_client import QdrantClient
from crawl4ai import AsyncWebCrawler, BrowserConfig

# Assuming these utilities are needed for lifespan setup
from .utils.qdrant_utils import get_qdrant_client #, ensure_qdrant_collection_async # ensure_qdrant_collection_async is defined below

# Load environment variables from the project root .env file
# This should run when this module is imported to set up env vars early
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path, override=True)
    print(f"Loaded .env from {dotenv_path}")
else:
    print(f".env file not found at {dotenv_path}, using environment defaults.")


# Create a dataclass for our application context
@dataclass
class Crawl4AIContext:
    """Context for the Crawl4AI MCP server."""
    crawler: AsyncWebCrawler
    qdrant_client: QdrantClient
    collection_name: str

@asynccontextmanager
async def crawl4ai_lifespan(server: FastMCP) -> AsyncIterator[Crawl4AIContext]:
    """
    Manages the Crawl4AI client lifecycle.
    
    Args:
        server: The FastMCP server instance (type hint only, not used in this version of lifespan)
        
    Yields:
        Crawl4AIContext: The context containing the Crawl4AI crawler and Qdrant client
    """
    # Create browser configuration
    browser_config = BrowserConfig(
        headless=True,
        verbose=os.getenv("CRAWLER_VERBOSE", "False").lower() == "true" # Configurable verbosity
    )
    
    # Initialize the crawler
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()
    
    # Initialize Qdrant client
    qdrant_client = get_qdrant_client() # From qdrant_utils
    collection_name = os.getenv("QDRANT_COLLECTION", "crawled_pages")
    vector_dim = int(os.getenv("VECTOR_DIM", "1024"))
    
    # Ensure collection exists
    # await ensure_qdrant_collection_async(qdrant_client, collection_name, vector_dim) # From qdrant_utils
    try:
        # client.get_collection is synchronous
        await asyncio.to_thread(qdrant_client.get_collection, collection_name=collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        # Assuming the error means the collection does not exist.
        # A more robust check for specific "Not Found" error types might be needed depending on qdrant_client version.
        print(f"Collection '{collection_name}' not found or error checking: {e}. Attempting to create it.")
        try:
            # client.create_collection is synchronous
            await asyncio.to_thread(
                qdrant_client.create_collection,
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_dim, distance=models.Distance.COSINE) # Ensure models is imported if not already
            )
            print(f"Collection '{collection_name}' created with vector_dim={vector_dim}.")
        except Exception as create_e:
            print(f"Failed to create collection '{collection_name}': {create_e}")
            raise # Re-raise the creation error if it fails

    try:
        print("Crawl4AI Lifespan started, crawler and Qdrant client initialized.")
        yield Crawl4AIContext(
            crawler=crawler,
            qdrant_client=qdrant_client,
            collection_name=collection_name
        )
    finally:
        # Clean up the crawler
        await crawler.__aexit__(None, None, None)
        print("Crawl4AI Lifespan ended, crawler resources released.")

# Initialize FastMCP server instance
# This `mcp` instance will be imported by tool modules to register their tools.
mcp = FastMCP(
    os.getenv("MCP_SERVER_NAME", "crawl4mcp"), # Server name from env or default
    description="MCP server for RAG and web crawling with Crawl4AI", # Hardcoded description
    lifespan=crawl4ai_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=int(os.getenv("PORT", "8051")), # Ensure port is int
    timeout=int(os.getenv("MCP_TIMEOUT", "1200")) # Ensure timeout is int
)

# Need to import models for VectorParams
from qdrant_client.http import models

server_name_for_print = os.getenv("MCP_SERVER_NAME", "crawl4mcp")
host_for_print = os.getenv("HOST", "0.0.0.0")
port_for_print = os.getenv("PORT", "8051") # For printing, string is fine
print(f"FastMCP instance created in mcp_setup.py: {server_name_for_print} on {host_for_print}:{port_for_print}") 