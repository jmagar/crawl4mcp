"""
Main application file for the Crawl4AI MCP server.

This file initializes the MCP server and registers all tools
by importing the necessary modules.
"""
import asyncio
import os

# Import logging utilities
from .utils.logging_utils import get_logger

# Initialize logger
logger = get_logger(__name__)

# Import the mcp instance from mcp_setup.py
# This mcp instance already has lifespan and basic configurations.
# The mcp object IS the ASGI application.
from .mcp_setup import mcp

# Import tool modules to register their MCP tools.
# The @mcp.tool() decorator in each tool module will automatically register them with the `mcp` instance.
from .tools import crawling_tools
from .tools import retrieval_tools
from .tools import management_tools
from .tools import analytics_tools

# Log that all modules have been imported and registered
logger.info("MCP server fully initialized with all tools registered")
logger.debug(f"Registered tools: crawling_tools, retrieval_tools, management_tools, analytics_tools")

# Export the mcp FastMCP instance as "app" for ASGI servers like uvicorn
app = mcp.http_app()

# If running directly, start the server (for development only)
if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 9130))
    
    logger.info(f"Starting development server on {host}:{port}")
    uvicorn.run(
        "src.app:app",
        host=host,
        port=port,
        reload=True,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    ) 