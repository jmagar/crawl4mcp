"""
Main application file for the Crawl4AI MCP server.

This file initializes the MCP server and registers all tools
by importing the necessary modules.
"""
import asyncio
import os
# import uvicorn # No longer explicitly needed here if FastMCP handles its own Uvicorn via run_async

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

# Potentially other utilities or configurations that are truly global for the app entry point.
# For now, most setup is in mcp_setup.py

async def main():
    """Configures and runs the MCP server using FastMCP's run_async method."""
    server_name = os.getenv("MCP_SERVER_NAME", "crawl4mcp")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8051")) # Ensure port is int for run_async
    
    print(f"Starting MCP server: {server_name} on {host}:{port}...")
    # The line for printing registered tools is still commented out.
    # We can investigate the correct way to list FastMCP tools later if needed.
    # print(f"Registered tools: {list(mcp.tools.keys())}")

    # Use mcp.run_async() as per FastMCP documentation for async contexts
    # Specify transport, host, and port. FastMCP's run_async will handle Uvicorn/ASGI server setup.
    await mcp.run_async(transport="streamable-http")
    # log_level can also be set here if supported by run_async, e.g., log_level="info"

if __name__ == "__main__":
    # Ensure any necessary environment setup (like loading .env) happens before this.
    # This is handled in mcp_setup.py which is imported before main() is called.
    asyncio.run(main()) 