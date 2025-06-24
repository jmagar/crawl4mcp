"""
Main application file for the Crawl4AI MCP server.

This file initializes the MCP server and registers all tools
by importing the necessary modules.
"""
import os

# Import logging utilities
from src.utils.logging_utils import get_logger

# Initialize logger
logger = get_logger(__name__)

# Import the mcp instance from mcp_setup.py
# This mcp instance already has lifespan and basic configurations.
# The mcp object IS the ASGI application.
from src.mcp_setup import mcp

# Import tool modules to register their MCP tools.
# The @mcp.tool() decorator in each tool module will automatically register them with the `mcp` instance.
from src.tools import crawling_tools
from src.tools import retrieval_tools
from src.tools import management_tools
from src.tools import analytics_tools
from src.tools import search_tools

# Log that all modules have been imported and registered
logger.info("MCP server fully initialized with all tool modules imported.")

# Enhanced logging for registered tools
try:
    tool_names_str_list = [] # This will store the final list of tool name strings

    if hasattr(mcp, 'tool_manager') and mcp.tool_manager is not None:
        # Try mcp.tool_manager.get_tools()
        if hasattr(mcp.tool_manager, 'get_tools') and callable(mcp.tool_manager.get_tools):
            registered_items = mcp.tool_manager.get_tools()
            if isinstance(registered_items, dict):
                for tool_name, tool_obj in registered_items.items():
                    if hasattr(tool_obj, 'name') and isinstance(tool_obj.name, str):
                        tool_names_str_list.append(tool_obj.name)
                    elif isinstance(tool_name, str): # Fallback if tool_obj.name isn't there but key is string
                        tool_names_str_list.append(tool_name)
            elif isinstance(registered_items, list):
                for item in registered_items:
                    if hasattr(item, 'name') and isinstance(item.name, str):
                        tool_names_str_list.append(item.name)
                    elif isinstance(item, str):
                        tool_names_str_list.append(item)
            else:
                logger.debug("mcp.tool_manager.get_tools() returned an unexpected type or empty.")
        
        # If get_tools() didn't yield names, try mcp.tool_manager.list_tools()
        if not tool_names_str_list and hasattr(mcp.tool_manager, 'list_tools') and callable(mcp.tool_manager.list_tools):
            logger.debug("Attempting to use mcp.tool_manager.list_tools().")
            tool_objects_or_names = mcp.tool_manager.list_tools()
            for item in tool_objects_or_names:
                if hasattr(item, 'name') and isinstance(item.name, str):
                    tool_names_str_list.append(item.name)
                elif isinstance(item, str):
                    tool_names_str_list.append(item)
        
        # If still no names, and _tool_manager exists, try mcp._tool_manager.list_tools()
        if not tool_names_str_list and hasattr(mcp, '_tool_manager') and hasattr(mcp._tool_manager, 'list_tools') and callable(mcp._tool_manager.list_tools):
            logger.warning("Accessing mcp._tool_manager.list_tools() as public methods on mcp.tool_manager did not yield tool names.")
            tool_objects_or_names = mcp._tool_manager.list_tools()
            for item in tool_objects_or_names:
                if hasattr(item, 'name') and isinstance(item.name, str):
                    tool_names_str_list.append(item.name)
                elif isinstance(item, str):
                    tool_names_str_list.append(item)

    # Direct fallback if mcp.tool_manager attribute itself wasn't found, but _tool_manager exists
    elif hasattr(mcp, '_tool_manager') and hasattr(mcp._tool_manager, 'list_tools') and callable(mcp._tool_manager.list_tools):
        logger.warning("Accessing mcp._tool_manager directly to list tools as mcp.tool_manager was not found.")
        tool_objects_or_names = mcp._tool_manager.list_tools()
        for item in tool_objects_or_names:
            if hasattr(item, 'name') and isinstance(item.name, str):
                tool_names_str_list.append(item.name)
            elif isinstance(item, str):
                tool_names_str_list.append(item)
    else:
        logger.error("MCP tool_manager or _tool_manager not found or accessible using known methods. Cannot list registered tools.")

    if tool_names_str_list:
        # Remove duplicates and sort before logging
        unique_sorted_tool_names = sorted(list(set(tool_names_str_list)))
        logger.info(f"Successfully registered MCP tools: {', '.join(unique_sorted_tool_names)}")
    else:
        logger.warning("No MCP tool names successfully extracted.")
            
except Exception as e:
    logger.error(f"Error retrieving registered tools: {e}", exc_info=True)

# Export the mcp FastMCP instance as "app" for ASGI servers like uvicorn
# NOTE: For SSE transport, we'll use mcp.run() directly instead of http_app()
app = mcp.http_app()  # Keep this for compatibility, but SSE will use run() method

# If running directly, start the server (for development only)
if __name__ == "__main__":
    # Get configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 9130))
    
    # Check if SSE transport is requested via environment variable
    transport_mode = os.getenv("FASTMCP_TRANSPORT", "http").lower()
    
    if transport_mode == "sse":
        logger.info(f"Starting FastMCP server with SSE transport on {host}:{port}")
        # Use FastMCP's built-in SSE transport
        mcp.run(
            transport="sse",
            host=host,
            port=port,
            log_level=os.getenv("LOG_LEVEL", "info").lower()
        )
    else:
        # Default to uvicorn with http_app for regular HTTP
        import uvicorn
        logger.info(f"Starting development server with HTTP transport on {host}:{port}")
        uvicorn.run(
            "src.app:app",
            host=host,
            port=port,
            reload=True,
            log_level=os.getenv("LOG_LEVEL", "info").lower()
        ) 