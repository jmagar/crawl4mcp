"""
MCP Tools for Qdrant collection management and information retrieval.
"""
import json
from typing import Optional
import os

from mcp.server.fastmcp import Context # MCP Context for tool arguments

# Import the centralized mcp instance
from ..mcp_setup import mcp
# Import utility functions from qdrant_utils
from ..utils.qdrant_utils import (
    get_available_sources as get_available_sources_util, # Alias
    get_collection_stats as get_collection_stats_util,   # Alias
    get_qdrant_client # Added import
)

@mcp.tool()
async def get_available_sources(ctx: Context) -> str:
    """
    Get all available sources based on unique source metadata values.
    """
    qdrant_client_instance = None
    collection_name_str = None

    try:
        if hasattr(ctx, 'request_context') and ctx.request_context is not None:
            qdrant_client_instance = ctx.request_context.lifespan_context.qdrant_client
            collection_name_str = ctx.request_context.lifespan_context.collection_name
        else:
            raise AttributeError("request_context not available for get_available_sources")
    except (AttributeError, ValueError) as e:
        print(f"Context access failed for get_available_sources ({type(e).__name__}: {e}). Initializing Qdrant from environment.")
        try:
            qdrant_client_instance = get_qdrant_client()
            collection_name_str = os.getenv("QDRANT_COLLECTION")
            if not collection_name_str:
                raise ValueError("QDRANT_COLLECTION environment variable must be set when context is not available.")
        except Exception as e_init:
            return json.dumps({"success": False, "error": f"Failed to initialize Qdrant: {str(e_init)}"}, indent=2)

    if not all([qdrant_client_instance, collection_name_str]):
        return json.dumps({"success": False, "error": "Qdrant client or collection name missing for get_available_sources."}, indent=2)

    try:
        sources = await get_available_sources_util(
            client=qdrant_client_instance, 
            collection_name=collection_name_str
        )
        return json.dumps({"success": True, "sources": sources, "count": len(sources)}, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)

@mcp.tool()
async def get_collection_stats(
    ctx: Context,
    collection_name: Optional[str] = None,
    include_segments: bool = False
) -> str:
    """
    Get statistics about a Qdrant collection or all collections.
    """
    qdrant_client_instance = None
    default_collection_name_from_context_or_env = None

    try:
        # Try to get client and default collection name from context
        if hasattr(ctx, 'request_context') and ctx.request_context is not None:
            qdrant_client_instance = ctx.request_context.lifespan_context.qdrant_client
            default_collection_name_from_context_or_env = ctx.request_context.lifespan_context.collection_name
        else:
            raise AttributeError("request_context not available for get_collection_stats") # Force fallback
    except (AttributeError, ValueError) as e: # Catch ValueError here too
        # Context is not available or not structured as expected (e.g., direct call with ctx={})
        print(f"Context access failed for get_collection_stats ({type(e).__name__}: {e}). Initializing Qdrant from environment.")
        try:
            qdrant_client_instance = get_qdrant_client() # From qdrant_utils
            default_collection_name_from_context_or_env = os.getenv("QDRANT_COLLECTION") # Changed from QDRANT_COLLECTION_NAME
        except Exception as e_client:
            return json.dumps({"success": False, "error": f"Failed to initialize Qdrant client: {str(e_client)}"}, indent=2)

    if not qdrant_client_instance:
        # This case should ideally be covered by the try-except block above
        return json.dumps({"success": False, "error": "Qdrant client could not be initialized."}, indent=2)

    # Determine the target collection name for the query
    # If collection_name parameter is provided to the tool, it takes precedence.
    # Otherwise, use the default derived from context or environment.
    # If that's also None, get_collection_stats_util will fetch for all collections.
    target_collection_name_for_query = collection_name if collection_name is not None else default_collection_name_from_context_or_env
    
    try:
        stats = await get_collection_stats_util(
            qdrant_client=qdrant_client_instance, 
            collection_name=target_collection_name_for_query,
            include_segments=include_segments
        )
        
        # Post-processing to add default_server_collection_name and is_default_server_collection flags
        if isinstance(stats, dict) and stats.get("success", False):
            stats["default_server_collection_name"] = default_collection_name_from_context_or_env # This can be None
            
            if "collection" in stats and isinstance(stats["collection"], dict) and stats["collection"].get("name"):
                # Handling for single collection result
                coll_name_in_stats = stats["collection"]["name"]
                is_default = (default_collection_name_from_context_or_env is not None) and \
                             (coll_name_in_stats == default_collection_name_from_context_or_env)
                stats["collection"]["is_default_server_collection"] = is_default
            elif "collections" in stats and isinstance(stats.get("collections"), list):
                # Handling for multiple collections result
                for coll_stat in stats["collections"]:
                    if isinstance(coll_stat, dict) and coll_stat.get("name"):
                        is_default = (default_collection_name_from_context_or_env is not None) and \
                                     (coll_stat.get("name") == default_collection_name_from_context_or_env)
                        coll_stat["is_default_server_collection"] = is_default
            
        elif not (isinstance(stats, dict) and stats.get("success", False)):
            error_message = stats.get("error") if isinstance(stats, dict) else "Failed to get valid stats object from utility."
            return json.dumps({"success": False, "error": error_message}, indent=2)

        return json.dumps(stats, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "error": f"Error in get_collection_stats tool: {str(e)}"}, indent=2)

# Ensure the file ends with a newline for linters 