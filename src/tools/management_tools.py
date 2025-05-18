"""
MCP Tools for Qdrant collection management and information retrieval.
"""
import json
from typing import Optional, List
import os
from pathlib import Path

from mcp.server.fastmcp import Context # MCP Context for tool arguments
from mcp.server.fastmcp.exceptions import ResourceError # Correct import path

# Import the centralized mcp instance
from ..mcp_setup import mcp
# Import utility functions from qdrant_utils
from ..utils.qdrant_utils import (
    get_available_sources as get_available_sources_util, # Alias
    get_collection_stats as get_collection_stats_util,   # Alias
    get_qdrant_client # Added import
)
# Import logging utilities
from ..utils.logging_utils import get_logger

# Initialize logger
logger = get_logger(__name__)

@mcp.resource()
async def get_available_sources(ctx: Context) -> List[str]:
    """
    Get all available sources based on unique source metadata values.
    Accessible via resource URI: resource://get_available_sources
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
        logger.warning(f"Context access failed for get_available_sources ({type(e).__name__}: {e}). Initializing Qdrant from environment.")
        try:
            qdrant_client_instance = get_qdrant_client()
            collection_name_str = os.getenv("QDRANT_COLLECTION")
            if not collection_name_str:
                raise ResourceError(message="QDRANT_COLLECTION environment variable must be set when context is not available.", code="CONFIG_ERROR")
        except Exception as e_init:
            logger.error(f"Failed to initialize Qdrant: {e_init}")
            raise ResourceError(message=f"Failed to initialize Qdrant: {str(e_init)}", code="INITIALIZATION_ERROR", details={"original_exception": str(e_init)})

    if not all([qdrant_client_instance, collection_name_str]):
        logger.error("Qdrant client or collection name missing for get_available_sources.")
        raise ResourceError(message="Qdrant client or collection name missing for get_available_sources.", code="MISSING_DEPENDENCY")

    try:
        logger.debug(f"Fetching available sources from collection '{collection_name_str}'")
        sources = await get_available_sources_util(
            client=qdrant_client_instance, 
            collection_name=collection_name_str
        )
        logger.info(f"Found {len(sources)} sources")
        return sources
    except Exception as e:
        logger.error(f"Error in get_available_sources: {e}")
        raise ResourceError(message=f"Error getting available sources: {str(e)}", code="QDRANT_ERROR", details={"original_exception": str(e)})

@mcp.resource()
async def get_collection_stats(
    ctx: Context,
    collection_name: Optional[str] = None,
    include_segments: bool = False
) -> dict:
    """
    Get statistics about a Qdrant collection or all collections.
    Accessible via resource URI: resource://get_collection_stats (for default)
    or resource://get_collection_stats/{collection_name}?include_segments={true|false}
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
        logger.warning(f"Context access failed for get_collection_stats ({type(e).__name__}: {e}). Initializing Qdrant from environment.")
        try:
            qdrant_client_instance = get_qdrant_client() # From qdrant_utils
            default_collection_name_from_context_or_env = os.getenv("QDRANT_COLLECTION") # Changed from QDRANT_COLLECTION_NAME
        except Exception as e_client:
            logger.error(f"Failed to initialize Qdrant client: {e_client}")
            raise ResourceError(message=f"Failed to initialize Qdrant client: {str(e_client)}", code="INITIALIZATION_ERROR", details={"original_exception": str(e_client)})

    if not qdrant_client_instance:
        # This case should ideally be covered by the try-except block above
        logger.error("Qdrant client could not be initialized.")
        raise ResourceError(message="Qdrant client could not be initialized.", code="INITIALIZATION_ERROR")

    # Determine the target collection name for the query
    # If collection_name parameter is provided to the tool, it takes precedence.
    # Otherwise, use the default derived from context or environment.
    # If that's also None, get_collection_stats_util will fetch for all collections.
    target_collection_name_for_query = collection_name if collection_name is not None else default_collection_name_from_context_or_env
    
    try:
        logger.debug(f"Fetching collection stats for '{target_collection_name_for_query if target_collection_name_for_query else 'all collections'}'")
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
            logger.error(f"Error from get_collection_stats_util: {error_message}")
            raise ResourceError(message=f"Error from get_collection_stats_util: {error_message}", code="STATS_ERROR", details={"utility_error": error_message})

        logger.info(f"Successfully retrieved collection stats")
        return stats
    except Exception as e:
        logger.error(f"Error in get_collection_stats tool: {e}")
        raise ResourceError(message=f"Error in get_collection_stats: {str(e)}", code="PROCESSING_ERROR", details={"original_exception": str(e)})

# Ensure the file ends with a newline for linters 