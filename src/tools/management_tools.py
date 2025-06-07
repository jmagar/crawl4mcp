"""
MCP Tools for Qdrant collection management and information retrieval.
"""
import json
from typing import Optional, List
import os
from pathlib import Path

from mcp.server.fastmcp.exceptions import ToolError

# Import the centralized mcp instance
from ..mcp_setup import mcp
# Import utility functions
from ..utils.qdrant.setup import get_qdrant_client
from ..utils.qdrant.admin import (
    get_available_sources as get_available_sources_util, 
    get_collection_stats as get_collection_stats_util    
)
# Import logging utilities
from ..utils.logging_utils import get_logger, LogAccessor

# Initialize logger
logger = get_logger(__name__)

@mcp.tool()
async def get_available_sources() -> List[str]:
    """
    Get all available sources based on unique source metadata values.
    """
    try:
        # Initialize components from environment
        qdrant_client_instance = get_qdrant_client()
        collection_name_str = os.getenv("QDRANT_COLLECTION")
        if not collection_name_str:
            raise ToolError("QDRANT_COLLECTION environment variable must be set.", "CONFIG_ERROR")

        logger.debug(f"Fetching available sources from collection '{collection_name_str}'")
        sources = await get_available_sources_util(
            client=qdrant_client_instance, 
            collection_name=collection_name_str
        )
        logger.info(f"Found {len(sources)} sources")
        return sources
    except Exception as e:
        logger.error(f"Error in get_available_sources: {e}")
        raise ToolError(f"Error getting available sources: {str(e)}", "QDRANT_ERROR", {"original_exception": str(e)})

@mcp.tool()
async def get_collection_stats(
    collection_name: Optional[str] = None,
    include_segments: bool = False
) -> dict:
    """
    Get statistics about a Qdrant collection or all collections.
    """
    try:
        # Initialize components from environment
        qdrant_client_instance = get_qdrant_client()
        default_collection_name = os.getenv("QDRANT_COLLECTION")

        # Determine the target collection name for the query
        target_collection_name_for_query = collection_name if collection_name is not None else default_collection_name
        
        logger.debug(f"Fetching collection stats for '{target_collection_name_for_query if target_collection_name_for_query else 'all collections'}'")
        stats = await get_collection_stats_util(
            qdrant_client=qdrant_client_instance, 
            collection_name=target_collection_name_for_query,
            include_segments=include_segments
        )
        
        # Post-processing to add default_server_collection_name and is_default_server_collection flags
        if isinstance(stats, dict) and stats.get("success", False):
            stats["default_server_collection_name"] = default_collection_name
            
            if "collection" in stats and isinstance(stats["collection"], dict) and stats["collection"].get("name"):
                # Handling for single collection result
                coll_name_in_stats = stats["collection"]["name"]
                is_default = (default_collection_name is not None) and \
                             (coll_name_in_stats == default_collection_name)
                stats["collection"]["is_default_server_collection"] = is_default
            elif "collections" in stats and isinstance(stats.get("collections"), list):
                # Handling for multiple collections result
                for coll_stat in stats["collections"]:
                    if isinstance(coll_stat, dict) and coll_stat.get("name"):
                        is_default = (default_collection_name is not None) and \
                                     (coll_stat.get("name") == default_collection_name)
                        coll_stat["is_default_server_collection"] = is_default
            
        elif not (isinstance(stats, dict) and stats.get("success", False)):
            error_message = stats.get("error") if isinstance(stats, dict) else "Failed to get valid stats object from utility."
            logger.error(f"Error from get_collection_stats_util: {error_message}")
            raise ToolError(f"Error from get_collection_stats_util: {error_message}", "STATS_ERROR", {"utility_error": error_message})

        logger.info(f"Successfully retrieved collection stats")
        return stats
    except Exception as e:
        logger.error(f"Error in get_collection_stats tool: {e}")
        raise ToolError(f"Error in get_collection_stats: {str(e)}", "PROCESSING_ERROR", {"original_exception": str(e)})

@mcp.tool()
async def view_server_logs(num_lines: int = 150) -> List[str]:
    """
    Retrieves the last N lines from the server's log file.
    """
    logger.info(f"Attempting to view last {num_lines} of server logs.")
    
    try:
        # Initialize LogAccessor directly from environment
        logger.debug("Initializing a new LogAccessor instance.")
        log_accessor_instance = LogAccessor()

        log_lines = await log_accessor_instance.get_last_log_lines(num_lines=num_lines)
        logger.info(f"Successfully retrieved {len(log_lines)} log lines.")
        return log_lines
    except Exception as e:
        logger.error(f"Error retrieving log lines via LogAccessor: {e}")
        raise ToolError(
            f"Error retrieving log lines: {str(e)}",
            "LOG_RETRIEVAL_ERROR",
            {"original_exception": str(e)}
        )

# Ensure the file ends with a newline for linters 