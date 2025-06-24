"""
MCP Tools for Qdrant collection management and information retrieval.
"""
import json
from typing import Optional, List
import os
from pathlib import Path
import asyncio

from mcp.server.fastmcp.exceptions import ToolError

# Import the centralized mcp instance
from src.mcp_setup import mcp
# Import utility functions
from src.utils.qdrant.setup import get_qdrant_client, create_rrf_collection, migrate_to_rrf_collection
from src.utils.qdrant.admin import (
    get_available_sources as get_available_sources_util, 
    get_collection_stats as get_collection_stats_util    
)
# Import logging utilities
from src.utils.logging_utils import get_logger, LogAccessor

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

@mcp.tool()
async def create_rrf_collection_tool(
    collection_name: Optional[str] = None,
    force_recreate: bool = False
) -> str:
    """
    Create a new collection with RRF support (dense + sparse vectors).
    
    Args:
        collection_name: Name for the new RRF collection
        force_recreate: Whether to recreate if collection exists
        
    Returns:
        Status message about collection creation
    """
    try:
        # Use config collection name if none provided
        if collection_name is None:
            collection_name = os.getenv("QDRANT_COLLECTION", "crawl4mcp-rrf")
        
        logger.info(f"Creating RRF collection: {collection_name}")
        
        # Get Qdrant client
        qdrant_client_instance = get_qdrant_client()
        if not qdrant_client_instance:
            return json.dumps({
                "success": False,
                "error": "Failed to connect to Qdrant"
            })
        
        # Create RRF collection
        success = await create_rrf_collection(
            client=qdrant_client_instance,
            collection_name=collection_name,
            force_recreate=force_recreate
        )
        
        if success:
            return json.dumps({
                "success": True,
                "message": f"RRF collection '{collection_name}' created successfully",
                "collection_name": collection_name,
                "features": [
                    f"Dense vectors ({os.getenv('VECTOR_DIM', '1024')}-dim for semantic search)",
                    "Sparse vectors (SPLADE for keyword search)", 
                    "Server-side RRF fusion",
                    "Optimized for search performance"
                ]
            })
        else:
            return json.dumps({
                "success": False,
                "error": f"Failed to create RRF collection '{collection_name}'"
            })
            
    except Exception as e:
        logger.error(f"Error creating RRF collection: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        })

@mcp.tool()
async def migrate_to_rrf_tool(
    source_collection: Optional[str] = None,
    target_collection: Optional[str] = None, 
    batch_size: int = 50,
    max_points: Optional[int] = None
) -> str:
    """
    Migrate data from existing collection to RRF collection with sparse vectors.
    
    Args:
        source_collection: Name of source collection (dense vectors only)
        target_collection: Name of target RRF collection  
        batch_size: Number of points to process per batch
        max_points: Maximum points to migrate (None for all)
        
    Returns:
        Migration status and progress information
    """
    try:
        # Use config collection names if none provided
        if source_collection is None:
            source_collection = "crawl4ai_mcp"  # Legacy collection name
        if target_collection is None:
            target_collection = os.getenv("QDRANT_COLLECTION", "crawl4mcp-rrf")
        
        logger.info(f"Starting migration from {source_collection} to {target_collection}")
        
        # Get Qdrant client
        qdrant_client_instance = get_qdrant_client()
        if not qdrant_client_instance:
            return json.dumps({
                "success": False,
                "error": "Failed to connect to Qdrant"
            })
        
        # Check if target collection exists
        try:
            collections = await asyncio.to_thread(qdrant_client_instance.get_collections)
            existing_collections = [col.name for col in collections.collections]
            
            if target_collection not in existing_collections:
                return json.dumps({
                    "success": False,
                    "error": f"Target collection '{target_collection}' does not exist. Create it first with create_rrf_collection_tool."
                })
                
        except Exception as e:
            return json.dumps({
                "success": False, 
                "error": f"Failed to check collections: {e}"
            })
        
        # Start migration
        success = await migrate_to_rrf_collection(
            client=qdrant_client_instance,
            source_collection=source_collection,
            target_collection=target_collection,
            batch_size=batch_size,
            max_points=max_points
        )
        
        if success:
            return json.dumps({
                "success": True,
                "message": f"Migration from '{source_collection}' to '{target_collection}' completed successfully",
                "source_collection": source_collection,
                "target_collection": target_collection,
                "batch_size": batch_size,
                "max_points": max_points,
                "next_steps": [
                    "Test the new RRF collection with perform_rag_query",
                    "Update collection_name in config to use new collection",
                    "Consider removing old collection after validation"
                ]
            })
        else:
            return json.dumps({
                "success": False,
                "error": f"Migration from '{source_collection}' to '{target_collection}' failed"
            })
            
    except Exception as e:
        logger.error(f"Error during migration: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        })

@mcp.tool()
async def test_rrf_collection(
    collection_name: Optional[str] = None,
    test_query: str = "cursor features"
) -> str:
    """
    Test RRF collection with a sample query to verify functionality.
    
    Args:
        collection_name: Name of RRF collection to test
        test_query: Query to test with
        
    Returns:
        Test results and performance comparison
    """
    try:
        from src.utils.qdrant.retrieval import query_qdrant_rrf_native, query_qdrant
        import time
        
        # Use config collection name if none provided
        if collection_name is None:
            collection_name = os.getenv("QDRANT_COLLECTION", "crawl4mcp-rrf")
        
        logger.info(f"Testing RRF collection: {collection_name}")
        
        # Get Qdrant client
        qdrant_client_instance = get_qdrant_client()
        if not qdrant_client_instance:
            return json.dumps({
                "success": False,
                "error": "Failed to connect to Qdrant"
            })
        
        # Test RRF search
        start_time = time.time()
        rrf_results = await query_qdrant_rrf_native(
            client=qdrant_client_instance,
            collection_name=collection_name,
            query_text=test_query,
            match_count=3
        )
        rrf_time = time.time() - start_time
        
        # Test regular semantic search for comparison (if old collection exists)
        semantic_results = []
        semantic_time = 0
        try:
            start_time = time.time()
            semantic_results = await query_qdrant(
                client=qdrant_client_instance,
                collection_name="crawl4ai_mcp",  # Old collection
                query_text=test_query,
                match_count=3
            )
            semantic_time = time.time() - start_time
        except:
            pass  # Old collection might not exist
        
        return json.dumps({
            "success": True,
            "test_query": test_query,
            "rrf_results": {
                "count": len(rrf_results),
                "search_time": round(rrf_time, 3),
                "search_type": "rrf_fusion",
                "results": rrf_results[:2]  # Show first 2 results
            },
            "semantic_results": {
                "count": len(semantic_results),
                "search_time": round(semantic_time, 3) if semantic_time > 0 else "N/A",
                "search_type": "semantic",
                "results": semantic_results[:2] if semantic_results else []
            },
            "performance": {
                "rrf_search_time": f"{rrf_time:.3f}s",
                "semantic_search_time": f"{semantic_time:.3f}s" if semantic_time > 0 else "N/A",
                "improvement": f"{((semantic_time - rrf_time) / semantic_time * 100):.1f}% faster" if semantic_time > 0 and semantic_time > rrf_time else "N/A"
            }
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error testing RRF collection: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        })

# Ensure the file ends with a newline for linters 