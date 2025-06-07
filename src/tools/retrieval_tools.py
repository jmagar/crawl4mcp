"""
MCP Tools for retrieving and searching content from Qdrant.
"""
import json
from typing import Optional
import os
import logging

from mcp.server.fastmcp.exceptions import ToolError # Correct import path

# Import the centralized mcp instance
from ..mcp_setup import mcp
# Import utility functions
from ..utils.qdrant.setup import get_qdrant_client
from ..utils.qdrant.retrieval import (
    query_qdrant,
    perform_hybrid_search as perform_hybrid_search_util,
    get_similar_items as get_similar_items_util,
    fetch_item_by_id as fetch_item_by_id_util,
    find_similar_content as find_similar_content_util
)
# Import logging utilities
from ..utils.logging_utils import get_logger

# Initialize logger
logger = get_logger(__name__)

@mcp.tool()
async def perform_rag_query(query: str, source: Optional[str] = None, match_count: int = 5) -> str:
    """
    Perform a RAG (Retrieval Augmented Generation) query on the stored content.
    Args:
        query: The search query.
        source: Optional source domain to filter results (e.g., 'example.com').
        match_count: Maximum number of results to return (default: 5).
    """
    logger.info(f"RAG query: '{query}' (source filter: {source}, match count: {match_count})")
    
    try:
        # Initialize components from environment
        qdrant_client_instance = get_qdrant_client()
        collection_name_str = os.getenv("QDRANT_COLLECTION")
        if not collection_name_str:
            error_msg = "QDRANT_COLLECTION environment variable must be set"
            logger.error(error_msg)
            raise ToolError(error_msg, "CONFIG_ERROR")
        
        # Execute the query
        logger.debug(f"Executing RAG query against collection: {collection_name_str}")
        query_result = await query_qdrant(
            client=qdrant_client_instance,
            collection_name=collection_name_str,
            query_text=query,
            source_filter=source,
            match_count=match_count
        )

        if not query_result:
            logger.warning(f"No results found for query: '{query}'")
            return json.dumps({
                "success": True,
                "query": query,
                "match_count": 0,
                "matches": []
            }, indent=2)

        # Format and return the result
        formatted_result = {
            "success": True,
            "query": query,
            "match_count": len(query_result),
            "matches": query_result
        }
        
        logger.info(f"RAG query complete: '{query}'. Found {len(query_result)} matches.")
        if logger.isEnabledFor(logging.DEBUG):
            # Only calculate and log top scores when in DEBUG level to avoid performance impact
            try:
                top_scores = [match["score"] for match in query_result[:3]] if query_result else []
                logger.debug(f"Top match scores: {top_scores}")
            except (KeyError, IndexError) as e:
                logger.debug(f"Could not extract top scores due to unexpected result format: {e}")
        
        return json.dumps(formatted_result, indent=2)
        
    except Exception as e:
        logger.error(f"Error in perform_rag_query: {e}", exc_info=True)
        raise ToolError(
            f"Error in RAG query: {str(e)}",
            "RAG_QUERY_ERROR",
            {"query": query, "source": source, "original_exception": str(e)}
        )

@mcp.tool()
async def perform_hybrid_search(
    query: str, 
    filter_text: Optional[str] = None, 
    vector_weight: float = 0.7, 
    keyword_weight: float = 0.3, 
    source: Optional[str] = None, 
    match_count: int = 5
) -> str:
    """
    Perform a hybrid search combining vector similarity with keyword/text-based filtering.
    """
    try:
        # Initialize components from environment
        qdrant_client_instance = get_qdrant_client()
        collection_name_str = os.getenv("QDRANT_COLLECTION")
        if not collection_name_str:
            raise ValueError("QDRANT_COLLECTION environment variable must be set.")

        if not (0.0 <= vector_weight <= 1.0 and 0.0 <= keyword_weight <= 1.0):
            raise ToolError("Hybrid search weights must be between 0.0 and 1.0", "INVALID_PARAMS", {"vector_weight": vector_weight, "keyword_weight": keyword_weight})
        
        results = await perform_hybrid_search_util(
            client=qdrant_client_instance,
            collection_name=collection_name_str,
            query_text=query,
            filter_text=filter_text,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
            source_filter=source if source and source.strip() else None,
            match_count=match_count
        )
        return json.dumps({
            "success": True, "query": query, "filter_text": filter_text,
            "weights": {"vector": vector_weight, "keyword": keyword_weight},
            "source_filter": source, "results": results, "count": len(results)
        }, indent=2)
    except Exception as e:
        logger.error(f"Error in perform_hybrid_search: {str(e)}", exc_info=True)
        raise ToolError(f"Error in hybrid search: {str(e)}", "HYBRID_SEARCH_ERROR", {"query": query, "original_exception": str(e)})

@mcp.tool()
async def get_similar_items(
    item_id: str,
    filter_source: Optional[str] = None,
    match_count: int = 5
) -> str:
    """
    Find similar items based on vector similarity using Qdrant's recommendation API.
    """
    try:
        # Initialize components from environment
        qdrant_client_instance = get_qdrant_client()
        collection_name_str = os.getenv("QDRANT_COLLECTION")
        if not collection_name_str:
            raise ValueError("QDRANT_COLLECTION environment variable must be set.")

        filter_condition = {"source": filter_source} if filter_source and filter_source.strip() else None
        
        results = await get_similar_items_util(
            client=qdrant_client_instance,
            collection_name=collection_name_str,
            item_id=item_id,
            filter_condition=filter_condition, 
            match_count=match_count
        )
        return json.dumps({"success": True, "item_id": item_id, "source_filter": filter_source, "results": results, "count": len(results)}, indent=2)
    except Exception as e:
        logger.error(f"Error in get_similar_items: {str(e)}", exc_info=True)
        raise ToolError(f"Error finding similar items: {str(e)}", "SIMILAR_ITEMS_ERROR", {"item_id": item_id, "original_exception": str(e)})

@mcp.tool()
async def fetch_item_by_id(item_id: str) -> str:
    """
    Fetch a specific item by ID from a Qdrant collection.
    """
    try:
        # Initialize components from environment
        qdrant_client_instance = get_qdrant_client()
        collection_name_str = os.getenv("QDRANT_COLLECTION")
        if not collection_name_str:
            raise ValueError("QDRANT_COLLECTION environment variable must be set.")

        result = await fetch_item_by_id_util(
            client=qdrant_client_instance,
            collection_name=collection_name_str,
            item_id=item_id
        )
        if result:
            return json.dumps({"success": True, "item_id": item_id, "item": result}, indent=2)
        else:
            raise ToolError(f"Item with ID '{item_id}' not found.", "NOT_FOUND", {"item_id": item_id})
    except Exception as e:
        if isinstance(e, ToolError) and e.args and isinstance(e.args[0], str) and e.args[0] == f"Item with ID '{item_id}' not found.":
             raise
        logger.error(f"Error in fetch_item_by_id for item '{item_id}': {str(e)}", exc_info=True)
        raise ToolError(f"Error fetching item {item_id}: {str(e)}", "FETCH_ITEM_ERROR", {"item_id": item_id, "original_exception": str(e)})

@mcp.tool()
async def find_similar_content(
    content_text: str,
    filter_source: Optional[str] = None,
    match_count: int = 5
) -> str:
    """
    Find similar content based on text, not an existing item ID.
    """
    try:
        # Initialize components from environment
        qdrant_client_instance = get_qdrant_client()
        collection_name_str = os.getenv("QDRANT_COLLECTION")
        if not collection_name_str:
            raise ValueError("QDRANT_COLLECTION environment variable must be set.")

        filter_condition = {"source": filter_source} if filter_source and filter_source.strip() else None
        
        results = await find_similar_content_util(
            client=qdrant_client_instance,
            collection_name=collection_name_str,
            content_text=content_text,
            filter_condition=filter_condition, 
            match_count=match_count
        )
        return json.dumps({"success": True, "text_length": len(content_text), "source_filter": filter_source, "results": results, "count": len(results)}, indent=2)
    except Exception as e:
        logger.error(f"Error in find_similar_content: {str(e)}", exc_info=True)
        raise ToolError(f"Error finding similar content: {str(e)}", "SIMILAR_CONTENT_ERROR", {"content_text_snippet": content_text[:100], "original_exception": str(e)})

# Ensure the file ends with a newline for linters 