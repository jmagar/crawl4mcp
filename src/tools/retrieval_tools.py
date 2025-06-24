"""
MCP Tools for retrieving and searching content from Qdrant.
"""
import asyncio
import json
from typing import Optional, Dict, Any
import os
import logging
from urllib.parse import urlparse

from mcp.server.fastmcp.exceptions import ToolError # Correct import path

# Import the centralized mcp instance
from src.mcp_setup import mcp
# Import utility functions
from src.utils.qdrant.setup import get_qdrant_client

from src.utils.qdrant.retrieval import (
    query_qdrant,
    query_qdrant_rrf,
    query_qdrant_rrf_native,
    perform_hybrid_search as perform_hybrid_search_util,
    get_similar_items as get_similar_items_util,
    fetch_item_by_id as fetch_item_by_id_util,
    find_similar_content as find_similar_content_util
)
# Import logging utilities
from src.utils.logging_utils import get_logger

# Initialize logger
logger = get_logger(__name__)

@mcp.tool()
async def perform_rag_query(
    query: str,
    source: Optional[str] = None,
    match_count: int = 5,
) -> Dict[str, Any]:
    """
    Performs a RAG (Retrieval Augmented Generation) query on the stored content.
    Args:
        query: The search query.
        source: Optional source domain to filter results (e.g., 'example.com' or 'https://example.com').
        match_count: Maximum number of results to return (default: 5).
    """
    logger.info(f"RAG query: '{query}' (source filter: {source}, match count: {match_count})")
    qdrant_client_instance = get_qdrant_client()
    collection_name_str = os.getenv("QDRANT_COLLECTION", "default_collection")
    results = []

    try:
        collection_info = await qdrant_client_instance.get_collection(collection_name=collection_name_str)
        # Correctly check for sparse vector support which indicates RRF capability
        has_rrf_support = hasattr(collection_info, 'config') and hasattr(collection_info.config, 'sparse_vectors_config') and collection_info.config.sparse_vectors_config is not None

        if has_rrf_support:
            logger.debug(f"Using RRF search for collection: {collection_name_str}")
            results = await query_qdrant_rrf_native(
                client=qdrant_client_instance,
                collection_name=collection_name_str,
                query_text=query,
                source_filter=source,
                match_count=match_count,
            )
        else:
            logger.debug("Collection does not have sparse vectors, falling back to standard semantic search.")
            results = await query_qdrant(
                client=qdrant_client_instance,
                collection_name=collection_name_str,
                query_text=query,
                source_filter=source,
                match_count=match_count,
            )
    except Exception as e:
        logger.error(f"Error during RAG query, falling back to standard semantic search: {e}")
        # If the check or RRF query fails, fall back to the basic semantic search
        results = await query_qdrant(
            client=qdrant_client_instance,
            collection_name=collection_name_str,
            query_text=query,
            source_filter=source,
            match_count=match_count,
        )

    logger.info(f"RAG query complete. Found {len(results)} results for '{query}'.")
    return {
        "success": True,
        "query": query,
        "source_filter": source,
        "match_count": len(results),
        "matches": results,
    }

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
