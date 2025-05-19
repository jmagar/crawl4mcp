"""
MCP Tools for retrieving and searching content from Qdrant.
"""
import json
from typing import Optional
import os
import logging

from mcp.server.fastmcp import Context # MCP Context for tool arguments
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
async def perform_rag_query(query: str, source: Optional[str] = None, match_count: int = 5, ctx: Optional[Context] = None) -> str:
    """
    Perform a RAG (Retrieval Augmented Generation) query on the stored content.
    Args:
        query: The search query.
        source: Optional source domain to filter results (e.g., 'example.com').
        match_count: Maximum number of results to return (default: 5).
        ctx: Optional context object
    """
    logger.info(f"RAG query: '{query}' (source filter: {source}, match count: {match_count})")
    
    try:
        # Try to get qdrant_client and collection_name from context
        qdrant_client_instance = None
        collection_name_str = None
        
        if hasattr(ctx, 'request_context') and ctx.request_context is not None:
            qdrant_client_instance = ctx.request_context.lifespan_context.qdrant_client
            collection_name_str = ctx.request_context.lifespan_context.collection_name
            logger.debug("Using Qdrant client and collection from request context")
        else:
            # Fallback to environment variables
            logger.warning("Context not available, initializing Qdrant client from environment variables")
            qdrant_client_instance = get_qdrant_client()
            collection_name_str = os.getenv("QDRANT_COLLECTION")
            if not collection_name_str:
                error_msg = "QDRANT_COLLECTION environment variable must be set when context is not available"
                logger.error(error_msg)
                raise ToolError(message=error_msg, code="CONFIG_ERROR")
        
        # Execute the query
        logger.debug(f"Executing RAG query against collection: {collection_name_str}")
        query_result = await query_qdrant(
            client=qdrant_client_instance,
            collection_name=collection_name_str,
            query_text=query,
            source_filter=source,
            limit=match_count,
            with_vectors=False,
            with_payload=True
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
            message=f"Error in RAG query: {str(e)}",
            code="RAG_QUERY_ERROR",
            details={"query": query, "source": source, "original_exception": str(e)}
        )

@mcp.tool()
async def perform_hybrid_search(
    query: str, 
    filter_text: Optional[str] = None, 
    vector_weight: float = 0.7, 
    keyword_weight: float = 0.3, 
    source: Optional[str] = None, 
    match_count: int = 5,
    ctx: Optional[Context] = None
) -> str:
    """
    Perform a hybrid search combining vector similarity with keyword/text-based filtering.
    """
    qdrant_client_instance = None
    collection_name_str = None

    try:
        if hasattr(ctx, 'request_context') and ctx.request_context is not None:
            qdrant_client_instance = ctx.request_context.lifespan_context.qdrant_client
            collection_name_str = ctx.request_context.lifespan_context.collection_name
        else:
            raise AttributeError("request_context not available for perform_hybrid_search")
    except (AttributeError, ValueError) as e:
        logger.warning(f"Context access failed for perform_hybrid_search ({type(e).__name__}: {e}). Initializing Qdrant from environment.")
        try:
            qdrant_client_instance = get_qdrant_client()
            collection_name_str = os.getenv("QDRANT_COLLECTION")
            if not collection_name_str:
                raise ValueError("QDRANT_COLLECTION environment variable must be set.")
        except Exception as e_init:
            raise ToolError(message=f"Failed to initialize Qdrant for hybrid search: {str(e_init)}", code="INITIALIZATION_ERROR", details={"original_exception": str(e_init)})

    if not all([qdrant_client_instance, collection_name_str]):
        raise ToolError(message="Qdrant client or collection name missing for hybrid search.", code="MISSING_DEPENDENCY")

    try:
        if not (0.0 <= vector_weight <= 1.0 and 0.0 <= keyword_weight <= 1.0):
            raise ToolError(message="Hybrid search weights must be between 0.0 and 1.0", code="INVALID_PARAMS", details={"vector_weight": vector_weight, "keyword_weight": keyword_weight})
        
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
        raise ToolError(message=f"Error in hybrid search: {str(e)}", code="HYBRID_SEARCH_ERROR", details={"query": query, "original_exception": str(e)})

@mcp.tool()
async def get_similar_items(
    item_id: str,
    filter_source: Optional[str] = None,
    match_count: int = 5,
    ctx: Optional[Context] = None
) -> str:
    """
    Find similar items based on vector similarity using Qdrant's recommendation API.
    """
    qdrant_client_instance = None
    collection_name_str = None

    try:
        if hasattr(ctx, 'request_context') and ctx.request_context is not None:
            qdrant_client_instance = ctx.request_context.lifespan_context.qdrant_client
            collection_name_str = ctx.request_context.lifespan_context.collection_name
        else:
            raise AttributeError("request_context not available for get_similar_items")
    except (AttributeError, ValueError) as e:
        logger.warning(f"Context access failed for get_similar_items ({type(e).__name__}: {e}). Initializing Qdrant from environment.")
        try:
            qdrant_client_instance = get_qdrant_client()
            collection_name_str = os.getenv("QDRANT_COLLECTION")
            if not collection_name_str:
                raise ValueError("QDRANT_COLLECTION environment variable must be set.")
        except Exception as e_init:
            raise ToolError(message=f"Failed to initialize Qdrant for similar items search: {str(e_init)}", code="INITIALIZATION_ERROR", details={"original_exception": str(e_init)})

    if not all([qdrant_client_instance, collection_name_str]):
        raise ToolError(message="Qdrant client or collection name missing for similar items search.", code="MISSING_DEPENDENCY")

    try:
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
        raise ToolError(message=f"Error finding similar items: {str(e)}", code="SIMILAR_ITEMS_ERROR", details={"item_id": item_id, "original_exception": str(e)})

@mcp.tool()
async def fetch_item_by_id(item_id: str, ctx: Optional[Context] = None) -> str:
    """
    Fetch a specific item by ID from a Qdrant collection.
    """
    qdrant_client_instance = None
    collection_name_str = None

    try:
        if hasattr(ctx, 'request_context') and ctx.request_context is not None:
            qdrant_client_instance = ctx.request_context.lifespan_context.qdrant_client
            collection_name_str = ctx.request_context.lifespan_context.collection_name
        else:
            raise AttributeError("request_context not available for fetch_item_by_id")
    except (AttributeError, ValueError) as e:
        logger.warning(f"Context access failed for fetch_item_by_id ({type(e).__name__}: {e}). Initializing Qdrant from environment.")
        try:
            qdrant_client_instance = get_qdrant_client()
            collection_name_str = os.getenv("QDRANT_COLLECTION")
            if not collection_name_str:
                raise ValueError("QDRANT_COLLECTION environment variable must be set.")
        except Exception as e_init:
            raise ToolError(message=f"Failed to initialize Qdrant for fetch_item_by_id: {str(e_init)}", code="INITIALIZATION_ERROR", details={"original_exception": str(e_init)})

    if not all([qdrant_client_instance, collection_name_str]):
        raise ToolError(message="Qdrant client or collection name missing for fetch_item_by_id.", code="MISSING_DEPENDENCY")

    try:
        result = await fetch_item_by_id_util(
            client=qdrant_client_instance,
            collection_name=collection_name_str,
            item_id=item_id
        )
        if result:
            return json.dumps({"success": True, "item_id": item_id, "item": result}, indent=2)
        else:
            raise ToolError(message=f"Item with ID '{item_id}' not found.", code="NOT_FOUND")
    except Exception as e:
        logger.error(f"Error in fetch_item_by_id: {str(e)}", exc_info=True)
        raise ToolError(message=f"Error fetching item by ID: {str(e)}", code="FETCH_ERROR", details={"item_id": item_id, "original_exception": str(e)})

@mcp.tool()
async def find_similar_content(
    content_text: str,
    filter_source: Optional[str] = None,
    match_count: int = 5,
    ctx: Optional[Context] = None
) -> str:
    """
    Find similar content based on text, not an existing item ID.
    """
    qdrant_client_instance = None
    collection_name_str = None

    try:
        if hasattr(ctx, 'request_context') and ctx.request_context is not None:
            qdrant_client_instance = ctx.request_context.lifespan_context.qdrant_client
            collection_name_str = ctx.request_context.lifespan_context.collection_name
        else:
            raise AttributeError("request_context not available for find_similar_content")
    except (AttributeError, ValueError) as e:
        logger.warning(f"Context access failed for find_similar_content ({type(e).__name__}: {e}). Initializing Qdrant from environment.")
        try:
            qdrant_client_instance = get_qdrant_client()
            collection_name_str = os.getenv("QDRANT_COLLECTION")
            if not collection_name_str:
                raise ValueError("QDRANT_COLLECTION environment variable must be set.")
        except Exception as e_init:
            raise ToolError(message=f"Failed to initialize Qdrant for find_similar_content: {str(e_init)}", code="INITIALIZATION_ERROR", details={"original_exception": str(e_init)})

    if not all([qdrant_client_instance, collection_name_str]):
        raise ToolError(message="Qdrant client or collection name missing for find_similar_content.", code="MISSING_DEPENDENCY")

    try:
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
        raise ToolError(message=f"Error finding similar content: {str(e)}", code="SIMILAR_CONTENT_ERROR", details={"original_exception": str(e)})

# Ensure the file ends with a newline for linters 