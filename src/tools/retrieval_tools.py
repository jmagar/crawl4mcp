"""
MCP Tools for retrieving and searching content from Qdrant.
"""
import json
from typing import Optional
import os

from mcp.server.fastmcp import Context # MCP Context for tool arguments

# Import the centralized mcp instance
from ..mcp_setup import mcp
# Import utility functions from qdrant_utils
from ..utils.qdrant_utils import (
    query_qdrant,
    perform_hybrid_search as perform_hybrid_search_util, # Alias to avoid name clash
    get_similar_items as get_similar_items_util,       # Alias
    fetch_item_by_id as fetch_item_by_id_util,         # Alias
    find_similar_content as find_similar_content_util,  # Alias
    get_qdrant_client                                  
)

@mcp.tool()
async def perform_rag_query(ctx: Context, query: str, source: Optional[str] = None, match_count: int = 5) -> str:
    """
    Perform a RAG (Retrieval Augmented Generation) query on the stored content.
    Args:
        query: The search query.
        source: Optional source domain to filter results (e.g., 'example.com').
        match_count: Maximum number of results to return (default: 5).
    """
    qdrant_client_instance = None
    collection_name_str = None

    try:
        if hasattr(ctx, 'request_context') and ctx.request_context is not None:
            qdrant_client_instance = ctx.request_context.lifespan_context.qdrant_client
            collection_name_str = ctx.request_context.lifespan_context.collection_name
        else:
            raise AttributeError("request_context not available for perform_rag_query")
    except (AttributeError, ValueError) as e:
        print(f"Context access failed for perform_rag_query ({type(e).__name__}: {e}). Initializing Qdrant from environment.")
        try:
            qdrant_client_instance = get_qdrant_client()
            collection_name_str = os.getenv("QDRANT_COLLECTION")
            if not collection_name_str:
                raise ValueError("QDRANT_COLLECTION environment variable must be set.")
        except Exception as e_init:
            return json.dumps({"success": False, "query": query, "error": f"Failed to initialize Qdrant: {str(e_init)}"}, indent=2)

    if not all([qdrant_client_instance, collection_name_str]):
        return json.dumps({"success": False, "query": query, "error": "Qdrant client or collection name missing."}, indent=2)

    try:
        results = await query_qdrant(
            client=qdrant_client_instance,
            collection_name=collection_name_str,
            query_text=query,
            source_filter=source if source and source.strip() else None,
            match_count=match_count
        )
        return json.dumps({"success": True, "query": query, "source_filter": source, "results": results, "count": len(results)}, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "query": query, "error": str(e)}, indent=2)

@mcp.tool()
async def perform_hybrid_search(
    ctx: Context, 
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
    qdrant_client_instance = None
    collection_name_str = None

    try:
        if hasattr(ctx, 'request_context') and ctx.request_context is not None:
            qdrant_client_instance = ctx.request_context.lifespan_context.qdrant_client
            collection_name_str = ctx.request_context.lifespan_context.collection_name
        else:
            raise AttributeError("request_context not available for perform_hybrid_search")
    except (AttributeError, ValueError) as e:
        print(f"Context access failed for perform_hybrid_search ({type(e).__name__}: {e}). Initializing Qdrant from environment.")
        try:
            qdrant_client_instance = get_qdrant_client()
            collection_name_str = os.getenv("QDRANT_COLLECTION")
            if not collection_name_str:
                raise ValueError("QDRANT_COLLECTION environment variable must be set.")
        except Exception as e_init:
            return json.dumps({"success": False, "query": query, "error": f"Failed to initialize Qdrant: {str(e_init)}"}, indent=2)

    if not all([qdrant_client_instance, collection_name_str]):
        return json.dumps({"success": False, "query": query, "error": "Qdrant client or collection name missing."}, indent=2)

    try:
        if not (0.0 <= vector_weight <= 1.0 and 0.0 <= keyword_weight <= 1.0):
            return json.dumps({"success": False, "query": query, "error": "Weights must be between 0.0 and 1.0"}, indent=2)
        
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
        return json.dumps({"success": False, "query": query, "error": str(e)}, indent=2)

@mcp.tool()
async def get_similar_items(
    ctx: Context,
    item_id: str,
    filter_source: Optional[str] = None,
    match_count: int = 5
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
        print(f"Context access failed for get_similar_items ({type(e).__name__}: {e}). Initializing Qdrant from environment.")
        try:
            qdrant_client_instance = get_qdrant_client()
            collection_name_str = os.getenv("QDRANT_COLLECTION")
            if not collection_name_str:
                raise ValueError("QDRANT_COLLECTION environment variable must be set.")
        except Exception as e_init:
            return json.dumps({"success": False, "item_id": item_id, "error": f"Failed to initialize Qdrant: {str(e_init)}"}, indent=2)

    if not all([qdrant_client_instance, collection_name_str]):
        return json.dumps({"success": False, "item_id": item_id, "error": "Qdrant client or collection name missing."}, indent=2)

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
        return json.dumps({"success": False, "item_id": item_id, "error": str(e)}, indent=2)

@mcp.tool()
async def fetch_item_by_id(ctx: Context, item_id: str) -> str:
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
        print(f"Context access failed for fetch_item_by_id ({type(e).__name__}: {e}). Initializing Qdrant from environment.")
        try:
            qdrant_client_instance = get_qdrant_client()
            collection_name_str = os.getenv("QDRANT_COLLECTION")
            if not collection_name_str:
                raise ValueError("QDRANT_COLLECTION environment variable must be set.")
        except Exception as e_init:
            return json.dumps({"success": False, "item_id": item_id, "error": f"Failed to initialize Qdrant: {str(e_init)}"}, indent=2)

    if not all([qdrant_client_instance, collection_name_str]):
        return json.dumps({"success": False, "item_id": item_id, "error": "Qdrant client or collection name missing."}, indent=2)

    try:
        result = await fetch_item_by_id_util(
            client=qdrant_client_instance,
            collection_name=collection_name_str,
            item_id=item_id
        )
        if result:
            return json.dumps({"success": True, "item_id": item_id, "item": result}, indent=2)
        else:
            return json.dumps({"success": False, "item_id": item_id, "error": "Item not found"}, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "item_id": item_id, "error": str(e)}, indent=2)

@mcp.tool()
async def find_similar_content(
    ctx: Context,
    content_text: str,
    filter_source: Optional[str] = None,
    match_count: int = 5
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
        print(f"Context access failed for find_similar_content ({type(e).__name__}: {e}). Initializing Qdrant from environment.")
        try:
            qdrant_client_instance = get_qdrant_client()
            collection_name_str = os.getenv("QDRANT_COLLECTION")
            if not collection_name_str:
                raise ValueError("QDRANT_COLLECTION environment variable must be set.")
        except Exception as e_init:
            return json.dumps({"success": False, "error": f"Failed to initialize Qdrant: {str(e_init)}"}, indent=2)

    if not all([qdrant_client_instance, collection_name_str]):
        return json.dumps({"success": False, "error": "Qdrant client or collection name missing for find_similar_content." }, indent=2)

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
        return json.dumps({"success": False, "error": str(e)}, indent=2)

# Ensure the file ends with a newline for linters 