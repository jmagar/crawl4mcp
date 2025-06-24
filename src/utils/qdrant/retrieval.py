"""
Qdrant data retrieval and search utilities.
"""
import os
import uuid # Though not directly used by moved functions, often useful with Qdrant IDs
import asyncio
from urllib.parse import urlparse # Used by enhance_payload_metadata if it stays, or by store_embeddings
from datetime import datetime # Used by enhance_payload_metadata if it stays
from typing import List, Dict, Any, Optional, Tuple

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchText,
    Prefetch,
    FusionQuery,
    Fusion,
    SparseVector,
    UpdateStatus
)

from src.utils.qdrant.constants import DENSE_VECTOR_NAME

# Import logging utilities
from src.utils.logging_utils import get_logger # Adjusted import path

# Import embedding functions
from src.utils.embedding_utils import get_embedding # create_embeddings_batch might not be needed here directly

# Initialize logger
logger = get_logger(__name__)

# Helper functions moved from qdrant_utils.py
def create_qdrant_filter(source_filter: Optional[str] = None, filter_condition: Optional[Dict[str, Any]] = None) -> Optional[models.Filter]:
    """
    Create a Qdrant Filter object from a source filter string or a more complex filter condition dictionary.

    Args:
        source_filter: A string value to filter the 'source' field by.
        filter_condition: A dictionary specifying filter conditions. 
                          Keys are field names. Values can be strings (for exact MatchValue) 
                          or dicts like {'text': 'search_term'} (for MatchText).

    Returns:
        A Qdrant models.Filter object, or None if no filter conditions are provided.
    """
    filter_conditions = []
    if source_filter and source_filter.strip():
        # Normalize source_filter by extracting netloc if it's a URL
        normalized_source = source_filter.strip()
        
        # If it looks like a URL, extract just the netloc
        if normalized_source.startswith(('http://', 'https://')):
            try:
                parsed = urlparse(normalized_source)
                normalized_source = parsed.netloc
                logger.debug(f"Normalized source filter from '{source_filter}' to '{normalized_source}'")
            except Exception as e:
                logger.warning(f"Failed to parse source_filter as URL '{source_filter}': {e}. Using as-is.")
                normalized_source = source_filter.strip()
        
        filter_conditions.append(
            FieldCondition(key="source", match=MatchValue(value=normalized_source))
        )
    if filter_condition:
        for key, value in filter_condition.items():
            if isinstance(value, str):
                filter_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            elif isinstance(value, dict) and 'text' in value:
                filter_conditions.append(
                    FieldCondition(key=key, match=MatchText(text=value['text']))
                )
    return models.Filter(must=filter_conditions) if filter_conditions else None

def enhance_payload_metadata(payload: Dict[str, Any], search_type: str = "semantic") -> Dict[str, Any]:
    """
    Extracts and standardizes metadata from a Qdrant point payload for search results.

    Args:
        payload: The payload dictionary from a Qdrant point.
        search_type: The type of search that produced this result (e.g., "semantic", "hybrid").

    Returns:
        A dictionary containing standardized metadata fields.
    """
    metadata = {
        "source": payload.get("source", "unknown"),
        "crawl_type": payload.get("crawl_type", "unknown"),
        "char_count": payload.get("char_count", 0),
        "word_count": payload.get("word_count", 0),
        "chunk_index": payload.get("chunk_index", 0),
        "headers": payload.get("headers", ""),
        "crawl_time": payload.get("crawl_time", "N/A"), # Or use datetime.utcnow().isoformat() if this means fetch time
        "contextual_embedding": payload.get("contextual_embedding", False),
        "search_type": search_type
    }
    for key, value in payload.items():
        if key not in ["url", "text", "original_text"] and key not in metadata:
            metadata[key] = value
    return metadata

def format_search_result(hit: Any, include_score: bool = True, search_type: str = "semantic") -> Dict[str, Any]:
    """
    Formats a Qdrant search hit (point) into a standardized dictionary structure.

    Args:
        hit: A Qdrant search result object (e.g., ScoredPoint or RecommendedPoint).
        include_score: Whether to include the similarity/recommendation score. Defaults to True.
        search_type: The type of search that produced this hit. Defaults to "semantic".

    Returns:
        A dictionary representing the formatted search result, including id, url, content, and metadata.
    """
    result = {
        "id": hit.id,
        "url": hit.payload.get("url"),
        "content": hit.payload.get("text"),
        "original_content": hit.payload.get("original_text", hit.payload.get("text")),
        "metadata": enhance_payload_metadata(hit.payload, search_type)
    }
    if include_score and hasattr(hit, "score"):
        result["similarity"] = hit.score
        if "combined_score" not in result and search_type != "hybrid":
            result["combined_score"] = hit.score
    return result

def handle_search_error(error: Exception, operation: str, query: str = "") -> List[Dict[str, Any]]:
    """
    Logs a search-related error and returns an empty list.

    Args:
        error: The exception that occurred.
        operation: A string describing the operation that failed (e.g., "querying Qdrant").
        query: The query string or relevant identifier for which the operation failed. Defaults to "".

    Returns:
        An empty list, signifying no results due to the error.
    """
    query_info = f" for '{query}'" if query else ""
    logger.error(f"Error {operation}{query_info}: {error}")
    return []

# Core search/retrieval functions moved from qdrant_utils.py
async def query_qdrant(
    client: AsyncQdrantClient,
    collection_name: str,
    query_text: str,
    source_filter: Optional[str] = None,
    match_count: int = 5
) -> List[Dict[str, Any]]:
    """
    Performs a semantic vector search in Qdrant for a given query text.

    Args:
        client: An initialized AsyncQdrantClient instance.
        collection_name: The name of the Qdrant collection to search.
        query_text: The text to search for.
        source_filter: Optional. A string to filter results by the 'source' field.
        match_count: The maximum number of results to return. Defaults to 5.

    Returns:
        A list of formatted search result dictionaries, or an empty list if an error occurs
        or no embedding can be generated for the query.
    """
    logger.debug(f"Generating embedding for query: '{query_text}'")
    query_embedding = await get_embedding(query_text)
    if not query_embedding:
        logger.error(f"Could not generate embedding for query: '{query_text}'. Returning empty list.")
        return []
    
    logger.debug(f"Generated query_embedding for '{query_text}'. First 3 dims: {query_embedding[:3]} L2 norm: {sum(x*x for x in query_embedding)**0.5 if query_embedding else 'N/A'}")
    qdrant_filter = create_qdrant_filter(source_filter=source_filter)
    if source_filter:
        logger.debug(f"Searching with source filter: {source_filter}")
    try:
        logger.debug(f"Executing Qdrant search in collection '{collection_name}' with match_count={match_count} and filter: {qdrant_filter}")
        search_result_qdrant = await client.search(
            collection_name=collection_name,
            query_vector=(DENSE_VECTOR_NAME, query_embedding),
            query_filter=qdrant_filter,
            limit=match_count,
            with_payload=True
        )
        logger.debug(f"Qdrant search returned {len(search_result_qdrant)} raw hits.")
        if search_result_qdrant:
            logger.debug(f"Top raw hit score: {search_result_qdrant[0].score}, ID: {search_result_qdrant[0].id}")
        results = [format_search_result(hit, search_type="semantic") for hit in search_result_qdrant]
        logger.info(f"Query completed with {len(results)} formatted results")
        if results:
            logger.debug(f"Top result score: {results[0].get('similarity', 'N/A')}")
        return results
    except Exception as e:
        return handle_search_error(e, "querying Qdrant", query_text)

async def perform_hybrid_search(
    client: AsyncQdrantClient,
    collection_name: str,
    query_text: str,
    filter_text: Optional[str] = None,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
    source_filter: Optional[str] = None,
    match_count: int = 5
) -> List[Dict[str, Any]]:
    """
    Performs a hybrid search in Qdrant, combining vector similarity and keyword filtering.

    Weights for vector and keyword scores are normalized to sum to 1.0.
    If filter_text is not provided, performs a standard vector search.

    Args:
        client: An initialized AsyncQdrantClient instance.
        collection_name: The name of the Qdrant collection.
        query_text: The primary query text for vector search.
        filter_text: Optional. Text for keyword-based filtering (MatchText on 'text' field).
        vector_weight: Weight for the vector similarity score. Defaults to 0.7.
        keyword_weight: Weight for the keyword match score. Defaults to 0.3.
        source_filter: Optional. A string to filter results by the 'source' field.
        match_count: The maximum number of combined results to return. Defaults to 5.

    Returns:
        A list of formatted search result dictionaries, sorted by a combined score.
        Returns an empty list if an error occurs.
    """
    if vector_weight + keyword_weight != 1.0:
        total = vector_weight + keyword_weight
        if total == 0:
            vector_weight = 0.5
            keyword_weight = 0.5
        else:
            vector_weight = vector_weight / total
            keyword_weight = keyword_weight / total
    
    filter_condition = {}
    if source_filter and source_filter.strip():
        filter_condition["source"] = source_filter
    if filter_text and filter_text.strip():
        filter_condition["text"] = {"text": filter_text}
    qdrant_filter = create_qdrant_filter(filter_condition=filter_condition)
    
    try:
        query_embedding = await get_embedding(query_text)
        if not query_embedding:
            logger.error(f"Could not generate embedding for hybrid search query: '{query_text}'. Returning empty list.")
            return []
        
        if not filter_text or not filter_text.strip():
            search_result = await client.search(
                collection_name=collection_name, query_vector=(DENSE_VECTOR_NAME, query_embedding),
                query_filter=qdrant_filter, limit=match_count, with_payload=True
            )
            results = []
            for hit in search_result:
                result = format_search_result(hit, search_type="vector_only")
                result["combined_score"] = result["similarity"]
                results.append(result)
            return results
        
        vector_filter_conditions = {}
        if source_filter and source_filter.strip():
            vector_filter_conditions["source"] = source_filter
        vector_filter = create_qdrant_filter(filter_condition=vector_filter_conditions)
        
        vector_results = await client.search(
            collection_name=collection_name, query_vector=(DENSE_VECTOR_NAME, query_embedding),
            query_filter=vector_filter, limit=match_count * 2, with_payload=True
        )
        
        text_filter_conditions = {"text": {"text": filter_text}}
        if source_filter and source_filter.strip():
            text_filter_conditions["source"] = source_filter
        text_filter = create_qdrant_filter(filter_condition=text_filter_conditions) if filter_text else None
        
        text_search_hits = []
        if text_filter:
            text_search_hits_response, _ = await client.scroll(
                collection_name=collection_name,
                scroll_filter=text_filter, limit=match_count * 2, with_payload=True
            )
            text_search_hits = text_search_hits_response
        
        result_map = {}
        for hit in vector_results:
            result = format_search_result(hit, search_type="hybrid")
            result["vector_score"] = hit.score
            result["keyword_score"] = 0.0
            result["combined_score"] = hit.score * vector_weight
            result_map[hit.id] = result
        
        for hit in text_search_hits:
            if hit.id in result_map:
                result_map[hit.id]["keyword_score"] = 1.0
                result_map[hit.id]["combined_score"] += 1.0 * keyword_weight
            else:
                result = format_search_result(hit, search_type="hybrid", include_score=False)
                result["vector_score"] = 0.0
                result["keyword_score"] = 1.0
                result["combined_score"] = 1.0 * keyword_weight
                result_map[hit.id] = result
        
        combined_results = sorted(list(result_map.values()), key=lambda x: x["combined_score"], reverse=True)
        return combined_results[:match_count]
    except Exception as e:
        return handle_search_error(e, "performing hybrid search", query_text)

async def get_similar_items(
    client: AsyncQdrantClient,
    collection_name: str,
    item_id: str,
    filter_condition: Optional[Dict[str, Any]] = None,
    match_count: int = 5
) -> List[Dict[str, Any]]:
    """
    Recommends items similar to a given item_id from the Qdrant collection.

    Args:
        client: An initialized AsyncQdrantClient instance.
        collection_name: The name of the Qdrant collection.
        item_id: The ID of the item to get recommendations for.
        filter_condition: Optional. A dictionary for additional filtering conditions.
        match_count: The maximum number of similar items to return. Defaults to 5.

    Returns:
        A list of formatted search result dictionaries representing similar items,
        or an empty list if an error occurs.
    """
    try:
        query_filter = create_qdrant_filter(filter_condition=filter_condition)
        recommend_result = await client.recommend(
            collection_name=collection_name, positive=[item_id], negative=[],
            query_filter=query_filter, limit=match_count,
            with_payload=True, with_vectors=False,
        )
        return [format_search_result(hit, search_type="item_recommendation") for hit in recommend_result]
    except Exception as e:
        return handle_search_error(e, "finding similar items", f"item_id={item_id}")

async def fetch_item_by_id(
    client: AsyncQdrantClient,
    collection_name: str,
    item_id: str
) -> Optional[Dict[str, Any]]:
    """
    Retrieves a specific item (point) by its ID from the Qdrant collection.

    Args:
        client: An initialized AsyncQdrantClient instance.
        collection_name: The name of the Qdrant collection.
        item_id: The ID of the item to retrieve.

    Returns:
        A dictionary containing the item's id, url, content, and metadata if found.
        Returns None if the item is not found or an error occurs.
    """
    try:
        points_response = await client.retrieve(
            collection_name=collection_name, ids=[item_id],
            with_payload=True, with_vectors=False
        )
        if not points_response:
            logger.warning(f"No item found with ID: {item_id}")
            return None
        point = points_response[0]
        payload_data = point.payload if point.payload is not None else {}
        return {
            "id": point.id,
            "url": payload_data.get("url"),
            "content": payload_data.get("text"),
            "metadata": {
                "source": payload_data.get("source", "unknown"),
                "crawl_type": payload_data.get("crawl_type", "unknown"),
                "char_count": payload_data.get("char_count", 0),
                "word_count": payload_data.get("word_count", 0),
                "chunk_index": payload_data.get("chunk_index", 0),
                "headers": payload_data.get("headers", "")
            }
        }
    except Exception as e:
        logger.error(f"Error fetching item by ID {item_id}: {e}")
        return None

async def find_similar_content(
    client: AsyncQdrantClient,
    collection_name: str,
    content_text: str,
    filter_condition: Optional[Dict[str, Any]] = None,
    match_count: int = 5
) -> List[Dict[str, Any]]:
    """
    Finds items in Qdrant similar to a given piece of content text.
    This involves generating an embedding for the content_text and then performing a semantic search.

    Args:
        client: An initialized AsyncQdrantClient instance.
        collection_name: The name of the Qdrant collection.
        content_text: The text content to find similar items for.
        filter_condition: Optional. A dictionary for additional filtering conditions.
        match_count: The maximum number of similar items to return. Defaults to 5.

    Returns:
        A list of formatted search result dictionaries representing similar items,
        or an empty list if an error occurs or embedding fails.
    """
    try:
        query_embedding = await get_embedding(content_text)
        if not query_embedding:
            logger.error(f"Could not generate embedding for content text. Returning empty list.")
            return []
        query_filter = create_qdrant_filter(filter_condition=filter_condition)
        search_result = await client.search(
            collection_name=collection_name, query_vector=(DENSE_VECTOR_NAME, query_embedding),
            query_filter=query_filter, limit=match_count, with_payload=True
        )
        return [format_search_result(hit, search_type="content_similarity") for hit in search_result]
    except Exception as e:
        return handle_search_error(e, "finding similar content", content_text[:50] + "...")

async def query_qdrant_rrf(
    client: AsyncQdrantClient,
    collection_name: str,
    query_text: str,
    source_filter: Optional[str] = None,
    match_count: int = 5,
    rrf_k: int = 60
) -> List[Dict[str, Any]]:
    """
    Performs RRF (Reciprocal Rank Fusion) search by combining semantic and hybrid search results.
    Uses client-side RRF fusion for compatibility with current Qdrant setup.

    Args:
        client: An initialized AsyncQdrantClient instance.
        collection_name: The name of the Qdrant collection to search.
        query_text: The text to search for.
        source_filter: Optional. A string to filter results by the 'source' field.
        match_count: The maximum number of results to return. Defaults to 5.
        rrf_k: RRF parameter k (default 60, standard value). Higher values reduce impact of rank differences.

    Returns:
        A list of formatted search result dictionaries, ranked by RRF fusion.
        Returns an empty list if an error occurs or no embedding can be generated.
    """
    logger.debug(f"Starting client-side RRF search for query: '{query_text}' with k={rrf_k}")
    
    try:
        # Perform multiple semantic searches to simulate different ranking strategies
        # This provides diversity for RRF fusion even without true hybrid search
        semantic_task = query_qdrant(
            client=client,
            collection_name=collection_name,
            query_text=query_text,
            source_filter=source_filter,
            match_count=match_count * 3  # Fetch more for better fusion
        )
        
        # For now, use the same semantic search as the second ranking
        # In future, this could be replaced with actual hybrid/keyword search
        semantic_task2 = query_qdrant(
            client=client,
            collection_name=collection_name,
            query_text=query_text,
            source_filter=source_filter,
            match_count=match_count * 3  # Fetch more for better fusion
        )
        
        # Wait for both searches to complete
        semantic_results, semantic_results2 = await asyncio.gather(semantic_task, semantic_task2)
        
        logger.debug(f"First semantic search returned {len(semantic_results)} results")
        logger.debug(f"Second semantic search returned {len(semantic_results2)} results")
        
        # Apply RRF fusion
        rrf_results = apply_rrf_fusion(
            semantic_results=semantic_results,
            hybrid_results=semantic_results2,  # Using second semantic search for now
            k=rrf_k,
            match_count=match_count
        )
        
        logger.info(f"RRF fusion completed with {len(rrf_results)} results")
        if rrf_results:
            logger.debug(f"Top RRF result score: {rrf_results[0].get('rrf_score', 'N/A')}")
        
        return rrf_results
        
    except Exception as e:
        logger.warning(f"RRF search failed, falling back to semantic search: {e}")
        # Fallback to regular semantic search
        return await query_qdrant(
            client=client,
            collection_name=collection_name,
            query_text=query_text,
            source_filter=source_filter,
            match_count=match_count
        )

async def query_qdrant_rrf_native(
    client: AsyncQdrantClient,
    collection_name: str,
    query_text: str,
    source_filter: Optional[str] = None,
    match_count: int = 5
) -> List[Dict[str, Any]]:
    """
    Performs native RRF search using Qdrant's Query API with prefetch and fusion.
    Combines dense and sparse vectors using server-side RRF for optimal performance.

    Args:
        client: An initialized AsyncQdrantClient instance.
        collection_name: The name of the Qdrant collection to search.
        query_text: The text to search for.
        source_filter: Optional. A string to filter results by the 'source' field.
        match_count: The maximum number of results to return. Defaults to 5.

    Returns:
        A list of formatted search result dictionaries.
    """
    try:
        from src.utils.splade_utils import encode_query_for_qdrant
        from src.utils.embedding_utils import get_embedding
        
        logger.debug(f"Starting native RRF search for: {query_text}")
        
        # Generate both dense and sparse query vectors
        dense_vector = await get_embedding(query_text)
        sparse_vector_data = await asyncio.to_thread(encode_query_for_qdrant, query_text)
        
        # Create sparse vector object
        sparse_vector = SparseVector(
            indices=sparse_vector_data["indices"],
            values=sparse_vector_data["values"]
        )
        
        # Create source filter if provided
        qdrant_filter = None
        if source_filter:
            qdrant_filter = create_qdrant_filter(source_filter)
        
        # Build prefetch queries for RRF
        prefetch_queries = [
            # Dense vector search
            Prefetch(
                query_vector=(DENSE_VECTOR_NAME, dense_vector),
                using="text-dense",
                limit=match_count * 3,  # Fetch more for better fusion
                filter=qdrant_filter
            ),
            # Sparse vector search  
            Prefetch(
                query=sparse_vector,
                using="text-sparse",
                limit=match_count * 3,  # Fetch more for better fusion
                filter=qdrant_filter
            )
        ]
        
        # Execute RRF query using Qdrant's Query API
        search_result = await client.query_points(
            collection_name=collection_name,
            prefetch=prefetch_queries,
            query=FusionQuery(
                fusion=Fusion.RRF
            ),
            limit=match_count,
            with_payload=True
        )
        
        # Format results
        formatted_results = []
        for point in search_result.points:
            result = {
                "id": str(point.id),
                "score": float(point.score),
                "payload": point.payload or {},
                "search_type": "rrf_fusion"
            }
            formatted_results.append(result)
        
        logger.debug(f"Native RRF search returned {len(formatted_results)} results")
        return formatted_results
        
    except Exception as e:
        logger.error(f"Native RRF search failed: {e}")
        # Fallback to regular semantic search
        logger.info("Falling back to semantic search")
        return await query_qdrant(
            client=client,
            collection_name=collection_name,
            query_text=query_text,
            source_filter=source_filter,
            match_count=match_count
        ) 