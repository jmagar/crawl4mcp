"""
Utility functions specifically for Qdrant interactions.
"""
import os
import uuid
from urllib.parse import urlparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import asyncio # Ensure asyncio is imported

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchText
)
from qdrant_client.http.exceptions import ResponseHandlingException

# Import logging utilities
from .logging_utils import get_logger

# Initialize logger
logger = get_logger(__name__)

# Import embedding functions
from .embedding_utils import get_embedding, create_embeddings_batch, generate_contextual_embedding


def get_qdrant_client() -> QdrantClient:
    """
    Get a Qdrant client with the URL and API key from environment variables.
    
    Returns:
        Qdrant client instance
    """
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    
    if not url:
        logger.error("QDRANT_URL must be set in the environment variables.")
        raise ValueError("QDRANT_URL must be set in the environment variables.")
    
    # Create client with or without API key, depending on what's provided
    if api_key:
        logger.debug(f"Creating Qdrant client with URL {url} and API key")
        return QdrantClient(url=url, api_key=api_key)
    else:
        logger.debug(f"Creating Qdrant client with URL {url} (no API key)")
        return QdrantClient(url=url)

async def ensure_qdrant_collection_async(client: QdrantClient, collection_name: str, vector_dim: int):
    """
    Ensure that the specified collection exists in Qdrant.
    This is an asynchronous version.
    """
    try:
        # Try to get collection info. This might raise ResponseHandlingException if the
        # client has trouble parsing the server's response (e.g., Pydantic validation error).
        client.get_collection(collection_name=collection_name)
        logger.info(f"Collection '{collection_name}' already exists or client successfully parsed its details.")
    except ResponseHandlingException as rHE:
        # This is the Pydantic validation error.
        # Assume the collection *exists*, but the client can't parse the response.
        # DO NOT try to create it, as that would likely lead to a 409 Conflict.
        logger.warning(f"Error parsing server response for collection '{collection_name}': {rHE}. Assuming collection exists but details are unparsable by this client version.")
        # We pass here, effectively treating the collection as existing.
        pass
    except Exception as e:
        # For other errors (e.g., true 'Not Found' if client.get_collection raises something else,
        # or network issues), assume it might not exist and try to create it.
        # Qdrant client versions vary in how they report "Not Found".
        # A more robust check for a 404 status code from 'e' might be needed if this proves problematic.
        logger.warning(f"An unexpected error or 'Not Found' occurred while checking collection '{collection_name}': {e}. Attempting to create it.")
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_dim, distance=models.Distance.COSINE)
            )
            logger.info(f"Collection '{collection_name}' created with vector_dim={vector_dim}.")
        except Exception as final_create_e:
            # If create_collection also fails (e.g., it was a 409 because it did exist,
            # or another network issue), then we log and re-raise.
            logger.error(f"Attempt to create collection '{collection_name}' also failed: {final_create_e}")
            raise

def create_qdrant_filter(source_filter: Optional[str] = None, filter_condition: Optional[Dict[str, Any]] = None) -> Optional[models.Filter]:
    """
    Create a standardized Qdrant filter object from various input types.
    
    This helper function consolidates filter creation logic used across multiple functions.
    
    Args:
        source_filter: Optional source domain for filtering (simple string filter)
        filter_condition: Optional dictionary of filter conditions (more complex filtering)
    
    Returns:
        Qdrant Filter object if any filters are provided, otherwise None
    """
    filter_conditions = []
    
    # Add source filter if provided
    if source_filter and source_filter.strip():
        filter_conditions.append(
            FieldCondition(
                key="source",
                match=MatchValue(value=source_filter)
            )
        )
    
    # Add dictionary-based filters if provided
    if filter_condition:
        for key, value in filter_condition.items():
            if isinstance(value, str):
                filter_conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            elif isinstance(value, dict) and 'text' in value:
                # Support text match filters
                filter_conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchText(text=value['text'])
                    )
                )
    
    # Return Filter object if we have conditions, otherwise None
    return models.Filter(must=filter_conditions) if filter_conditions else None

def format_search_result(hit: Any, include_score: bool = True, search_type: str = "semantic") -> Dict[str, Any]:
    """
    Format a Qdrant search result hit into a standardized dictionary.
    
    This helper function provides consistent result formatting across different search functions.
    
    Args:
        hit: A search result hit from Qdrant
        include_score: Whether to include the similarity score in the result
        search_type: Type of search that produced this result (semantic, hybrid, etc.)
    
    Returns:
        Dictionary with standardized format for search results
    """
    result = {
        "id": hit.id,
        "url": hit.payload.get("url"),
        "content": hit.payload.get("text"),
        "original_content": hit.payload.get("original_text", hit.payload.get("text")),
        "metadata": enhance_payload_metadata(hit.payload, search_type)
    }
    
    # Add score information if available and requested
    if include_score and hasattr(hit, "score"):
        result["similarity"] = hit.score
        
        # For backward compatibility
        if "combined_score" not in result and search_type != "hybrid":
            result["combined_score"] = hit.score
    
    return result

def enhance_payload_metadata(payload: Dict[str, Any], search_type: str = "semantic") -> Dict[str, Any]:
    """
    Create or enhance metadata from a payload with consistent fields.
    
    This helper ensures all metadata fields are present with sensible defaults.
    
    Args:
        payload: The payload dictionary from a Qdrant point
        search_type: Type of search that produced this result
    
    Returns:
        Dictionary with standardized metadata
    """
    # Core metadata fields with defaults
    metadata = {
        "source": payload.get("source", "unknown"),
        "crawl_type": payload.get("crawl_type", "unknown"),
        "char_count": payload.get("char_count", 0),
        "word_count": payload.get("word_count", 0),
        "chunk_index": payload.get("chunk_index", 0),
        "headers": payload.get("headers", ""),
        "crawl_time": payload.get("crawl_time", "N/A"),
        "contextual_embedding": payload.get("contextual_embedding", False),
        "search_type": search_type
    }
    
    # Add any additional metadata fields present in payload
    for key, value in payload.items():
        if key not in ["url", "text", "original_text"] and key not in metadata:
            metadata[key] = value
    
    return metadata

def handle_search_error(error: Exception, operation: str, query: str = "") -> List[Dict[str, Any]]:
    """
    Standardized error handling for search operations.
    
    This helper provides consistent error logging and empty results for error cases.
    
    Args:
        error: The exception that occurred
        operation: Description of the operation that failed (for logging)
        query: Optional query string that was being processed (for logging)
    
    Returns:
        Empty list (standardized error result)
    """
    query_info = f" for '{query}'" if query else ""
    logger.error(f"Error {operation}{query_info}: {error}")
    return []

async def store_embeddings(
    client: QdrantClient,
    collection_name: str,
    chunks: List[Dict[str, Any]],
    source_url: str,
    crawl_type: str,
    query_for_contextual_embedding: Optional[str] = None
) -> Tuple[int, int]:
    """
    Store text chunks and their embeddings in Qdrant, handling contextual embeddings.
    Uses the global EMBEDDING_SERVER_BATCH_SIZE for batching requests to the embedding server.
    Qdrant upsert batch size is controlled by QDRANT_UPSERT_BATCH_SIZE environment variable.
    """
    # Get Qdrant upsert batch size from environment variable or use a default
    try:
        qdrant_upsert_batch_size = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "64"))
        if qdrant_upsert_batch_size <= 0:
            logger.warning(f"QDRANT_UPSERT_BATCH_SIZE must be positive. Defaulting to 64.")
            qdrant_upsert_batch_size = 64
    except ValueError:
        logger.warning(f"QDRANT_UPSERT_BATCH_SIZE is not a valid integer. Defaulting to 64.")
        qdrant_upsert_batch_size = 64

    points_to_upsert = []
    successful_chunks = 0
    failed_chunks = 0

    # Prepare texts for batch embedding
    texts_to_embed = []
    for i, chunk_data in enumerate(chunks):
        text_content = chunk_data.get("text", "")
        # MODIFICATION: Always use original text, bypass summarization
        texts_to_embed.append(text_content) # Embed original text only
        # Old summarization logic to be removed/commented:
        # if query_for_contextual_embedding and openai and SUMMARIZATION_MODEL_CHOICE: # openai and SUMMARIZATION_MODEL_CHOICE will be in embedding_utils
        #     # If contextual embedding is enabled, generate summary and prepend it
        #     contextual_summary = await generate_contextual_embedding(text_content, source_url, query_for_contextual_embedding)
        #     texts_to_embed.append(contextual_summary) # Embed the summary + original text
        # else:
        #     texts_to_embed.append(text_content) # Embed original text only

    # Get all embeddings in a batch
    # create_embeddings_batch will be imported from embedding_utils
    # It will use the EMBEDDING_SERVER_BATCH_SIZE constant defined in embedding_utils
    all_embeddings = await create_embeddings_batch(texts_to_embed)

    for i, chunk_data in enumerate(chunks):
        text_content = chunk_data.get("text", "")
        embedding = all_embeddings[i]

        if not embedding:  # Skip if embedding failed for this chunk
            logger.warning(f"Skipping chunk {i+1} from {source_url} due to embedding failure.")
            failed_chunks += 1
            continue

        # Use the text_content that was actually embedded (original or with summary)
        embedded_text_payload = texts_to_embed[i]

        payload = {
            "url": source_url,
            "text": embedded_text_payload, # Store the text that was embedded
            "source": urlparse(source_url).netloc,
            "crawl_type": crawl_type,
            "char_count": len(text_content), # char_count of the original text
            "word_count": len(text_content.split()), # word_count of the original text
            "chunk_index": i + 1,
            "headers": chunk_data.get("headers", ""),
            "contextual_embedding": False, # Always false now
            "crawl_timestamp": datetime.utcnow().isoformat() # Add timestamp
        }

        points_to_upsert.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=payload
            )
        )
        successful_chunks += 1

        if len(points_to_upsert) >= qdrant_upsert_batch_size: # Use the configured batch size
            try:
                # If this util is called from an async function, it will block.
                # To make it truly async, use an async http library.
                # For simplicity in refactoring the existing synchronous logic:
                await asyncio.to_thread(client.upsert, collection_name=collection_name, points=points_to_upsert)
                logger.info(f"Upserted {len(points_to_upsert)} points to '{collection_name}'.")
                points_to_upsert = []
            except Exception as e:
                logger.error(f"Error upserting batch to Qdrant: {e}")
                # Decide how to handle batch failures, e.g., mark all in batch as failed
                failed_chunks += len(points_to_upsert)
                successful_chunks -= len(points_to_upsert)
                points_to_upsert = [] 

    if points_to_upsert: # Upsert any remaining points
        try:
            # If this util is called from an async function, it will block.
            # To make it truly async, use an async http library.
            # For simplicity in refactoring the existing synchronous logic:
            await asyncio.to_thread(client.upsert, collection_name=collection_name, points=points_to_upsert)
            logger.info(f"Upserted remaining {len(points_to_upsert)} points to '{collection_name}'.")
        except Exception as e:
            logger.error(f"Error upserting final batch to Qdrant: {e}")
            failed_chunks += len(points_to_upsert)
            successful_chunks -= len(points_to_upsert)

    return successful_chunks, failed_chunks

async def query_qdrant(
    client: QdrantClient,
    collection_name: str,
    query_text: str,
    source_filter: Optional[str] = None,
    match_count: int = 5
) -> List[Dict[str, Any]]:
    """
    Query Qdrant for relevant documents.
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
        search_result_qdrant = await asyncio.to_thread(
            client.search,
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=qdrant_filter,
            limit=match_count,
            with_payload=True
        )
        
        logger.debug(f"Qdrant search returned {len(search_result_qdrant)} raw hits.")
        if search_result_qdrant:
            logger.debug(f"Top raw hit score: {search_result_qdrant[0].score}, ID: {search_result_qdrant[0].id}")
            # logger.debug(f"Top raw hit payload: {search_result_qdrant[0].payload}") # Can be very verbose

        results = []
        for hit in search_result_qdrant:
            result = format_search_result(hit, search_type="semantic")
            results.append(result)
        
        logger.info(f"Query completed with {len(results)} formatted results")
        if results:
            logger.debug(f"Top result score: {results[0].get('similarity', 'N/A')}")
        return results
    except Exception as e:
        # Use the new helper function for error handling
        return handle_search_error(e, "querying Qdrant", query_text)

async def perform_hybrid_search(
    client: QdrantClient,
    collection_name: str,
    query_text: str,
    filter_text: Optional[str] = None,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
    source_filter: Optional[str] = None,
    match_count: int = 5
) -> List[Dict[str, Any]]:
    """
    Perform a hybrid search combining vector similarity with keyword/text-based filtering.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection to search
        query_text: The query text for semantic vector search
        filter_text: Optional keyword text for filtering (text search)
        vector_weight: Weight for vector search results (0.0-1.0)
        keyword_weight: Weight for keyword search results (0.0-1.0)
        source_filter: Optional source domain to filter results
        match_count: Maximum number of results to return
    
    Returns:
        List of dictionaries containing search results with combined score
    """
    if vector_weight + keyword_weight != 1.0:
        # Normalize weights if they don't sum to 1.0
        total = vector_weight + keyword_weight
        if total == 0: # Avoid division by zero if both are zero
            vector_weight = 0.5
            keyword_weight = 0.5
        else:
            vector_weight = vector_weight / total
            keyword_weight = keyword_weight / total
    
    # Create filter conditions using the helper function
    # Create a filter condition dictionary to include both source and text filters
    filter_condition = {}
    if source_filter and source_filter.strip():
        filter_condition["source"] = source_filter
    
    # If we have keyword filter text, add it to the filter condition
    if filter_text and filter_text.strip():
        filter_condition["text"] = {"text": filter_text}
    
    # Use the helper function to create the Qdrant filter
    qdrant_filter = create_qdrant_filter(filter_condition=filter_condition)
    
    try:
        # Get embedding for vector search
        # get_embedding will be imported from embedding_utils
        query_embedding = await get_embedding(query_text)
        if not query_embedding:
            logger.error(f"Could not generate embedding for hybrid search query: '{query_text}'. Returning empty list.")
            return []
        
        # Perform hybrid search
        # If there's no keyword filter, perform standard vector search
        if not filter_text or not filter_text.strip():
            # client.search is synchronous
            search_result = await asyncio.to_thread(
                client.search,
                collection_name=collection_name,
                query_vector=query_embedding,
                query_filter=qdrant_filter, # This would contain only source_filter if filter_text is empty
                limit=match_count,
                with_payload=True
            )
            
            # Use the helper function for result formatting
            results = []
            for hit in search_result:
                # The search type is "vector_only" in this case
                result = format_search_result(hit, search_type="vector_only")
                # Ensure combined_score is present for consistency
                result["combined_score"] = result["similarity"]
                results.append(result)
            return results
        
        # For true hybrid search, use Qdrant's query API which supports combined searches
        # Create a filter without the text condition for vector search
        vector_filter_conditions = {}
        if source_filter and source_filter.strip():
            vector_filter_conditions["source"] = source_filter
        vector_filter = create_qdrant_filter(filter_condition=vector_filter_conditions)
        
        # First get vector search results
        # client.search is synchronous
        vector_results = await asyncio.to_thread(
            client.search,
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=vector_filter,
            limit=match_count * 2,  # Get more results for better combination
            with_payload=True
        )
        
        # Create a filter with only the text condition for text search
        text_filter_conditions = {"text": {"text": filter_text}}
        if source_filter and source_filter.strip(): # also apply source filter to text search part
             text_filter_conditions["source"] = source_filter
        text_filter = create_qdrant_filter(filter_condition=text_filter_conditions) if filter_text else None
        
        # Get text search results
        text_search_hits = []
        if text_filter:
            # client.scroll is synchronous
            text_search_hits_response, _ = await asyncio.to_thread(
                client.scroll, # scroll returns a tuple (results, next_page_offset)
                collection_name=collection_name,
                scroll_filter=text_filter, # scroll_filter instead of filter for scroll API
                limit=match_count * 2, 
                with_payload=True
            )
            text_search_hits = text_search_hits_response
        
        # Combine results
        result_map = {}  # Map of ID to combined result
        
        # Process vector results using the helper function
        for hit in vector_results:
            # Format the result with the helper
            result = format_search_result(hit, search_type="hybrid")
            # Add hybrid-specific scores
            result["vector_score"] = hit.score
            result["keyword_score"] = 0.0 # Default, might be updated if also in text results
            result["combined_score"] = hit.score * vector_weight
            result_map[hit.id] = result
        
        # Process text results and combine with vector results
        for hit in text_search_hits:
            if hit.id in result_map:
                # Update existing result with text score
                result_map[hit.id]["keyword_score"] = 1.0  # Simplified score for text match
                result_map[hit.id]["combined_score"] += 1.0 * keyword_weight
            else:
                # Add new result from text search using the helper
                result = format_search_result(hit, search_type="hybrid", include_score=False) # text search has no inherent score from qdrant like vector search
                # Add hybrid-specific scores
                result["vector_score"] = 0.0
                result["keyword_score"] = 1.0  # Simplified score for text match
                result["combined_score"] = 1.0 * keyword_weight
                result_map[hit.id] = result
        
        # Convert map to list and sort by combined score
        combined_results = list(result_map.values())
        combined_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Return top results
        return combined_results[:match_count]
    
    except Exception as e:
        return handle_search_error(e, "performing hybrid search", query_text)

async def get_available_sources(client: QdrantClient, collection_name: str) -> List[str]:
    """
    Get a list of unique source values from the Qdrant collection.
    """
    sources = set()
    try:
        # Scroll through points with a limit to fetch distinct sources.
        # This can be slow on very large datasets.
        # Consider alternative strategies if performance becomes an issue (e.g., dedicated metadata store or Qdrant aggregations if available).
        next_page_offset = None
        processed_count = 0
        max_scroll_limit = 10000 # Safety limit to prevent excessive scrolling

        while processed_count < max_scroll_limit :
            # client.scroll is synchronous
            response_data, current_offset = await asyncio.to_thread(
                client.scroll,
                collection_name=collection_name, 
                limit=1000, # Adjust batch size as needed
                offset=next_page_offset,
                with_payload=["source"],
                with_vectors=False # No need for vectors here
            )
            if not response_data: # No more points
                break

            for hit in response_data:
                if hit.payload and "source" in hit.payload:
                    sources.add(hit.payload["source"])
            
            processed_count += len(response_data)

            if current_offset is None: # End of scrolling
                break
            next_page_offset = current_offset
        
        if processed_count >= max_scroll_limit:
            logger.warning(f"Reached max scroll limit ({max_scroll_limit}) while fetching sources. List may be incomplete.")

        logger.debug(f"Found sources: {sources}")
        return sorted(list(sources))
    except Exception as e:
        logger.error(f"Error getting available sources from Qdrant: {e}")
        return []

async def get_collection_stats(
    qdrant_client: QdrantClient, 
    collection_name: Optional[str] = None,
    include_segments: bool = False # include_segments is not directly used by qdrant_client.get_collection
) -> Dict[str, Any]:
    """
    Get detailed statistics about a Qdrant collection, or all collections if collection_name is None.
    Includes payload schema, cluster status, and optimizer status.
    """
    results: Dict[str, Any] = {
        "success": False,
        "timestamp": datetime.utcnow().isoformat()
    }

    try:
        # Fetch cluster status (applies to the entire Qdrant instance)
        try:
            # qdrant_client.cluster_status is synchronous
            cluster_status_info = await asyncio.to_thread(qdrant_client.cluster_status)
            results["cluster_status"] = {
                "status": str(cluster_status_info.status),
                "peer_id": cluster_status_info.peer_id,
                "peers": {
                    peer_id: {
                        "uri": peer_info.uri,
                        "state": str(peer_info.state),
                        "is_witness": peer_info.is_witness,
                        "consensus_thread_status": str(peer_info.consensus_thread_status),
                        "message_send_failures": peer_info.message_send_failures,
                        "last_responded": peer_info.last_responded.isoformat() if peer_info.last_responded else None
                    } for peer_id, peer_info in cluster_status_info.peers.items()
                } if cluster_status_info.peers else None,
                "raft_info": {
                    "term": cluster_status_info.raft_info.term,
                    "commit": cluster_status_info.raft_info.commit,
                    "pending_operations": cluster_status_info.raft_info.pending_operations,
                    "leader": cluster_status_info.raft_info.leader,
                    "role": str(cluster_status_info.raft_info.role) if cluster_status_info.raft_info.role else None,
                    "is_voter": cluster_status_info.raft_info.is_voter
                },
                "consensus_thread_status": str(cluster_status_info.consensus_thread_status),
                "message_send_failures": {str(key): val for key, val in cluster_status_info.message_send_failures.items()} if cluster_status_info.message_send_failures else None
            }
        except Exception as e_cluster:
            results["cluster_status"] = {"error": f"Could not retrieve cluster status: {str(e_cluster)}"}

        if collection_name:
            # qdrant_client.get_collection is synchronous
            collection_info = await asyncio.to_thread(qdrant_client.get_collection, collection_name=collection_name)
            
            vector_size = None
            distance_metric = None
            on_disk_payload = None
            named_vectors_params = {}

            # Correctly access vector parameters
            vectors_config = collection_info.config.params.vectors
            if isinstance(vectors_config, dict): # Named vectors
                results["collection_vector_config_type"] = "named_vectors"
                for name, params in vectors_config.items():
                    named_vectors_params[name] = {
                        "size": params.size,
                        "distance": str(params.distance),
                        "on_disk": params.on_disk if hasattr(params, 'on_disk') else None,
                        "hnsw_config": params.hnsw_config.model_dump(mode='json') if hasattr(params.hnsw_config, 'model_dump') else str(params.hnsw_config),
                        "quantization_config": str(params.quantization_config) if params.quantization_config else None,
                        "multivector_config": params.multivector_config.model_dump(mode='json') if hasattr(params.multivector_config, 'model_dump') else str(params.multivector_config)
                    }
                # Try to pick a default or first for top-level display
                if "default" in named_vectors_params:
                    vector_size = named_vectors_params["default"].get("size")
                    distance_metric = named_vectors_params["default"].get("distance")
                    on_disk_payload = named_vectors_params["default"].get("on_disk")
                elif named_vectors_params:
                    first_key = next(iter(named_vectors_params))
                    vector_size = named_vectors_params[first_key].get("size")
                    distance_metric = named_vectors_params[first_key].get("distance")
                    on_disk_payload = named_vectors_params[first_key].get("on_disk")
                else: # Should not happen if vectors_config is a dict and not empty
                    vector_size = "N/A (empty named vectors map)"
                    distance_metric = "N/A (empty named vectors map)"
                    on_disk_payload = "N/A (empty named vectors map)"
            elif hasattr(vectors_config, 'size'): # Single unnamed vector (VectorParams object)
                results["collection_vector_config_type"] = "single_unnamed_vector"
                vector_size = vectors_config.size
                distance_metric = str(vectors_config.distance)
                on_disk_payload = vectors_config.on_disk if hasattr(vectors_config, 'on_disk') else None
                # Store single vector params in a similar structure for consistency if needed later
                named_vectors_params["_default"] = { # Use a conventional key like _default
                    "size": vector_size,
                    "distance": distance_metric,
                    "on_disk": on_disk_payload,
                    "hnsw_config": vectors_config.hnsw_config.model_dump(mode='json') if hasattr(vectors_config.hnsw_config, 'model_dump') else str(vectors_config.hnsw_config),
                    "quantization_config": str(vectors_config.quantization_config) if vectors_config.quantization_config else None,
                    "multivector_config": vectors_config.multivector_config.model_dump(mode='json') if hasattr(vectors_config.multivector_config, 'model_dump') else str(vectors_config.multivector_config)
                }
            else: # Should not happen
                 results["collection_vector_config_type"] = "unknown_format"
                 vector_size = "Unknown vector config format"
                 distance_metric = "Unknown vector config format"
                 on_disk_payload = "Unknown vector config format"

            results["collection"] = {
                "name": collection_name,
                "status": str(collection_info.status), # Convert CollectionStatus to string
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count if hasattr(collection_info, 'indexed_vectors_count') else None,
                "points_count": collection_info.points_count if hasattr(collection_info, 'points_count') else None,
                "segments_count": collection_info.segments_count if hasattr(collection_info, 'segments_count') else None,
                "config": {
                    "params": {
                        "vector_size": vector_size, # Updated
                        "distance": distance_metric, # Updated
                        "shard_number": collection_info.config.params.shard_number,
                        "replication_factor": collection_info.config.params.replication_factor,
                        "write_consistency_factor": collection_info.config.params.write_consistency_factor,
                        "on_disk_payload": on_disk_payload, # Updated
                        "read_fan_out_factor": collection_info.config.params.read_fan_out_factor if hasattr(collection_info.config.params, 'read_fan_out_factor') else None, # Optional
                        "sparse_vectors": collection_info.config.params.sparse_vectors.model_dump(mode='json') if hasattr(collection_info.config.params.sparse_vectors, 'model_dump') else str(collection_info.config.params.sparse_vectors), # Optional
                    },
                    "hnsw_config": collection_info.config.hnsw_config.model_dump(mode='json') if hasattr(collection_info.config.hnsw_config, 'model_dump') else str(collection_info.config.hnsw_config),
                    "optimizer_config": collection_info.config.optimizer_config.model_dump(mode='json') if hasattr(collection_info.config.optimizer_config, 'model_dump') else str(collection_info.config.optimizer_config),
                    "wal_config": collection_info.config.wal_config.model_dump(mode='json') if hasattr(collection_info.config.wal_config, 'model_dump') else str(collection_info.config.wal_config),
                    "quantization_config": str(collection_info.config.quantization_config) if collection_info.config.quantization_config else None
                },
                "payload_schema": {
                    str(key): val.model_dump(mode='json') if hasattr(val, 'model_dump') else str(val) 
                    for key, val in collection_info.payload_schema.items()
                } if collection_info.payload_schema else {},
                "optimizer_status": collection_info.optimizer_status.model_dump(mode='json') if hasattr(collection_info.optimizer_status, 'model_dump') else str(collection_info.optimizer_status)
            }
            if named_vectors_params: # Add the detailed named vector params if they exist
                 results["collection"]["config"]["params"]["named_vectors_params"] = named_vectors_params

            if include_segments:
                # This requires listing segments, which isn't directly on CollectionInfo
                # For now, placeholder or remove if too complex for this util
                results["collection"]["segments"] = "Segments info not implemented in this version of get_collection_stats"
        else:
            # If no collection_name, list all collections
            # qdrant_client.get_collections is synchronous
            collections_response = await asyncio.to_thread(qdrant_client.get_collections)
            collections_list = collections_response.collections
            results["collections_overview"] = []
            for col_desc in collections_list:
                # Fetch full info for each collection to get optimizer status etc.
                # This could be slow if there are many collections.
                # Consider a "light" version if this becomes an issue.
                try:
                    # qdrant_client.get_collection is synchronous
                    detailed_info = await asyncio.to_thread(qdrant_client.get_collection, collection_name=col_desc.name)
                    
                    # Simplified vector info for overview
                    col_vector_size = "N/A"
                    col_distance = "N/A"
                    col_vectors_config = detailed_info.config.params.vectors
                    if isinstance(col_vectors_config, dict):
                        if "default" in col_vectors_config:
                            col_vector_size = col_vectors_config["default"].size
                            col_distance = str(col_vectors_config["default"].distance)
                        elif col_vectors_config:
                             first_vec_params = next(iter(col_vectors_config.values()))
                             col_vector_size = first_vec_params.size
                             col_distance = str(first_vec_params.distance)
                    elif hasattr(col_vectors_config, 'size'):
                         col_vector_size = col_vectors_config.size
                         col_distance = str(col_vectors_config.distance)


                    results["collections_overview"].append({
                        "name": col_desc.name,
                        "status": str(detailed_info.status),
                        "points_count": detailed_info.points_count if hasattr(detailed_info, 'points_count') else None,
                        "vectors_count": detailed_info.vectors_count,
                        "vector_size": col_vector_size, # Simplified
                        "distance": col_distance, # Simplified
                        "optimizer_status_ok": detailed_info.optimizer_status.ok if detailed_info.optimizer_status else None
                    })
                except Exception as e_detail:
                    results["collections_overview"].append({
                        "name": col_desc.name,
                        "error": f"Could not retrieve detailed info: {str(e_detail)}"
                    })
            
        results["success"] = True

    except Exception as e:
        results["success"] = False
        results["error"] = f"Failed to get collection stats for '{collection_name if collection_name else 'all collections'}': {str(e)}"
        results["error_type"] = type(e).__name__
        import traceback
        results["traceback"] = traceback.format_exc()

    return results

async def get_similar_items(
    client: QdrantClient,
    collection_name: str,
    item_id: str,
    filter_condition: Optional[Dict[str, Any]] = None,
    match_count: int = 5
) -> List[Dict[str, Any]]:
    """
    Find similar items based on vector similarity using Qdrant's recommendation API.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection to search
        item_id: ID of the item to find recommendations for
        filter_condition: Optional filter criteria
        match_count: Number of recommendations to return
    
    Returns:
        List of dictionaries containing recommendation results
    """
    try:
        # Use helper function to create filter
        query_filter = create_qdrant_filter(filter_condition=filter_condition)
        
        # Find similar items using the recommend API
        # client.recommend is synchronous
        recommend_result = await asyncio.to_thread(
            client.recommend,
            collection_name=collection_name,
            positive=[item_id],  # ID of the item we want recommendations for
            negative=[],  # Optional IDs to use as negative examples
            query_filter=query_filter,
            limit=match_count,
            with_payload=True,
            with_vectors=False,  # Usually not needed in results
        )
        
        # Process results using helper function
        results = []
        for hit in recommend_result:
            result = format_search_result(hit, search_type="item_recommendation")
            results.append(result)
        
        return results
    
    except Exception as e:
        return handle_search_error(e, "finding similar items", f"item_id={item_id}")

async def fetch_item_by_id(
    client: QdrantClient,
    collection_name: str,
    item_id: str
) -> Optional[Dict[str, Any]]:
    """
    Fetch a specific item by ID from a Qdrant collection.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection
        item_id: ID of the item to fetch
    
    Returns:
        Dictionary containing the item data if found, None otherwise
    """
    try:
        # Retrieve the point by ID
        # Qdrant client's retrieve method expects a list of IDs
        # client.retrieve is synchronous
        points_response = await asyncio.to_thread(
            client.retrieve,
            collection_name=collection_name,
            ids=[item_id],
            with_payload=True,
            with_vectors=False  # Usually not needed
        )
        
        if not points_response: # Check if the list is empty
            logger.warning(f"No item found with ID: {item_id}")
            return None
        
        point = points_response[0] # Get the first item from the list
        # Ensure payload is not None before trying to access attributes
        payload_data = point.payload if point.payload is not None else {}

        result = {
            "id": point.id,
            "url": payload_data.get("url"),
            "content": payload_data.get("text"),
            "metadata": { # Extracted from enhance_payload_metadata for direct use
                "source": payload_data.get("source", "unknown"),
                "crawl_type": payload_data.get("crawl_type", "unknown"),
                "char_count": payload_data.get("char_count", 0),
                "word_count": payload_data.get("word_count", 0),
                "chunk_index": payload_data.get("chunk_index", 0),
                "headers": payload_data.get("headers", "")
            }
        }
        
        return result
    
    except Exception as e:
        logger.error(f"Error fetching item by ID {item_id}: {e}")
        return None

async def find_similar_content(
    client: QdrantClient,
    collection_name: str,
    content_text: str,
    filter_condition: Optional[Dict[str, Any]] = None,
    match_count: int = 5
) -> List[Dict[str, Any]]:
    """
    Find similar content based on text, not an existing item ID.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection to search
        content_text: Text content to find similar items for
        filter_condition: Optional filter criteria
        match_count: Number of recommendations to return
    
    Returns:
        List of dictionaries containing similar content results
    """
    try:
        # First generate an embedding for the content
        # get_embedding will be imported from embedding_utils
        query_embedding = await get_embedding(content_text)
        if not query_embedding:
            logger.error(f"Could not generate embedding for content text. Returning empty list.")
            return []
        
        # Use helper function to create filter
        query_filter = create_qdrant_filter(filter_condition=filter_condition)
        
        # Search for similar content
        # client.search is synchronous
        search_result = await asyncio.to_thread(
            client.search,
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=query_filter,
            limit=match_count,
            with_payload=True
        )
        
        # Process results using helper function
        results = []
        for hit in search_result:
            result = format_search_result(hit, search_type="content_similarity")
            results.append(result)
        
        return results
    
    except Exception as e:
        return handle_search_error(e, "finding similar content", content_text[:50] + "...") 