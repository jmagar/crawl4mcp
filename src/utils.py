"""
Utility functions for the Crawl4AI MCP server.
"""
import os
import concurrent.futures
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import uuid
from urllib.parse import urlparse
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchText
from qdrant_client.http.exceptions import ResponseHandlingException

# Attempt to import OpenAI and initialize if API key is present (for summarization)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUMMARIZATION_MODEL_CHOICE = os.getenv("SUMMARIZATION_MODEL_CHOICE")
openai = None
if OPENAI_API_KEY and SUMMARIZATION_MODEL_CHOICE:
    try:
        import openai as openai_client
        openai_client.api_key = OPENAI_API_KEY
        openai = openai_client
    except ImportError:
        print("OpenAI library not installed, but OPENAI_API_KEY and SUMMARIZATION_MODEL_CHOICE are set. Summarization will be disabled.")

EMBEDDING_SERVER_URL = os.getenv("EMBEDDING_SERVER_URL")
if not EMBEDDING_SERVER_URL:
    raise ValueError("EMBEDDING_SERVER_URL must be set in the environment variables.")

# New: Configurable batch size for embedding server requests
EMBEDDING_SERVER_BATCH_SIZE = int(os.getenv("EMBEDDING_SERVER_BATCH_SIZE", "32"))

def create_qdrant_filter(source_filter: Optional[str] = None, filter_condition: Optional[Dict[str, Any]] = None) -> Optional[Filter]:
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
    return Filter(must=filter_conditions) if filter_conditions else None

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
    print(f"Error {operation}{query_info}: {error}")
    return []

def get_qdrant_client() -> QdrantClient:
    """
    Get a Qdrant client with the URL and API key from environment variables.
    
    Returns:
        Qdrant client instance
    """
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    
    if not url:
        raise ValueError("QDRANT_URL must be set in the environment variables.")
    
    # Create client with or without API key, depending on what's provided
    if api_key:
        return QdrantClient(url=url, api_key=api_key)
    else:
        return QdrantClient(url=url)

async def get_embedding(text: str) -> List[float]:
    """
    Get an embedding for a single text using the embedding server.
    """
    if not text.strip():
        print("Attempted to get embedding for empty or whitespace text, returning empty list.")
        return []
    try:
        response = requests.post(
            EMBEDDING_SERVER_URL,
            json={"inputs": [text]}
        )
        response.raise_for_status() # Raise an exception for HTTP errors
        embeddings = response.json()
        if isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], list):
            return embeddings[0] # The server returns a list of embeddings, even for a single input
        else:
            print(f"Unexpected embedding format: {embeddings}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error getting embedding from {EMBEDDING_SERVER_URL}: {e}")
        return []
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error decoding JSON response or unexpected structure from {EMBEDDING_SERVER_URL}: {e}")
        return []

def create_embeddings_batch(texts: List[str], server_batch_size: int = 1) -> List[List[float]]:
    """
    Create embeddings for a batch of texts using the self-hosted BGE-large model.
    Sends texts to the embedding server in sub-batches of `server_batch_size`.
    """
    if not texts:
        return []
        
    all_embeddings_results = []
    
    # Filter out empty or whitespace-only strings to prevent errors with the embedding server
    # and keep track of their original indices to reconstruct the final list accurately.
    valid_texts_with_indices = []
    for i, text in enumerate(texts):
        if text and text.strip():
            valid_texts_with_indices.append((i, text))

    if not valid_texts_with_indices:
        return [[] for _ in texts] # All texts were invalid

    original_indices = [item[0] for item in valid_texts_with_indices]
    texts_to_process = [item[1] for item in valid_texts_with_indices]

    for i in range(0, len(texts_to_process), server_batch_size):
        batch_texts = texts_to_process[i:i + server_batch_size]
        if not batch_texts:
            continue
        try:
            response = requests.post(
                EMBEDDING_SERVER_URL,
                json={"inputs": batch_texts}
            )
            response.raise_for_status()
            batch_embeddings = response.json()
            # Ensure the response is a list of lists (embeddings)
            if isinstance(batch_embeddings, list) and all(isinstance(emb, list) for emb in batch_embeddings):
                all_embeddings_results.extend(batch_embeddings)
            else:
                print(f"Unexpected embedding format in batch response: {batch_embeddings}")
                vector_dim_for_fallback = int(os.getenv("VECTOR_DIM", "1024")) # Use VECTOR_DIM, default 1024
                all_embeddings_results.extend([[0.0] * vector_dim_for_fallback for _ in batch_texts]) # Use correct dimension
        except requests.exceptions.RequestException as e:
            print(f"Error creating embeddings for a sub-batch from {EMBEDDING_SERVER_URL}: {e}")
            vector_dim_for_fallback = int(os.getenv("VECTOR_DIM", "1024")) # Use VECTOR_DIM, default 1024
            all_embeddings_results.extend([[0.0] * vector_dim_for_fallback for _ in batch_texts]) # Use correct dimension
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error decoding JSON response or unexpected structure for a sub-batch from {EMBEDDING_SERVER_URL}: {e}")
            vector_dim_for_fallback = int(os.getenv("VECTOR_DIM", "1024")) # Use VECTOR_DIM, default 1024
            all_embeddings_results.extend([[0.0] * vector_dim_for_fallback for _ in batch_texts]) # Use correct dimension

    # Reconstruct the full list, inserting empty embeddings for original empty/failed strings
    final_embeddings = [[] for _ in texts] # Initialize with empty lists
    for i, original_idx in enumerate(original_indices):
        if i < len(all_embeddings_results):
            final_embeddings[original_idx] = all_embeddings_results[i]
        else:
            # This case should ideally not be hit if logic is correct, but as a safeguard:
            print(f"Mismatch between processed embeddings and original texts. Index {original_idx} out of bounds.")

    return final_embeddings

async def generate_contextual_embedding(text_chunk: str, source_url: str, query: str) -> str:
    """
    Generate a contextual embedding for a text chunk using the embedding server.
    """
    if not openai or not SUMMARIZATION_MODEL_CHOICE:
        print("OpenAI client not available or SUMMARIZATION_MODEL_CHOICE not set. Skipping contextual summary.")
        return text_chunk # Return original text if summarization is not configured

    prompt = (
        f"Given the following text chunk from {source_url} and the user query \"{query}\", "
        f"provide a concise summary that helps determine if this chunk is relevant to the query. "
        f"Focus on keywords and entities related to the query. If the chunk is very short or clearly irrelevant, "
        f"you can indicate that. Text chunk:\n\n{text_chunk}"
    )
    try:
        completion = await openai.chat.completions.create(
            model=SUMMARIZATION_MODEL_CHOICE,
            messages=[
                {"role": "system", "content": "You are an expert at creating concise, query-focused summaries for RAG retrieval."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.2
        )
        summary = completion.choices[0].message.content.strip()
        # Combine summary with original text for better retrieval context
        return f"Contextual Summary (Query: {query}): {summary}\n---\nOriginal Text: {text_chunk}"
    except Exception as e:
        print(f"Error generating contextual summary with {SUMMARIZATION_MODEL_CHOICE}: {e}")
        return text_chunk # Fallback to original text on error

async def ensure_qdrant_collection_async(client: QdrantClient, collection_name: str, vector_dim: int):
    """
    Ensure that the specified collection exists in Qdrant.
    This is an asynchronous version.
    """
    try:
        # Try to get collection info. This might raise ResponseHandlingException if the
        # client has trouble parsing the server's response (e.g., Pydantic validation error).
        client.get_collection(collection_name=collection_name)
        print(f"Collection '{collection_name}' already exists or client successfully parsed its details.")
    except ResponseHandlingException as rHE:
        # This is the Pydantic validation error.
        # Assume the collection *exists*, but the client can't parse the response.
        # DO NOT try to create it, as that would likely lead to a 409 Conflict.
        print(f"Error parsing server response for collection '{collection_name}': {rHE}. Assuming collection exists but details are unparsable by this client version.")
        # We pass here, effectively treating the collection as existing.
        pass
    except Exception as e:
        # For other errors (e.g., true 'Not Found' if client.get_collection raises something else,
        # or network issues), assume it might not exist and try to create it.
        # Qdrant client versions vary in how they report "Not Found".
        # A more robust check for a 404 status code from 'e' might be needed if this proves problematic.
        print(f"An unexpected error or 'Not Found' occurred while checking collection '{collection_name}': {e}. Attempting to create it.")
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_dim, distance=models.Distance.COSINE)
            )
            print(f"Collection '{collection_name}' created with vector_dim={vector_dim}.")
        except Exception as final_create_e:
            # If create_collection also fails (e.g., it was a 409 because it did exist,
            # or another network issue), then we log and re-raise.
            print(f"Attempt to create collection '{collection_name}' also failed: {final_create_e}")
            raise

async def store_embeddings(
    client: QdrantClient,
    collection_name: str,
    chunks: List[Dict[str, Any]],
    source_url: str,
    crawl_type: str,
    batch_size: int = 32, # This is Qdrant upsert batch size
    query_for_contextual_embedding: Optional[str] = None
    # embedding_server_batch_size: int = 1 # Parameter removed, will use global EMBEDDING_SERVER_BATCH_SIZE
) -> Tuple[int, int]:
    """
    Store text chunks and their embeddings in Qdrant, handling contextual embeddings.
    Uses the global EMBEDDING_SERVER_BATCH_SIZE for batching requests to the embedding server.
    """
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
        # if query_for_contextual_embedding and openai and SUMMARIZATION_MODEL_CHOICE:
        #     # If contextual embedding is enabled, generate summary and prepend it
        #     contextual_summary = await generate_contextual_embedding(text_content, source_url, query_for_contextual_embedding)
        #     texts_to_embed.append(contextual_summary) # Embed the summary + original text
        # else:
        #     texts_to_embed.append(text_content) # Embed original text only

    # Get all embeddings in a batch
    all_embeddings = create_embeddings_batch(texts_to_embed, server_batch_size=EMBEDDING_SERVER_BATCH_SIZE)

    for i, chunk_data in enumerate(chunks):
        text_content = chunk_data.get("text", "")
        embedding = all_embeddings[i]

        if not embedding:  # Skip if embedding failed for this chunk
            print(f"Skipping chunk {i+1} from {source_url} due to embedding failure.")
            failed_chunks += 1
            continue

        # Use the text_content that was actually embedded (original or with summary)
        embedded_text_payload = texts_to_embed[i]

        payload = {
            "url": source_url,
            "text": embedded_text_payload, # Store the text that was embedded
            # "original_text": text_content, # No longer needed if not summarizing
            "source": urlparse(source_url).netloc,
            "crawl_type": crawl_type,
            "char_count": len(text_content), # char_count of the original text
            "word_count": len(text_content.split()), # word_count of the original text
            "chunk_index": i + 1,
            "headers": chunk_data.get("headers", ""),
            "contextual_embedding": False # Always false now
            # Old logic for contextual_embedding:
            # "contextual_embedding": bool(query_for_contextual_embedding and openai and SUMMARIZATION_MODEL_CHOICE and embedded_text_payload.startswith("Contextual Summary"))
        }

        points_to_upsert.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=payload
            )
        )
        successful_chunks += 1

        if len(points_to_upsert) >= batch_size:
            try:
                client.upsert(collection_name=collection_name, points=points_to_upsert)
                print(f"Upserted {len(points_to_upsert)} points to '{collection_name}'.")
                points_to_upsert = []
            except Exception as e:
                print(f"Error upserting batch to Qdrant: {e}")
                # Decide how to handle batch failures, e.g., mark all in batch as failed
                failed_chunks += len(points_to_upsert)
                successful_chunks -= len(points_to_upsert)
                points_to_upsert = [] 

    if points_to_upsert: # Upsert any remaining points
        try:
            client.upsert(collection_name=collection_name, points=points_to_upsert)
            print(f"Upserted remaining {len(points_to_upsert)} points to '{collection_name}'.")
        except Exception as e:
            print(f"Error upserting final batch to Qdrant: {e}")
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
    query_embedding = await get_embedding(query_text)
    if not query_embedding:
        print(f"Could not generate embedding for query: '{query_text}'. Returning empty list.")
        return []

    # Use the new helper function to create a filter from source_filter
    qdrant_filter = create_qdrant_filter(source_filter=source_filter)

    try:
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=qdrant_filter,
            limit=match_count,
            with_payload=True
        )
        
        # Use the new helper function for result formatting
        results = []
        for hit in search_result:
            result = format_search_result(hit, search_type="semantic")
            results.append(result)
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
        query_embedding = await get_embedding(query_text)
        if not query_embedding:
            print(f"Could not generate embedding for hybrid search query: '{query_text}'. Returning empty list.")
            return []
        
        # Perform hybrid search
        # If there's no keyword filter, perform standard vector search
        if not filter_text or not filter_text.strip():
            search_result = client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                query_filter=qdrant_filter,
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
        vector_filter = create_qdrant_filter(source_filter=source_filter)
        
        # First get vector search results
        vector_results = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=vector_filter,
            limit=match_count * 2,  # Get more results for better combination
            with_payload=True
        )
        
        # Create a filter with only the text condition for text search
        text_filter = create_qdrant_filter(filter_condition={"text": {"text": filter_text}}) if filter_text else None
        
        # Get text search results
        text_results = client.scroll(
            collection_name=collection_name,
            filter=text_filter,
            limit=match_count * 2,  # Get more results for better combination
            with_payload=True
        )[0]  # scroll returns a tuple (results, next_page_offset)
        
        # Combine results
        result_map = {}  # Map of ID to combined result
        
        # Process vector results using the helper function
        for hit in vector_results:
            # Format the result with the helper
            result = format_search_result(hit, search_type="hybrid")
            # Add hybrid-specific scores
            result["vector_score"] = hit.score
            result["keyword_score"] = 0.0
            result["combined_score"] = hit.score * vector_weight
            result_map[hit.id] = result
        
        # Process text results and combine with vector results
        for hit in text_results:
            if hit.id in result_map:
                # Update existing result with text score
                result_map[hit.id]["keyword_score"] = 1.0  # Simplified score for text match
                result_map[hit.id]["combined_score"] += 1.0 * keyword_weight
            else:
                # Add new result from text search using the helper
                result = format_search_result(hit, search_type="hybrid")
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
    # Note: This is a placeholder for how you might implement this.
    # Efficiently getting distinct metadata values in Qdrant can be tricky.
    # One common approach is to scroll through all points, which can be slow.
    # Or maintain a separate list/set of sources if this query is frequent.
    # For now, returning a placeholder.
    # A better approach would be to use Qdrant's scroll API and collect unique sources.
    # However, for simplicity in this example, we'll leave it as a TODO if performance becomes an issue.
    
    # For a more robust solution, consider if your Qdrant client version supports aggregations or specific metadata queries.
    # This is a simplified example that might not be performant on large datasets.
    sources = set()
    try:
        # Scroll through all points with a small limit to fetch distinct sources
        # This is NOT efficient for large datasets. Consider alternative strategies.
        response, next_page_offset = client.scroll(
            collection_name=collection_name, 
            limit=1000, # Adjust as needed, but be mindful of performance
            with_payload=["source"]
        )
        while response:
            for hit in response:
                if hit.payload and "source" in hit.payload:
                    sources.add(hit.payload["source"])
            if next_page_offset is None:
                break
            response, next_page_offset = client.scroll(
                collection_name=collection_name, 
                limit=1000, 
                offset=next_page_offset, 
                with_payload=["source"]
            )
        print(f"Found sources: {sources}")
        return sorted(list(sources))
    except Exception as e:
        print(f"Error getting available sources from Qdrant: {e}")
        return []

async def get_collection_stats(
    client: QdrantClient, 
    collection_name: Optional[str] = None,
    include_segments: bool = False
) -> Dict[str, Any]:
    """
    Get statistics about a Qdrant collection or all collections.
    
    Args:
        client: QdrantClient instance
        collection_name: Optional name of the collection to query 
                        (if None, stats for all collections are returned)
        include_segments: Whether to include segment-level details
        
    Returns:
        Dictionary containing collection statistics
    """
    try:
        # Get collections info
        collections_info = []
        all_collection_names = []
        
        # Get list of all collections if collection_name is not specified
        if collection_name is None:
            try:
                # Get all collections
                all_collections = client.get_collections()
                all_collection_names = [coll.name for coll in all_collections.collections]
            except Exception as e:
                print(f"Error getting all collections: {e}")
                return {
                    "success": False,
                    "error": f"Failed to get collections list: {str(e)}"
                }
        else:
            all_collection_names = [collection_name]
        
        # Get info for each collection
        for coll_name in all_collection_names:
            try:
                # Get collection info
                collection_info = client.get_collection(coll_name)
                
                # Get collection cluster info (shard stats, etc.)
                try:
                    cluster_info = client.collection_cluster_info(coll_name)
                    cluster_data = {
                        "peer_count": len(cluster_info.peer_id_to_shard_count) if hasattr(cluster_info, 'peer_id_to_shard_count') else 0,
                        "shard_count": sum(cluster_info.peer_id_to_shard_count.values()) if hasattr(cluster_info, 'peer_id_to_shard_count') else 0,
                    }
                except Exception as e:
                    # Cluster info may not be available in single-node setups
                    cluster_data = {
                        "peer_count": 1,
                        "shard_count": 1,
                        "note": "Cluster info unavailable or running in single-node mode"
                    }
                
                # Get collection telemetry
                try:
                    telemetry = client.get_collection_telemetry(coll_name)
                    telemetry_data = {
                        "api_call_distributions": telemetry.api_call_distribution,
                        "latency_distributions": telemetry.latency_percentiles
                    } if telemetry else {}
                except Exception as e:
                    telemetry_data = {
                        "note": "Telemetry data unavailable"
                    }
                
                # Get segment info if requested
                segments_data = {}
                if include_segments:
                    try:
                        segments = client.get_collection_shards(coll_name)
                        segments_data = {
                            "segments": segments.shards
                        } if segments else {}
                    except Exception as e:
                        segments_data = {
                            "note": f"Segment data unavailable: {str(e)}"
                        }
                
                # Count points
                count_result = client.count(
                    collection_name=coll_name,
                    exact=True
                )
                
                # Combine info
                collection_data = {
                    "name": coll_name,
                    "status": collection_info.status,
                    "vectors_count": count_result.count,
                    "vectors": collection_info.config.params.vectors,
                    "hnsw_config": collection_info.config.hnsw_config._asdict() if hasattr(collection_info.config, 'hnsw_config') else {},
                    "optimizers_config": collection_info.config.optimizers_config._asdict() if hasattr(collection_info.config, 'optimizers_config') else {},
                    "replication_factor": collection_info.config.params.replication_factor if hasattr(collection_info.config.params, 'replication_factor') else 1,
                    "write_consistency_factor": collection_info.config.params.write_consistency_factor if hasattr(collection_info.config.params, 'write_consistency_factor') else 1,
                    "on_disk_payload": collection_info.config.params.on_disk_payload if hasattr(collection_info.config.params, 'on_disk_payload') else False,
                    "cluster_info": cluster_data,
                    "telemetry": telemetry_data,
                }
                
                # Add segments data if requested
                if include_segments:
                    collection_data["segments"] = segments_data
                
                collections_info.append(collection_data)
                
            except Exception as e:
                # If a specific collection has an error, add error info but continue with others
                print(f"Error getting info for collection '{coll_name}': {e}")
                collections_info.append({
                    "name": coll_name,
                    "error": str(e)
                })
        
        # Calculate summary stats
        total_vectors = sum(coll_info.get("vectors_count", 0) for coll_info in collections_info if "vectors_count" in coll_info)
        
        # Return formatted stats
        return {
            "success": True,
            "timestamp": str(uuid.uuid4()),  # Use as a unique ID for this stats snapshot
            "total_collections": len(collections_info),
            "total_vectors": total_vectors,
            "collections": collections_info
        }
        
    except Exception as e:
        print(f"Error getting collection stats: {e}")
        return {
            "success": False,
            "error": str(e)
        }

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
        recommend_result = client.recommend(
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
        points = client.retrieve(
            collection_name=collection_name,
            ids=[item_id],
            with_payload=True,
            with_vectors=False  # Usually not needed
        )
        
        if not points:
            print(f"No item found with ID: {item_id}")
            return None
        
        point = points[0]
        result = {
            "id": point.id,
            "url": point.payload.get("url"),
            "content": point.payload.get("text"),
            "metadata": {
                "source": point.payload.get("source"),
                "crawl_type": point.payload.get("crawl_type"),
                "char_count": point.payload.get("char_count"),
                "word_count": point.payload.get("word_count"),
                "chunk_index": point.payload.get("chunk_index"),
                "headers": point.payload.get("headers", "")
            }
        }
        
        return result
    
    except Exception as e:
        print(f"Error fetching item by ID: {e}")
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
        query_embedding = await get_embedding(content_text)
        if not query_embedding:
            print(f"Could not generate embedding for content text. Returning empty list.")
            return []
        
        # Use helper function to create filter
        query_filter = create_qdrant_filter(filter_condition=filter_condition)
        
        # Search for similar content
        search_result = client.search(
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

async def fetch_vectors_for_clustering(
    client: QdrantClient,
    collection_name: str,
    filter_condition: Optional[Dict[str, Any]] = None,
    sample_size: int = 1000,
    with_payload_fields: Optional[List[str]] = None
) -> Tuple[List[List[float]], List[Dict[str, Any]], List[str]]:
    """
    Fetch vectors and their payloads from Qdrant for clustering analysis.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection
        filter_condition: Optional filter to apply (e.g., by source)
        sample_size: Maximum number of vectors to fetch
        with_payload_fields: Specific payload fields to include
    
    Returns:
        Tuple containing:
        - List of vectors
        - List of corresponding payloads
        - List of corresponding IDs
    """
    try:
        # Use helper function to create filter
        query_filter = create_qdrant_filter(filter_condition=filter_condition)
        
        # Default payload fields if none specified
        if with_payload_fields is None:
            with_payload_fields = ["url", "text", "source", "crawl_type", "headers"]
        
        # Fetch points using scroll - this is more efficient for getting many points
        points, _ = client.scroll(
            collection_name=collection_name,
            limit=sample_size,
            filter=query_filter,
            with_payload=with_payload_fields,
            with_vectors=True
        )
        
        if not points:
            print(f"No points found in collection '{collection_name}' with the given filter.")
            return [], [], []
        
        # Extract vectors, payloads, and IDs
        vectors = []
        payloads = []
        ids = []
        
        for point in points:
            if point.vector:
                vectors.append(point.vector)
                payloads.append(point.payload)
                ids.append(point.id)
        
        print(f"Fetched {len(vectors)} vectors for clustering from collection '{collection_name}'.")
        return vectors, payloads, ids
    
    except Exception as e:
        print(f"Error fetching vectors for clustering: {e}")
        return [], [], []

async def perform_kmeans_clustering(
    vectors: List[List[float]],
    payloads: List[Dict[str, Any]],
    ids: List[str],
    num_clusters: int = 5,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Perform K-means clustering on a set of vectors.
    
    Args:
        vectors: List of vectors to cluster
        payloads: List of corresponding payloads
        ids: List of corresponding IDs
        num_clusters: Number of clusters to create
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with clustering results
    """
    try:
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except ImportError:
        print("Required libraries (numpy, scikit-learn) not found. Please install with:")
        print("pip install numpy scikit-learn")
        return {
            "success": False,
            "error": "Required libraries not installed. Install with: pip install numpy scikit-learn"
        }
    
    if not vectors:
        return {
            "success": False,
            "error": "No vectors provided for clustering"
        }
    
    try:
        # Convert to numpy array
        X = np.array(vectors)
        
        # Adjust num_clusters if we have fewer points than requested clusters
        if len(X) < num_clusters:
            num_clusters = max(2, len(X) // 2)  # At least 2 clusters if possible
            print(f"Adjusted number of clusters to {num_clusters} based on available data")
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=random_seed, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Calculate silhouette score if we have enough clusters and data points
        silhouette_avg = None
        if num_clusters > 1 and len(X) > num_clusters:
            try:
                silhouette_avg = float(silhouette_score(X, cluster_labels))
            except Exception as e:
                print(f"Error calculating silhouette score: {e}")
        
        # Organize items by cluster
        clusters = {}
        for i in range(num_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            cluster_items = []
            
            for idx in cluster_indices:
                if idx < len(ids) and idx < len(payloads):
                    item = {
                        "id": ids[idx],
                        "payload": payloads[idx],
                        "distance_to_centroid": float(np.linalg.norm(X[idx] - kmeans.cluster_centers_[i]))
                    }
                    cluster_items.append(item)
            
            # Sort items by distance to centroid
            cluster_items.sort(key=lambda x: x["distance_to_centroid"])
            
            # Get representative items (closest to centroid)
            representative_items = cluster_items[:min(5, len(cluster_items))]
            
            # Try to extract common keywords or themes for this cluster
            all_text = " ".join([item["payload"].get("text", "") for item in representative_items])
            cluster_themes = extract_cluster_themes(all_text)
            
            clusters[str(i)] = {
                "size": len(cluster_indices),
                "percentage": float(len(cluster_indices) / len(X) * 100),
                "themes": cluster_themes,
                "representative_items": representative_items
            }
        
        # Return clustering results
        return {
            "success": True,
            "num_clusters": num_clusters,
            "total_points": len(X),
            "silhouette_score": silhouette_avg,
            "clusters": clusters,
            "cluster_distribution": {str(i): int(np.sum(cluster_labels == i)) for i in range(num_clusters)}
        }
    
    except Exception as e:
        print(f"Error performing K-means clustering: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def extract_cluster_themes(text: str, max_themes: int = 5) -> List[str]:
    """
    Extract potential themes or keywords from cluster content.
    
    Args:
        text: Combined text from cluster items
        max_themes: Maximum number of themes to extract
    
    Returns:
        List of potential themes or keywords
    """
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from nltk.corpus import stopwords
        import nltk
        
        # Ensure nltk resources are available
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        # Get stopwords
        stop_words = set(stopwords.words('english'))
        
        # Add additional stopwords relevant to our content
        additional_stops = {'https', 'http', 'www', 'com', 'html', 'the', 'and', 'for', 'with'}
        stop_words.update(additional_stops)
        
        # Use CountVectorizer to extract keywords
        vectorizer = CountVectorizer(
            max_features=max_themes*3,  # Get more words initially for filtering
            stop_words=list(stop_words),
            ngram_range=(1, 2)  # Allow for both single words and bigrams
        )
        
        # Fit the vectorizer and get the top words
        X = vectorizer.fit_transform([text])
        words = vectorizer.get_feature_names_out()
        counts = X.toarray()[0]
        
        # Pair words with their counts and sort
        word_counts = list(zip(words, counts))
        word_counts.sort(key=lambda x: x[1], reverse=True)
        
        # Take the top words
        themes = [word for word, count in word_counts[:max_themes] if count > 1]
        
        return themes
    
    except Exception as e:
        print(f"Error extracting cluster themes: {e}")
        # Return empty list if there's an error
        return []

async def visualize_clusters(
    vectors: List[List[float]], 
    cluster_labels: List[int],
    output_format: str = "plotly_json"
) -> Optional[Dict[str, Any]]:
    """
    Generate a visualization of clusters.
    
    Args:
        vectors: List of vectors
        cluster_labels: List of cluster assignments
        output_format: Format for output ('plotly_json', 'base64_image', etc.)
    
    Returns:
        Dictionary with visualization data or None if error
    """
    try:
        import numpy as np
        from sklearn.manifold import TSNE
        import plotly.graph_objects as go
        import plotly.express as px
        import json
        
        # Convert to numpy arrays
        X = np.array(vectors)
        labels = np.array(cluster_labels)
        
        # Reduce dimensions with t-SNE for visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(X) // 10)))
        X_2d = tsne.fit_transform(X)
        
        # Create plot with plotly
        fig = px.scatter(
            x=X_2d[:, 0], 
            y=X_2d[:, 1],
            color=[str(label) for label in labels],
            title="Vector Clustering Visualization (t-SNE 2D projection)",
            labels={"color": "Cluster"}
        )
        
        # Return in requested format
        if output_format == "plotly_json":
            return {
                "type": "plotly",
                "data": json.loads(fig.to_json())
            }
        else:
            # Default to returning plotly json
            return {
                "type": "plotly",
                "data": json.loads(fig.to_json()),
                "note": f"Requested format '{output_format}' not supported, defaulting to plotly_json"
            }
    
    except ImportError:
        print("Required libraries (numpy, scikit-learn, plotly) not found for visualization.")
        return None
    except Exception as e:
        print(f"Error generating cluster visualization: {e}")
        return None