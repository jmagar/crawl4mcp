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
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
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

    filter_conditions = []
    if source_filter:
        filter_conditions.append(
            FieldCondition(
                key="source",
                match=MatchValue(value=source_filter)
            )
        )
    
    qdrant_filter = Filter(must=filter_conditions) if filter_conditions else None

    try:
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=qdrant_filter,
            limit=match_count,
            with_payload=True
        )
        
        results = []
        for hit in search_result:
            result = {
                "url": hit.payload.get("url"),
                "content": hit.payload.get("text"), # This will be summary + original if contextual was used
                "original_content": hit.payload.get("original_text", hit.payload.get("text")), # Fallback to text if original_text not present
                "metadata": {
                    "source": hit.payload.get("source"),
                    "crawl_type": hit.payload.get("crawl_type"),
                    "char_count": hit.payload.get("char_count"),
                    "word_count": hit.payload.get("word_count"),
                    "chunk_index": hit.payload.get("chunk_index"),
                    "headers": hit.payload.get("headers", ""),
                    "crawl_time": hit.payload.get("crawl_time", "N/A"), # Add if available
                    "contextual_embedding": hit.payload.get("contextual_embedding", False)
                },
                "similarity": hit.score
            }
            results.append(result)
        return results
    except Exception as e:
        print(f"Error querying Qdrant for '{query_text}': {e}")
        return []

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