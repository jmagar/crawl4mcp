"""
Utility functions for generating text embeddings and related operations.
"""
import os
import json
import requests
from typing import List, Optional, Dict, Any

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
    # This would ideally be a fatal error or have a fallback,
    # but for now, we print a warning and allow the module to load.
    # Functions using it will fail.
    print("CRITICAL WARNING: EMBEDDING_SERVER_URL must be set in the environment variables.")

# New: Configurable batch size for embedding server requests
EMBEDDING_SERVER_BATCH_SIZE = int(os.getenv("EMBEDDING_SERVER_BATCH_SIZE", "32"))
# VECTOR_DIM is used as a fallback in create_embeddings_batch, so define it here or ensure it's globally available.
# For module independence, let's define it here based on environment variable.
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1024"))


async def get_embedding(text: str) -> List[float]:
    """
    Get an embedding for a single text using the embedding server.
    """
    if not EMBEDDING_SERVER_URL:
        print("Error: EMBEDDING_SERVER_URL not configured.")
        return []
    if not text.strip():
        print("Attempted to get embedding for empty or whitespace text, returning empty list.")
        return []
    try:
        # Using await with requests is not standard. Assuming this should be a synchronous call
        # or needs an async HTTP client like aiohttp if it's meant to be non-blocking.
        # For now, sticking to the original utils.py structure which used synchronous requests.
        # If this util is called from an async function, it will block.
        # To make it truly async, use an async http library.
        # For simplicity in refactoring the existing synchronous logic:
        response = await asyncio.to_thread(
            requests.post,
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

async def create_embeddings_batch(texts: List[str], server_batch_size: int = None) -> List[List[float]]:
    """
    Create embeddings for a batch of texts using the self-hosted BGE-large model.
    Sends texts to the embedding server in sub-batches of `server_batch_size`.
    """
    if not EMBEDDING_SERVER_URL:
        print("Error: EMBEDDING_SERVER_URL not configured.")
        return [[] for _ in texts] # Return list of empty lists matching input structure

    if server_batch_size is None:
        server_batch_size = EMBEDDING_SERVER_BATCH_SIZE

    if not texts:
        return []
        
    all_embeddings_results = []
    
    valid_texts_with_indices = []
    for i, text in enumerate(texts):
        if text and text.strip():
            valid_texts_with_indices.append((i, text))

    if not valid_texts_with_indices:
        return [[] for _ in texts] 

    original_indices = [item[0] for item in valid_texts_with_indices]
    texts_to_process = [item[1] for item in valid_texts_with_indices]

    for i in range(0, len(texts_to_process), server_batch_size):
        batch_texts = texts_to_process[i:i + server_batch_size]
        if not batch_texts:
            continue
        try:
            # Similar to get_embedding, using asyncio.to_thread for synchronous requests
            response = await asyncio.to_thread(
                requests.post,
                EMBEDDING_SERVER_URL,
                json={"inputs": batch_texts}
            )
            response.raise_for_status()
            batch_embeddings = response.json()
            if isinstance(batch_embeddings, list) and all(isinstance(emb, list) for emb in batch_embeddings):
                all_embeddings_results.extend(batch_embeddings)
            else:
                print(f"Unexpected embedding format in batch response: {batch_embeddings}")
                all_embeddings_results.extend([[0.0] * VECTOR_DIM for _ in batch_texts])
        except requests.exceptions.RequestException as e:
            print(f"Error creating embeddings for a sub-batch from {EMBEDDING_SERVER_URL}: {e}")
            all_embeddings_results.extend([[0.0] * VECTOR_DIM for _ in batch_texts])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error decoding JSON response or unexpected structure for a sub-batch from {EMBEDDING_SERVER_URL}: {e}")
            all_embeddings_results.extend([[0.0] * VECTOR_DIM for _ in batch_texts])

    final_embeddings = [[] for _ in texts] 
    for i, original_idx in enumerate(original_indices):
        if i < len(all_embeddings_results):
            final_embeddings[original_idx] = all_embeddings_results[i]
        else:
            print(f"Mismatch between processed embeddings and original texts. Index {original_idx} out of bounds for results.")
            # Fallback to empty list of correct dimension if something went wrong
            final_embeddings[original_idx] = [0.0] * VECTOR_DIM if VECTOR_DIM > 0 else []


    return final_embeddings

async def generate_contextual_embedding(text_chunk: str, source_url: str, query: str) -> str:
    """
    Generate a contextual embedding for a text chunk using the embedding server.
    NOTE: This function creates a *summary* intended for later embedding, not the embedding itself.
    The name might be slightly misleading; it generates text *for* contextual embedding.
    """
    if not openai or not SUMMARIZATION_MODEL_CHOICE:
        # print("OpenAI client not available or SUMMARIZATION_MODEL_CHOICE not set. Skipping contextual summary.")
        return text_chunk # Return original text if summarization is not configured

    prompt = (
        f"Given the following text chunk from {source_url} and the user query \"{query}\", "
        f"provide a concise summary that helps determine if this chunk is relevant to the query. "
        f"Focus on keywords and entities related to the query. If the chunk is very short or clearly irrelevant, "
        f"you can indicate that. Text chunk:\n\n{text_chunk}"
    )
    try:
        # Assuming openai client is already async or we run this in a thread
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

# Need to import asyncio if using asyncio.to_thread
import asyncio 