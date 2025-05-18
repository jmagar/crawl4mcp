"""
Utility functions for generating text embeddings and related operations.
"""
import os
import json
import requests
from typing import List, Optional, Dict, Any

# Import logging utilities
from .logging_utils import get_logger

# Initialize logger
logger = get_logger(__name__)

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
        logger.warning("OpenAI library not installed, but OPENAI_API_KEY and SUMMARIZATION_MODEL_CHOICE are set. Summarization will be disabled.")

EMBEDDING_SERVER_URL = os.getenv("EMBEDDING_SERVER_URL")
if not EMBEDDING_SERVER_URL:
    # This would ideally be a fatal error or have a fallback,
    # but for now, we print a warning and allow the module to load.
    # Functions using it will fail.
    logger.critical("EMBEDDING_SERVER_URL must be set in the environment variables.")

# New: Configurable batch size for embedding server requests
EMBEDDING_SERVER_BATCH_SIZE = int(os.getenv("EMBEDDING_SERVER_BATCH_SIZE", "32"))
# VECTOR_DIM is used as a fallback in create_embeddings_batch, so define it here or ensure it's globally available.
# For module independence, let's define it here based on environment variable.
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1024"))


async def get_embedding(text: str) -> Optional[List[float]]:
    """
    Get an embedding for a single text string.
    
    Args:
        text: Text to embed
        
    Returns:
        List of floats representing the embedding, or None if embedding fails
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for embedding. Returning None.")
        return None

    try:
        # This implementation directly uses the embedding server.
        # For batched processing, use create_embeddings_batch instead.
        embedding_server_url = os.getenv("EMBEDDING_SERVER_URL")
        if not embedding_server_url:
            logger.error("EMBEDDING_SERVER_URL environment variable is not set.")
            return None
            
        response = requests.post(
            embedding_server_url,
            json={"text": text}
        )
        response.raise_for_status()
        embedding_response = response.json()
        
        # The actual key might differ based on your embedding server's output format
        if "embedding" in embedding_response:
            return embedding_response["embedding"]
        else:
            # If the embedding is the direct response (depends on your server API design)
            return embedding_response
            
    except requests.RequestException as e:
        logger.error(f"Error requesting embedding from server: {e}")
        return None
    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"Error parsing embedding server response: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in get_embedding: {e}")
        return None

async def create_embeddings_batch(texts: List[str]) -> List[Optional[List[float]]]:
    """
    Create embeddings for a batch of texts.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List of embeddings (each is a list of floats, or None if embedding failed for that text)
    """
    if not texts:
        logger.warning("Empty texts list provided for batch embedding.")
        return []
        
    # Filter out empty texts
    filtered_texts = [text for text in texts if text and text.strip()]
    if len(filtered_texts) < len(texts):
        logger.warning(f"Filtered out {len(texts) - len(filtered_texts)} empty texts from batch.")
    
    if not filtered_texts:
        logger.warning("No valid texts left after filtering empty strings.")
        return [None] * len(texts)  # Return proper length list of None
        
    batch_size = os.getenv("EMBEDDING_SERVER_BATCH_SIZE")
    try:
        batch_size = int(batch_size) if batch_size else 16
        if batch_size <= 0:
            logger.warning("EMBEDDING_SERVER_BATCH_SIZE must be positive. Using default 16.")
            batch_size = 16
    except ValueError:
        logger.warning(f"Invalid EMBEDDING_SERVER_BATCH_SIZE: {batch_size}. Using default 16.")
        batch_size = 16
        
    embedding_server_url = os.getenv("EMBEDDING_SERVER_URL")
    if not embedding_server_url:
        logger.error("EMBEDDING_SERVER_URL environment variable is not set.")
        return [None] * len(texts)
        
    # Create a mapping of filtered text positions to original positions
    filtered_to_original = {}
    original_to_filtered = {}
    filtered_texts_list = []
    
    for i, text in enumerate(texts):
        if text and text.strip():
            filtered_to_original[len(filtered_texts_list)] = i
            original_to_filtered[i] = len(filtered_texts_list)
            filtered_texts_list.append(text)
            
    # Process in batches
    all_embeddings = [None] * len(texts)  # Initialize with None for all texts
    
    for i in range(0, len(filtered_texts_list), batch_size):
        batch = filtered_texts_list[i:i+batch_size]
        logger.debug(f"Processing embedding batch {i//batch_size + 1}/{(len(filtered_texts_list) + batch_size - 1)//batch_size}")
        
        try:
            response = requests.post(
                embedding_server_url,
                json={"texts": batch}
            )
            response.raise_for_status()
            batch_embeddings = response.json()
            
            # The actual structure might differ based on your embedding server's output format
            if isinstance(batch_embeddings, list):
                # If the server returns a list of embeddings directly
                embeddings_list = batch_embeddings
            else:
                # If the server returns a structured response with embeddings inside
                embeddings_list = batch_embeddings.get("embeddings", [])
                
            # Map the embeddings back to the original positions
            for j, embedding in enumerate(embeddings_list):
                original_idx = filtered_to_original.get(i + j)
                if original_idx is not None:
                    all_embeddings[original_idx] = embedding
                    
        except requests.RequestException as e:
            logger.error(f"Error requesting batch embeddings from server: {e}")
            # Leave as None for this batch
        except (ValueError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing batch embedding server response: {e}")
            # Leave as None for this batch
        except Exception as e:
            logger.error(f"Unexpected error in batch embedding creation: {e}")
            # Leave as None for this batch
            
    return all_embeddings

async def generate_contextual_embedding(text_chunk: str, source_url: str, query: str) -> str:
    """
    Generate a contextual embedding for a text chunk using the embedding server.
    NOTE: This function creates a *summary* intended for later embedding, not the embedding itself.
    The name might be slightly misleading; it generates text *for* contextual embedding.
    """
    if not openai or not SUMMARIZATION_MODEL_CHOICE:
        # logger.debug("OpenAI client not available or SUMMARIZATION_MODEL_CHOICE not set. Skipping contextual summary.")
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
        logger.error(f"Error generating contextual summary with {SUMMARIZATION_MODEL_CHOICE}: {e}")
        return text_chunk # Fallback to original text on error

# Need to import asyncio if using asyncio.to_thread
import asyncio 