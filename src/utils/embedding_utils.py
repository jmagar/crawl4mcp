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

EMBEDDING_SERVER_URL = os.getenv("EMBEDDING_SERVER_URL")
if not EMBEDDING_SERVER_URL:
    # This would ideally be a fatal error or have a fallback,
    # but for now, we print a warning and allow the module to load.
    # Functions using it will fail.
    logger.critical("EMBEDDING_SERVER_URL must be set in the environment variables.")

# New: Configurable batch size for embedding server requests
EMBEDDING_SERVER_BATCH_SIZE = int(os.getenv("EMBEDDING_SERVER_BATCH_SIZE", "32"))
# VECTOR_DIM (from environment variable, default 1024) is primarily used during Qdrant collection creation 
# (see src/mcp_setup.py and src/utils/qdrant/setup.py). It is not used as a fallback in create_embeddings_batch.
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
            
        response = await asyncio.to_thread(
            requests.post,
            embedding_server_url,
            json={"inputs": text}
        )
        response.raise_for_status()
        embedding_response = response.json()
        
        # TEI server for single input might return: [[0.1, 0.2, ...]]
        # or a dict like {"embedding": [0.1, 0.2, ...]} or {"embeddings": [[0.1, 0.2, ..]]}
        # We need to ensure we return a flat List[float]

        if isinstance(embedding_response, list):
            if len(embedding_response) > 0 and isinstance(embedding_response[0], list):
                # It's likely [[vector]], so return the first vector
                logger.debug(f"Embedding server returned list of lists for single text, taking first element.")
                return embedding_response[0]
            else:
                # It might be a flat list already, or an error/unexpected format
                logger.warning(f"Embedding server returned a list for single text, but not a list of lists. Response: {embedding_response}")
                return None # Or handle as error
        elif isinstance(embedding_response, dict) and "embedding" in embedding_response:
            # Handling for a dict with an "embedding" key that is a flat list
            if isinstance(embedding_response["embedding"], list):
                 logger.debug(f"Embedding server returned dict with flat list at key 'embedding'.")
                 return embedding_response["embedding"]
            else:
                logger.warning(f"Embedding server returned dict with 'embedding' key, but value is not a list. Response: {embedding_response}")
                return None
        elif isinstance(embedding_response, dict) and "embeddings" in embedding_response:
            # Handling for a dict with an "embeddings" key (which should be list of lists)
            if isinstance(embedding_response["embeddings"], list) and len(embedding_response["embeddings"]) > 0 and isinstance(embedding_response["embeddings"][0], list):
                logger.debug(f"Embedding server returned dict with list of lists at key 'embeddings', taking first element.")
                return embedding_response["embeddings"][0]
            else:
                logger.warning(f"Embedding server returned dict with 'embeddings' key, but not a list of lists of vectors. Response: {embedding_response}")
                return None
        else:
            logger.warning(f"Unexpected embedding server response format for single text: {embedding_response}")
            return None
            
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
            response = await asyncio.to_thread(
                requests.post,
                embedding_server_url,
                json={"inputs": batch}
            )
            response.raise_for_status()
            batch_embeddings = response.json()
            logger.debug(f"Successfully received response from embedding server. Type: {type(batch_embeddings)}")
            if isinstance(batch_embeddings, dict):
                logger.debug(f"Embedding server response keys: {list(batch_embeddings.keys())}")
            elif isinstance(batch_embeddings, list) and len(batch_embeddings) > 0:
                logger.debug(f"Embedding server response is a list of {len(batch_embeddings)} items. First item type: {type(batch_embeddings[0])}")
            
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

# Need to import asyncio if using asyncio.to_thread
import asyncio 