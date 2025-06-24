"""
Utility functions for generating text embeddings and related operations.
"""
import os
import json
import requests
from typing import List, Optional, Dict, Any

# Import logging utilities
from src.utils.logging_utils import get_logger

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
    Create embeddings for a batch of texts by calling the embedding server.

    Args:
        texts: List of text strings to embed. This should be a single, pre-sized batch.

    Returns:
        List of embeddings (each is a list of floats, or None if embedding failed for that text).
    """
    if not texts:
        logger.warning("Empty texts list provided for batch embedding.")
        return []
        
    embedding_server_url = os.getenv("EMBEDDING_SERVER_URL")
    if not embedding_server_url:
        logger.error("EMBEDDING_SERVER_URL environment variable is not set.")
        return [None] * len(texts)
        
    try:
        # This function now processes the entire list of texts as a single batch.
        # The caller is responsible for batching.
        logger.debug(f"Sending batch of {len(texts)} texts to embedding server.")
        
        response = await asyncio.to_thread(
            requests.post,
            embedding_server_url,
            json={"inputs": texts}
        )
        response.raise_for_status()
        
        response_json = response.json()
        
        # Handle different response formats from TEI server
        if isinstance(response_json, list):
            embeddings_list = response_json
        elif isinstance(response_json, dict) and "embeddings" in response_json:
            embeddings_list = response_json["embeddings"]
        else:
            logger.error(f"Unexpected embedding server response format: {response_json}")
            return [None] * len(texts)

        # Basic validation
        if len(embeddings_list) != len(texts):
            logger.error(
                f"Mismatch between number of texts sent ({len(texts)}) "
                f"and embeddings received ({len(embeddings_list)})."
            )
            return [None] * len(texts)
            
        return embeddings_list
            
    except requests.RequestException as e:
        logger.error(f"Error requesting batch embeddings from server: {e}")
        return [None] * len(texts)
    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"Error parsing batch embedding server response: {e}")
        return [None] * len(texts)
    except Exception as e:
        logger.error(f"Unexpected error in batch embedding creation: {e}", exc_info=True)
        return [None] * len(texts)

# Need to import asyncio if using asyncio.to_thread
import asyncio
import requests

# Need to import asyncio if using asyncio.to_thread
import asyncio 