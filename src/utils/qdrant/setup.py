"""
Qdrant client and collection setup utilities.
"""
import os
import asyncio # Ensure asyncio is imported

from qdrant_client import QdrantClient
from qdrant_client.http import models # For models.VectorParams
from qdrant_client.http.exceptions import ResponseHandlingException

# Import logging utilities
from ..logging_utils import get_logger # Adjusted import path
from typing import Optional

# Initialize logger
logger = get_logger(__name__)

def get_qdrant_client(qdrant_url: Optional[str] = None, api_key: Optional[str] = None) -> QdrantClient:
    """
    Get a Qdrant client with the URL and API key from environment variables.
    
    Returns:
        Qdrant client instance
    """
    url_to_use = qdrant_url if qdrant_url is not None else os.getenv("QDRANT_URL")
    api_key_to_use = api_key if api_key is not None else os.getenv("QDRANT_API_KEY")
    
    if not url_to_use:
        logger.error("QDRANT_URL must be provided either as a parameter or set in the environment variables.")
        raise ValueError("QDRANT_URL must be provided either as a parameter or set in the environment variables.")
    
    # Create client with or without API key, depending on what's provided
    if api_key_to_use:
        logger.debug(f"Creating Qdrant client with URL {url_to_use} and API key")
        return QdrantClient(url=url_to_use, api_key=api_key_to_use)
    else:
        logger.debug(f"Creating Qdrant client with URL {url_to_use} (no API key)")
        return QdrantClient(url=url_to_use)

async def ensure_qdrant_collection_async(client: QdrantClient, collection_name: str, vector_dim: int):
    """
    Ensure that the specified Qdrant collection exists, creating it if necessary.

    This function first attempts to get the collection details. If the collection
    is not found (e.g., due to a 404 error), it tries to create it with the
    specified vector dimension and COSINE distance metric.

    Args:
        client: An initialized QdrantClient instance.
        collection_name: The name of the collection to ensure existence of.
        vector_dim: The dimension of the vectors to be stored in the collection.
    """
    try:
        # Try to get collection info. This might raise ResponseHandlingException if the
        # client has trouble parsing the server's response (e.g., Pydantic validation error).
        await asyncio.to_thread(client.get_collection, collection_name=collection_name)
        logger.info(f"Collection '{collection_name}' already exists or client successfully parsed its details.")
    except ResponseHandlingException as rHE:
        logger.warning(f"Error parsing server response for collection '{collection_name}': {rHE}. Assuming collection exists but details are unparsable by this client version.")
        pass
    except Exception as e:
        # A more robust check for a 404 status code from 'e' might be needed if this proves problematic.
        # For now, assuming other exceptions mean it might not exist.
        is_not_found_error = False
        if hasattr(e, 'status_code') and e.status_code == 404:
             is_not_found_error = True
        elif "not found" in str(e).lower() or "could not find" in str(e).lower(): # Heuristic for other not found messages
            is_not_found_error = True

        if is_not_found_error:
            logger.warning(f"Collection '{collection_name}' not found. Attempting to create it.")
        else:
            logger.warning(f"An unexpected error occurred while checking collection '{collection_name}': {type(e).__name__} - {e}. Assuming it might not exist and attempting to create it.")
        
        try:
            await asyncio.to_thread(
                client.create_collection,
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_dim, distance=models.Distance.COSINE)
            )
            logger.info(f"Collection '{collection_name}' created with vector_dim={vector_dim}.")
        except Exception as final_create_e:
            logger.error(f"Attempt to create collection '{collection_name}' also failed: {type(final_create_e).__name__} - {final_create_e}")
            # If creation fails (e.g. 409 Conflict if it actually existed despite initial error, or other reasons),
            # re-raise to indicate a setup problem.
            raise 