"""
Qdrant client and collection setup utilities.
"""
import os
import asyncio # Ensure asyncio is imported

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models # For models.VectorParams
from qdrant_client.http.exceptions import ResponseHandlingException

# Import logging utilities
from src.utils.logging_utils import get_logger # Adjusted import path
from typing import Optional

# Initialize logger
logger = get_logger(__name__)

def get_qdrant_client(qdrant_url: Optional[str] = None, api_key: Optional[str] = None) -> AsyncQdrantClient:
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
        return AsyncQdrantClient(url=url_to_use, api_key=api_key_to_use)
    else:
        logger.debug(f"Creating Qdrant client with URL {url_to_use} (no API key)")
        return AsyncQdrantClient(url=url_to_use)

async def ensure_qdrant_collection_async(client: AsyncQdrantClient, collection_name: str, vector_dim: int):
    """
    Ensure that the specified Qdrant collection exists, creating it if necessary.

    This function first attempts to get the collection details. If the collection
    is not found (e.g., due to a 404 error), it tries to create it with the
    specified vector dimension and COSINE distance metric.

    Args:
        client: An initialized AsyncQdrantClient instance.
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

async def create_rrf_collection(
    client: AsyncQdrantClient,
    collection_name: str,
    dense_vector_size: Optional[int] = None,
    distance: models.Distance = models.Distance.COSINE,
    force_recreate: bool = False
) -> bool:
    """
    Create a new collection with both dense and sparse vector support for RRF.
    
    Args:
        client: AsyncQdrantClient instance
        collection_name: Name of the collection
        dense_vector_size: Size of dense vectors (default 1536 for OpenAI)
        distance: Distance metric for dense vectors
        force_recreate: Whether to recreate if collection exists
        
    Returns:
        True if collection created/updated successfully
    """
    try:
        # Use config vector dimension if none provided
        if dense_vector_size is None:
            from src.config import settings
            dense_vector_size = settings.VECTOR_DIM
        
        # Check if collection exists
        collections_response = await client.get_collections()
        existing_collections = [col.name for col in collections_response.collections]
        
        if collection_name in existing_collections:
            if force_recreate:
                logger.info(f"Recreating existing collection: {collection_name}")
                await client.delete_collection(collection_name)
            else:
                logger.info(f"Collection {collection_name} already exists")
                return True
        
        # Create collection with both dense and sparse vectors
        logger.info(f"Creating RRF collection: {collection_name}")
        
        await client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "text-dense": models.VectorParams(
                    size=dense_vector_size,
                    distance=distance,
                    on_disk=False  # Keep dense vectors in memory for speed
                )
            },
            sparse_vectors_config={
                "text-sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=False  # Keep sparse index in memory for speed
                    )
                )
            },
            # Optimize for search performance
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=10000,  # Lower threshold for faster indexing
                default_segment_number=0,  # Auto-select based on CPU cores
                max_optimization_threads=None  # Use all available threads
            ),
            # HNSW config for dense vectors
            hnsw_config=models.HnswConfigDiff(
                m=16,  # Good balance of accuracy and memory
                ef_construct=200,  # Higher for better accuracy
                full_scan_threshold=10000,
                max_indexing_threads=0  # Auto-select
            )
        )
        
        logger.info(f"Successfully created RRF collection: {collection_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create RRF collection {collection_name}: {e}")
        return False

async def migrate_to_rrf_collection(
    client: AsyncQdrantClient,
    source_collection: str,
    target_collection: str,
    batch_size: int = 100,
    max_points: Optional[int] = None
) -> bool:
    """
    Migrate data from existing collection to new RRF collection with sparse vectors.
    
    Args:
        client: AsyncQdrantClient instance
        source_collection: Name of source collection (dense vectors only)
        target_collection: Name of target RRF collection
        batch_size: Number of points to process per batch
        max_points: Maximum number of points to migrate (None for all)
        
    Returns:
        True if migration completed successfully
    """
    try:
        from src.utils.splade_utils import get_splade_encoder
        
        logger.info(f"Starting migration from {source_collection} to {target_collection}")
        
        # Initialize SPLADE encoder
        splade_encoder = get_splade_encoder()
        
        # Get total point count
        collection_info = await asyncio.to_thread(client.get_collection, source_collection)
        total_points = collection_info.points_count
        
        if max_points:
            total_points = min(total_points, max_points)
        
        logger.info(f"Migrating {total_points} points in batches of {batch_size}")
        
        # Process in batches
        offset = None
        processed = 0
        
        while processed < total_points:
            # Calculate batch size for this iteration
            current_batch_size = min(batch_size, total_points - processed)
            
            # Scroll through source collection
            scroll_result = await asyncio.to_thread(
                client.scroll,
                collection_name=source_collection,
                limit=current_batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )
            
            points = scroll_result[0]
            offset = scroll_result[1]
            
            if not points:
                break
            
            # Process batch
            new_points = []
            texts_to_encode = []
            
            # Extract texts and prepare for sparse encoding
            for point in points:
                # Get content text for sparse encoding
                content = ""
                if point.payload:
                    # Try common content fields
                    content = (
                        point.payload.get("content", "") or
                        point.payload.get("text", "") or
                        point.payload.get("markdown", "") or
                        str(point.payload)
                    )
                
                texts_to_encode.append(content)
            
            # Generate sparse vectors in batch
            logger.debug(f"Generating sparse vectors for batch of {len(texts_to_encode)} texts")
            sparse_vectors = await asyncio.to_thread(
                splade_encoder.encode_documents, 
                texts_to_encode, 
                batch_size=32
            )
            
            # Create new points with both dense and sparse vectors
            for i, point in enumerate(points):
                sparse_vec = sparse_vectors[i]
                
                new_point = models.PointStruct(
                    id=point.id,
                    payload=point.payload,
                    vector={
                        "text-dense": point.vector,  # Existing dense vector
                        "text-sparse": models.SparseVector(
                            indices=sparse_vec["indices"],
                            values=sparse_vec["values"]
                        )
                    }
                )
                new_points.append(new_point)
            
            # Upsert batch to target collection
            await asyncio.to_thread(
                client.upsert,
                collection_name=target_collection,
                points=new_points
            )
            
            processed += len(points)
            logger.info(f"Migrated {processed}/{total_points} points ({processed/total_points*100:.1f}%)")
            
            # Clear memory
            del new_points, sparse_vectors, texts_to_encode
            
            if not offset:  # No more points
                break
        
        logger.info(f"Migration completed: {processed} points migrated to {target_collection}")
        return True
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False 