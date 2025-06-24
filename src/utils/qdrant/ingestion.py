"""
Qdrant data ingestion utilities (e.g., storing embeddings).
"""
import os
import uuid
import asyncio
from urllib.parse import urlparse
from datetime import datetime
from typing import List, Dict, Any, Tuple

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import PointStruct, SparseVector

# Import logging utilities
from src.utils.logging_utils import get_logger
from src.config import settings

# Import embedding functions
from src.utils.embedding_utils import create_embeddings_batch
from src.utils.splade_utils import encode_documents_for_qdrant

# Initialize logger
logger = get_logger(__name__)

async def store_embeddings(
    client: AsyncQdrantClient,
    collection_name: str,
    documents: List[Dict[str, Any]],
) -> Tuple[int, int]:
    """
    Processes a single batch of documents: generates embeddings and upserts them to Qdrant.

    Args:
        client: An initialized AsyncQdrantClient instance.
        collection_name: The name of the Qdrant collection.
        documents: A list of document dictionaries to process.

    Returns:
        A tuple containing (successful_chunks, failed_chunks).
    """
    successful_chunks = 0
    failed_chunks = 0
    batch_size = len(documents)
    
    if batch_size == 0:
        return 0, 0

    texts_to_embed = [doc.get("text", "") for doc in documents]

    # --- 1. Generate Dense Embeddings ---
    logger.info(f"Generating {len(texts_to_embed)} dense embeddings...")
    dense_embeddings = await create_embeddings_batch(texts_to_embed)
    logger.info("Dense embeddings generated.")

    # --- 2. Generate Sparse Embeddings ---
    logger.info(f"Generating {len(texts_to_embed)} sparse vectors (SPLADE)...")
    try:
        sparse_vectors = await asyncio.to_thread(
            encode_documents_for_qdrant,
            texts_to_embed,
            batch_size=settings.SPLADE_BATCH_SIZE
        )
        logger.info("Sparse vectors generated.")
    except Exception as e:
        logger.error(f"Failed to generate sparse vectors: {e}. Proceeding without them.")
        sparse_vectors = []

    # --- 3. Prepare Points for Upsert ---
    logger.info("Preparing points for Qdrant upsert...")
    points_to_upsert = []
    for i, doc_data in enumerate(documents):
        if not dense_embeddings[i]:
            logger.warning(f"Skipping document due to dense embedding failure. Source: {doc_data.get('source_path', 'N/A')}")
            failed_chunks += 1
            continue

        payload = {
            "text": doc_data.get("text"),
            "source": doc_data.get("source"),
            "source_path": doc_data.get("source_path"),
            "url": doc_data.get("url"),
            "crawl_timestamp": datetime.utcnow().isoformat(),
            **doc_data.get("metadata", {})
        }
        
        point_id = doc_data.get("id", str(uuid.uuid4()))

        # RRF collection: use named vectors
        vector_data = {
            "text-dense": dense_embeddings[i],
        }
        if i < len(sparse_vectors):
             vector_data["text-sparse"] = SparseVector(
                indices=sparse_vectors[i]["indices"],
                values=sparse_vectors[i]["values"]
            )

        points_to_upsert.append(
            PointStruct(id=point_id, vector=vector_data, payload=payload)
        )

    # --- 4. Upsert to Qdrant ---
    if not points_to_upsert:
        logger.warning("No points to upsert after processing.")
        return 0, failed_chunks

    logger.info(f"Upserting {len(points_to_upsert)} points to Qdrant collection '{collection_name}'...")
    try:
        await client.upsert(
            collection_name=collection_name,
            points=points_to_upsert,
            wait=True
        )
        successful_chunks = len(points_to_upsert)
        logger.info(f"Successfully upserted {successful_chunks} points.")
    except Exception as e:
        logger.error(f"Error upserting batch to Qdrant: {e}", exc_info=True)
        failed_chunks += len(points_to_upsert)
        return 0, failed_chunks

    return successful_chunks, failed_chunks 