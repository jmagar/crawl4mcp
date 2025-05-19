"""
Qdrant data ingestion utilities (e.g., storing embeddings).
"""
import os
import uuid
import asyncio
from urllib.parse import urlparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

# Import logging utilities
from ..logging_utils import get_logger # Adjusted import path

# Import embedding functions
from ..embedding_utils import create_embeddings_batch

# Initialize logger
logger = get_logger(__name__)

async def store_embeddings(
    client: QdrantClient,
    collection_name: str,
    chunks: List[Dict[str, Any]],
    source_url: str,
    crawl_type: str
) -> Tuple[int, int]:
    """
    Store text chunks and their embeddings in Qdrant.
    
    This function processes a list of text chunks, generates embeddings for them in batches,
    and then upserts these points (embedding vector + payload) into the specified Qdrant collection.
    Batch sizes for embedding generation and Qdrant upsert are controlled by environment variables
    EMBEDDING_SERVER_BATCH_SIZE and QDRANT_UPSERT_BATCH_SIZE respectively.

    Args:
        client: An initialized QdrantClient instance.
        collection_name: The name of the Qdrant collection to store embeddings in.
        chunks: A list of dictionaries, where each dictionary represents a chunk of text
                and should contain at least a "text" key. Other keys like "headers" might be used.
        source_url: The original URL from which the chunks were derived.
        crawl_type: A string indicating the type of crawl (e.g., 'single_page', 'sitemap').

    Returns:
        A tuple containing two integers: (successful_chunks, failed_chunks).
    """
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

    texts_to_embed = []
    for i, chunk_data in enumerate(chunks):
        text_content = chunk_data.get("text", "")
        texts_to_embed.append(text_content)

    all_embeddings = await create_embeddings_batch(texts_to_embed)

    for i, chunk_data in enumerate(chunks):
        text_content = chunk_data.get("text", "")
        embedding = all_embeddings[i]

        if not embedding:
            logger.warning(f"Skipping chunk {i+1} from {source_url} due to embedding failure.")
            failed_chunks += 1
            continue

        embedded_text_payload = texts_to_embed[i]

        payload = {
            "url": source_url,
            "text": embedded_text_payload,
            "source": urlparse(source_url).netloc,
            "crawl_type": crawl_type,
            "char_count": len(text_content),
            "word_count": len(text_content.split()),
            "chunk_index": i + 1,
            "headers": chunk_data.get("headers", ""),
            "contextual_embedding": False, # As per current logic in qdrant_utils.py
            "crawl_timestamp": datetime.utcnow().isoformat()
        }

        points_to_upsert.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=payload
            )
        )
        successful_chunks += 1

        if len(points_to_upsert) >= qdrant_upsert_batch_size:
            try:
                await asyncio.to_thread(client.upsert, collection_name=collection_name, points=points_to_upsert)
                logger.info(f"Upserted {len(points_to_upsert)} points to '{collection_name}'.")
                points_to_upsert = []
            except Exception as e:
                logger.error(f"Error upserting batch to Qdrant: {e}")
                failed_chunks += len(points_to_upsert)
                successful_chunks -= len(points_to_upsert)
                points_to_upsert = [] 

    if points_to_upsert:
        try:
            await asyncio.to_thread(client.upsert, collection_name=collection_name, points=points_to_upsert)
            logger.info(f"Upserted remaining {len(points_to_upsert)} points to '{collection_name}'.")
        except Exception as e:
            logger.error(f"Error upserting final batch to Qdrant: {e}")
            failed_chunks += len(points_to_upsert)
            successful_chunks -= len(points_to_upsert)

    return successful_chunks, failed_chunks 