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
from ..embedding_utils import create_embeddings_batch # generate_contextual_embedding was unused by current store_embeddings

# Initialize logger
logger = get_logger(__name__)

async def store_embeddings(
    client: QdrantClient,
    collection_name: str,
    chunks: List[Dict[str, Any]],
    source_url: str,
    crawl_type: str,
    query_for_contextual_embedding: Optional[str] = None # Parameter kept for signature consistency, though logic is commented out
) -> Tuple[int, int]:
    """
    Store text chunks and their embeddings in Qdrant.
    Uses the global EMBEDDING_SERVER_BATCH_SIZE for batching requests to the embedding server.
    Qdrant upsert batch size is controlled by QDRANT_UPSERT_BATCH_SIZE environment variable.
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
        # Original contextual embedding logic was commented out in qdrant_utils.py already:
        # if query_for_contextual_embedding and openai and SUMMARIZATION_MODEL_CHOICE:
        #     contextual_summary = await generate_contextual_embedding(text_content, source_url, query_for_contextual_embedding)
        #     texts_to_embed.append(contextual_summary)
        # else:
        #     texts_to_embed.append(text_content)

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