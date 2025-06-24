#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MCP Tools for searching content stored in Qdrant, including specialized code example search.
"""
from typing import Optional, List, Dict, Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest
from sentence_transformers import CrossEncoder # For reranking

from src.mcp_setup import mcp, get_qdrant_client, get_reranker # Import mcp and accessor functions
from src.config import settings
from src.utils.logging_utils import get_logger
from src.utils.embedding_utils import get_embedding # For single query embedding

logger = get_logger(__name__)

@mcp.tool(
    name="search_code_examples",
    description="Searches for relevant code examples extracted from crawled content. Supports filtering and optional reranking."
)
async def search_code_examples(
    query: str,
    limit: int = 10,
    language_filter: Optional[str] = None,
    repo_url_filter: Optional[str] = None,
    # use_hybrid_search: bool = True, # Placeholder for future hybrid search capability
    min_score_threshold: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Searches for relevant code examples stored in Qdrant.

    Args:
        query: The search query string.
        limit: Maximum number of results to return.
        language_filter: Optional. Filter results by programming language (e.g., 'python').
        repo_url_filter: Optional. Filter results by repository URL.
        min_score_threshold: Optional. Minimum similarity score for results from Qdrant.

    Returns:
        A list of dictionaries, where each dictionary represents a found code example
        with its metadata and score.
    """
    # Access resources from the lifespan context stored in app.state
    try:
        # Access resources using the getter functions from mcp_setup
        qdrant_client: Optional[AsyncQdrantClient] = get_qdrant_client()
        reranker: Optional[CrossEncoder] = get_reranker()
    except RuntimeError as e: # Getter functions raise RuntimeError if resource not available
        logger.error(f"Failed to access lifespan context or its attributes via mcp.app.state: {e}")
        logger.error("This might indicate an issue with how FastMCP exposes state or how context is named.")
        # Attempt to access qdrant_client and reranker directly from mcp.app.state if LifespanContext wrapper isn't used as expected
        try:
            qdrant_client = mcp.state.qdrant_client
            reranker = mcp.state.reranking_model # or reranker, depending on how it was set
            logger.info("Successfully accessed qdrant_client and reranker directly from mcp.app.state.")
        except AttributeError:
            logger.error("Failed to access qdrant_client/reranker directly from mcp.app.state. Critical resources unavailable.")
            raise ValueError("Critical resources (Qdrant client, reranker) not available.") from e

    if not qdrant_client:
        logger.error("Qdrant client not available in lifespan context.")
        raise ValueError("Qdrant client not initialized.")

    logger.info(f"Searching for code examples with query: '{query[:50]}...' L:{limit} Lang:{language_filter} Repo:{repo_url_filter}")

    query_embedding = await get_embedding(query)
    if not query_embedding:
        logger.error("Failed to generate embedding for the query.")
        return []

    # Construct Qdrant filters
    qdrant_filters = rest.Filter(
        must=[
            rest.FieldCondition(
                key="crawl_type",
                match=rest.MatchValue(value="code_example")
            )
        ]
    )

    if language_filter:
        qdrant_filters.must.append(
            rest.FieldCondition(key="language", match=rest.MatchValue(value=language_filter.lower()))
        )
    if repo_url_filter:
        # Assuming 'repo_url' is stored in the payload for code examples from repos
        qdrant_filters.must.append(
            rest.FieldCondition(key="repo_url", match=rest.MatchValue(value=repo_url_filter))
        )
    
    # Determine search limit: if reranking, fetch more initially
    search_limit = limit
    if settings.USE_RERANKING and reranker:
        search_limit = max(limit, settings.RERANK_TOP_N) # Fetch at least RERANK_TOP_N or requested limit
        logger.debug(f"Reranking enabled. Initial search limit set to: {search_limit}")

    try:
        search_results = await qdrant_client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=query_embedding,
            query_filter=qdrant_filters,
            limit=search_limit,
            score_threshold=min_score_threshold,
            with_payload=True
        )
    except Exception as e:
        logger.error(f"Error during Qdrant search for code examples: {e}")
        return []

    if not search_results:
        logger.info("No code examples found matching the criteria.")
        return []

    formatted_results = []
    if settings.USE_RERANKING and reranker and search_results:
        logger.info(f"Reranking {len(search_results)} initial results for query: '{query[:50]}...'")
        # Prepare pairs for reranker: [query, document_text]
        rerank_pairs = []
        for hit in search_results:
            # 'text' in payload for code_example is the code itself
            code_text = hit.payload.get("text", "") 
            if code_text: # Ensure there's text to rerank
                rerank_pairs.append([query, code_text])
            else:
                logger.warning(f"Skipping result ID {hit.id} for reranking due to missing 'text' in payload.")

        if not rerank_pairs:
            logger.warning("No valid pairs to rerank after filtering. Returning Qdrant direct results.")
            # Fallback to Qdrant results if reranking pairs are empty
            for hit in search_results[:limit]: # Apply original limit
                formatted_results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                })
            return formatted_results

        try:
            scores = await reranker.predict(rerank_pairs, show_progress_bar=False) # type: ignore
        except Exception as e:
            logger.error(f"Error during reranking prediction: {e}. Returning direct Qdrant results.")
            # Fallback to Qdrant results if reranking fails
            for hit in search_results[:limit]: # Apply original limit
                formatted_results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                })
            return formatted_results

        # Combine reranked scores with original hits
        # Need to align scores with the original search_results, considering some might have been skipped
        scored_hits = []
        score_idx = 0
        for hit in search_results:
            if hit.payload.get("text", ""): # Only consider hits that were part of rerank_pairs
                if score_idx < len(scores):
                    scored_hits.append((scores[score_idx], hit))
                    score_idx += 1
                else:
                    logger.warning("Mismatch in rerank scores and valid hits count. Some results might be missing rerank scores.")
                    break # Avoid index out of bounds
        
        # Sort by new reranked score in descending order
        scored_hits.sort(key=lambda x: x[0], reverse=True)

        for score, hit in scored_hits[:limit]: # Apply original limit
            formatted_results.append({
                "id": hit.id,
                "score": float(score), # Reranked score
                "payload": hit.payload,
                "original_qdrant_score": hit.score # Keep original score for reference
            })
        logger.info(f"Returning {len(formatted_results)} reranked code examples.")

    else: # No reranking or reranker not available
        for hit in search_results: # Already limited by search_limit which is 'limit' if not reranking
            formatted_results.append({
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            })
        logger.info(f"Returning {len(formatted_results)} code examples from Qdrant (no reranking).")

    return formatted_results

# Example of how to register another search tool if needed in the future
# @mcp.tool(
# name="search_general_content",
# description="Searches for general content (excluding code examples) stored in Qdrant."
# )
# async def search_general_content(query: str, limit: int = 5) -> List[Dict[str, Any]]:
# # ... implementation similar to search_code_examples but with different crawl_type filter ...
# pass
