"""
MCP Tools for content clustering and analytics.
"""
import json
from typing import Optional, List, Dict, Any
import os

from mcp.server.fastmcp import Context # MCP Context for tool arguments
from mcp.server.fastmcp.exceptions import ToolError # Correct import path

# Import the centralized mcp instance
from ..mcp_setup import mcp
# Import utility functions
from ..utils.qdrant_utils import get_qdrant_client
from ..utils.analytics_utils import (
    fetch_vectors_for_clustering,
    perform_clustering,
    generate_cluster_visualization,
    # Removed: perform_kmeans_clustering,
    # Removed: visualize_clusters
)
# Import logging utilities
from ..utils.logging_utils import get_logger

# Initialize logger
logger = get_logger(__name__)

@mcp.tool()
async def cluster_content(
    ctx: Context,
    source_filter: Optional[str] = None,
    num_clusters: Optional[int] = None,
    sample_size: Optional[int] = None,
    include_visualization: bool = False
) -> str:
    """
    Cluster stored content into semantically similar groups.
    """
    logger.info(f"Starting content clustering (source_filter={source_filter}, num_clusters={num_clusters}, sample_size={sample_size})")
    
    # Try to get instances from context
    qdrant_client_instance = None
    collection_name_str = None
    
    try:
        if hasattr(ctx, 'request_context') and ctx.request_context is not None:
            qdrant_client_instance = ctx.request_context.lifespan_context.qdrant_client
            collection_name_str = ctx.request_context.lifespan_context.collection_name
        else:
            raise AttributeError("request_context not available on ctx object for cluster_content")
            
    except (AttributeError, ValueError) as e: # Catch both expected errors
        logger.warning(f"Context access failed for cluster_content ({type(e).__name__}: {e}). Initializing Qdrant from environment.")
        try:
            qdrant_client_instance = get_qdrant_client()
            collection_name_str = os.getenv("QDRANT_COLLECTION")
            if not collection_name_str:
                error_msg = "QDRANT_COLLECTION environment variable must be set when context is not available."
                logger.error(error_msg)
                raise ToolError(message=error_msg, code="CONFIG_ERROR")
        except Exception as e_init:
            error_msg = f"Failed to initialize Qdrant components: {str(e_init)}"
            logger.error(error_msg)
            raise ToolError(message=error_msg, code="INITIALIZATION_ERROR", details={"original_exception": str(e_init)})
    
    # Check for required analytics components early
    try:
        # Try importing necessary dependencies
        import numpy as np
        from sklearn.cluster import KMeans
    except ImportError as e:
        error_msg = f"Required analytics libraries not available: {str(e)}. Install scipy, numpy, and scikit-learn."
        logger.error(error_msg)
        raise ToolError(message=error_msg, code="DEPENDENCY_ERROR", details={"missing_libraries": "scipy, numpy, scikit-learn", "original_exception": str(e)})
    
    # Fetch vectors for clustering
    try:
        ctx.log.info("Step 1/3: Fetching vectors for clustering analysis...")
        ctx.report_progress(1, 3, "Fetching vectors")
        logger.info("Fetching vectors for clustering analysis...")
        vectors_and_metadata = await fetch_vectors_for_clustering(
            client=qdrant_client_instance,
            collection_name=collection_name_str,
            source_filter=source_filter,
            limit=sample_size if sample_size else 500  # Default sample size
        )
        
        if not vectors_and_metadata or len(vectors_and_metadata) == 0:
            error_msg = f"No vectors retrieved for clustering. Check if collection has data" + (f" for source '{source_filter}'" if source_filter else ".")
            logger.warning(error_msg)
            raise ToolError(message=error_msg, code="NO_DATA")
            
        logger.info(f"Retrieved {len(vectors_and_metadata)} vectors for clustering")
        ctx.log.info(f"Retrieved {len(vectors_and_metadata)} vectors.")
        
        # Perform clustering
        ctx.log.info("Step 2/3: Performing clustering...")
        ctx.report_progress(2, 3, "Performing clustering")
        clustering_results = await perform_clustering(
            vectors_and_metadata=vectors_and_metadata,
            num_clusters=num_clusters,  # If None, perform_clustering will determine optimal
            include_visualization=include_visualization
        )
        
        # Format results for return
        result = {
            "success": True,
            "total_vectors": len(vectors_and_metadata),
            "clusters": clustering_results["clusters"],
            "cluster_count": len(clustering_results["clusters"]),
            "source_filter": source_filter if source_filter else "all sources"
        }
        
        # Include visualization data if requested and available
        if include_visualization and "visualization" in clustering_results:
            logger.info("Including visualization data in response")
            result["visualization"] = clustering_results["visualization"]
            
        ctx.report_progress(3, 3, "Clustering complete")
        ctx.log.info("Step 3/3: Clustering complete.")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_msg = f"Error during clustering process: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.debug(traceback.format_exc())
        raise ToolError(message=error_msg, code="CLUSTERING_ERROR", details={"original_exception": str(e), "traceback": traceback.format_exc()})

# Ensure the file ends with a newline for linters 