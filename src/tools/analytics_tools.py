"""
MCP Tools for content clustering and analytics.
"""
import json
from typing import Optional, List, Dict, Any
import os

from mcp.server.fastmcp.exceptions import ToolError

# Import the centralized mcp instance
from src.mcp_setup import mcp
# Import utility functions
from src.utils.qdrant.setup import get_qdrant_client
from src.utils.analytics_utils import (
    fetch_vectors_for_clustering,
    perform_clustering,
    generate_cluster_visualization,
)
# Import logging utilities
from src.utils.logging_utils import get_logger

# Initialize logger
logger = get_logger(__name__)

@mcp.tool()
async def cluster_content(
    source_filter: Optional[str] = None,
    num_clusters: Optional[int] = None,
    sample_size: Optional[int] = None,
    include_visualization: bool = False
) -> str:
    """
    Cluster stored content into semantically similar groups.
    """
    logger.info(f"Starting content clustering (source_filter={source_filter}, num_clusters={num_clusters}, sample_size={sample_size})")
    
    try:
        # Initialize components from environment
        qdrant_client_instance = get_qdrant_client()
        collection_name_str = os.getenv("QDRANT_COLLECTION")
        if not collection_name_str:
            error_msg = "QDRANT_COLLECTION environment variable must be set."
            logger.error(error_msg)
            raise ToolError(error_msg, "CONFIG_ERROR")
    
        # Check for required analytics components early
        try:
            # Try importing necessary dependencies
            import numpy as np
            from sklearn.cluster import KMeans
        except ImportError as e:
            error_msg = f"Required analytics libraries not available: {str(e)}. Install scipy, numpy, and scikit-learn."
            logger.error(error_msg)
            raise ToolError(error_msg, "DEPENDENCY_ERROR", {"missing_libraries": "scipy, numpy, scikit-learn", "original_exception": str(e)})
        
        # Fetch vectors for clustering
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
            raise ToolError(error_msg, "NO_DATA")
            
        logger.info(f"Retrieved {len(vectors_and_metadata)} vectors for clustering")
        
        # Perform clustering
        logger.info("Performing clustering...")
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
            
        logger.info("Clustering complete.")
        return json.dumps(result, indent=2)
        
    except Exception as e:
        error_msg = f"Error during clustering process: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.debug(traceback.format_exc())
        raise ToolError(error_msg, "CLUSTERING_ERROR", {"original_exception": str(e), "traceback": traceback.format_exc()})

# Ensure the file ends with a newline for linters 