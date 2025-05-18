"""
MCP Tools for content clustering and analytics.
"""
import json
from typing import Optional, List, Dict, Any
import os

from mcp.server.fastmcp import Context # MCP Context for tool arguments

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
    qdrant_client_instance = None
    collection_name_str = None

    try:
        if hasattr(ctx, 'request_context') and ctx.request_context is not None:
            qdrant_client_instance = ctx.request_context.lifespan_context.qdrant_client
            collection_name_str = ctx.request_context.lifespan_context.collection_name
        else:
            raise AttributeError("request_context not available for cluster_content")
    except (AttributeError, ValueError) as e:
        print(f"Context access failed for cluster_content ({type(e).__name__}: {e}). Initializing Qdrant from environment.")
        try:
            qdrant_client_instance = get_qdrant_client()
            collection_name_str = os.getenv("QDRANT_COLLECTION")
            if not collection_name_str:
                raise ValueError("QDRANT_COLLECTION environment variable must be set.")
        except Exception as e_init:
            return json.dumps({"success": False, "error": f"Failed to initialize Qdrant: {str(e_init)}"}, indent=2)

    if not all([qdrant_client_instance, collection_name_str]):
        return json.dumps({"success": False, "error": "Qdrant client or collection name missing for cluster_content."}, indent=2)

    try:
        # Fetch vectors for clustering
        # The utility function expects filter_condition as a dict.
        # If source_filter is provided, wrap it in the expected dict structure.
        fetch_filter_condition = None
        if source_filter:
            fetch_filter_condition = {"source": source_filter}

        vectors_data = await fetch_vectors_for_clustering(
            client=qdrant_client_instance,
            collection_name=collection_name_str,
            filter_condition=fetch_filter_condition,
            sample_size=sample_size,
            with_payload_fields=["url", "text", "source"]
        )
        
        if not vectors_data or not vectors_data["vectors"]:
            return json.dumps({"success": False, "error": "No vectors found for clustering with the given filters.", "details": vectors_data}, indent=2)

        # 2. Perform clustering
        # num_clusters will be calculated if None by perform_clustering
        cluster_results = await perform_clustering(vectors_data["vectors"], num_clusters=num_clusters) 

        # Check if perform_clustering itself was successful
        if not cluster_results.get("success"): # Check the success flag from perform_clustering
            error_message = cluster_results.get("error", "Clustering sub-process failed without specific error.")
            # Include num_clusters from the error response if available, for context
            attempted_num_clusters = cluster_results.get("num_clusters", "N/A") 
            return json.dumps({
                "success": False, 
                "error": f"Clustering failed: {error_message}",
                "details": {
                    "source_filter": source_filter,
                    "sample_size_used": vectors_data.get("actual_sample_size"),
                    "attempted_num_clusters_by_util": attempted_num_clusters
                }
            }, indent=2)

        # 3. Prepare results
        output: Dict[str, Any] = {
            "success": True,
            "source_filter": source_filter,
            "sample_size_used": vectors_data["actual_sample_size"],
            "total_vectors_in_source": vectors_data["total_vectors_in_source"],
            "num_clusters": cluster_results["num_clusters"],
            "cluster_sizes": cluster_results["cluster_sizes"],
            # Optionally add item_ids per cluster if needed, though it might be large
            # "clusters": cluster_results["clusters_with_ids"] 
        }

        # 4. Generate visualization if requested
        if include_visualization:
            if vectors_data["actual_sample_size"] < 2:
                output["visualization_error"] = "Not enough data points for visualization (need at least 2)."
            else:
                try:
                    visualization_html = await generate_cluster_visualization(vectors_data["vectors"], cluster_results["labels"])
                    # For simplicity, we might not embed HTML directly in JSON response, 
                    # but indicate availability or save it and return a path/URL.
                    # Here, we'll just confirm it was generated.
                    output["visualization_generated"] = True
                    # If you want to include the HTML (can be large):
                    # output["visualization_html"] = visualization_html
                except Exception as viz_e:
                    output["visualization_error"] = f"Failed to generate visualization: {str(viz_e)}"

        return json.dumps(output, indent=2)

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)

# Ensure the file ends with a newline for linters 