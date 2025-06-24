"""
Qdrant administrative utilities (e.g., getting stats, sources).
"""
import os
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import traceback # For get_collection_stats error reporting

from qdrant_client import AsyncQdrantClient
# models might be needed if more detailed stats parsing is done, but not for current functions
# from qdrant_client.http import models 

# Import logging utilities
from src.utils.logging_utils import get_logger # Adjusted import path

# Initialize logger
logger = get_logger(__name__)

async def get_available_sources(client: AsyncQdrantClient, collection_name: str) -> List[str]:
    """
    Get a list of unique source values from the Qdrant collection.
    """
    sources = set()
    try:
        next_page_offset = None
        processed_count = 0
        max_scroll_limit = 10000

        while processed_count < max_scroll_limit:
            response_data, current_offset = await client.scroll(
                collection_name=collection_name, 
                limit=1000,
                offset=next_page_offset,
                with_payload=["source"],
                with_vectors=False
            )
            if not response_data:
                break

            for hit in response_data:
                if hit.payload and "source" in hit.payload:
                    sources.add(hit.payload["source"])
            
            processed_count += len(response_data)

            if current_offset is None:
                break
            next_page_offset = current_offset
        
        if processed_count >= max_scroll_limit:
            logger.warning(f"Reached max scroll limit ({max_scroll_limit}) while fetching sources. List may be incomplete.")

        logger.debug(f"Found sources: {sources}")
        return sorted(list(sources))
    except Exception as e:
        logger.error(f"Error getting available sources from Qdrant: {e}")
        return []

async def get_collection_stats(
    qdrant_client: AsyncQdrantClient, 
    collection_name: Optional[str] = None,
    include_segments: bool = False
) -> Dict[str, Any]:
    """
    Get detailed statistics about a Qdrant collection, or all collections if collection_name is None.
    Includes payload schema, cluster status, and optimizer status.
    """
    results: Dict[str, Any] = {
        "success": False,
        "timestamp": datetime.utcnow().isoformat()
    }

    try:
        try:
            cluster_status_info = await qdrant_client.cluster_status()
            results["cluster_status"] = {
                "status": str(cluster_status_info.status),
                "peer_id": cluster_status_info.peer_id,
                "peers": {
                    peer_id: {
                        "uri": peer_info.uri,
                        "state": str(peer_info.state),
                        "is_witness": peer_info.is_witness,
                        "consensus_thread_status": str(peer_info.consensus_thread_status),
                        "message_send_failures": peer_info.message_send_failures,
                        "last_responded": peer_info.last_responded.isoformat() if peer_info.last_responded else None
                    } for peer_id, peer_info in cluster_status_info.peers.items()
                } if cluster_status_info.peers else None,
                "raft_info": {
                    "term": cluster_status_info.raft_info.term,
                    "commit": cluster_status_info.raft_info.commit,
                    "pending_operations": cluster_status_info.raft_info.pending_operations,
                    "leader": cluster_status_info.raft_info.leader,
                    "role": str(cluster_status_info.raft_info.role) if cluster_status_info.raft_info.role else None,
                    "is_voter": cluster_status_info.raft_info.is_voter
                },
                "consensus_thread_status": str(cluster_status_info.consensus_thread_status),
                "message_send_failures": {str(key): val for key, val in cluster_status_info.message_send_failures.items()} if cluster_status_info.message_send_failures else None
            }
        except Exception as e_cluster:
            results["cluster_status"] = {"error": f"Could not retrieve cluster status: {str(e_cluster)}"}

        if collection_name:
            collection_info = await qdrant_client.get_collection(collection_name=collection_name)
            vector_size, distance_metric, on_disk_payload, named_vectors_params = None, None, None, {}
            vectors_config = collection_info.config.params.vectors

            if isinstance(vectors_config, dict):
                results["collection_vector_config_type"] = "named_vectors"
                for name, params in vectors_config.items():
                    named_vectors_params[name] = {
                        "size": params.size,
                        "distance": str(params.distance),
                        "on_disk": params.on_disk if hasattr(params, 'on_disk') else None,
                        "hnsw_config": params.hnsw_config.model_dump(mode='json') if hasattr(params.hnsw_config, 'model_dump') else str(params.hnsw_config),
                        "quantization_config": str(params.quantization_config) if params.quantization_config else None,
                        "multivector_config": params.multivector_config.model_dump(mode='json') if hasattr(params.multivector_config, 'model_dump') else str(params.multivector_config)
                    }
                default_vec = named_vectors_params.get("default", next(iter(named_vectors_params.values())) if named_vectors_params else {})
                vector_size = default_vec.get("size", "N/A (empty named vectors map)")
                distance_metric = default_vec.get("distance", "N/A (empty named vectors map)")
                on_disk_payload = default_vec.get("on_disk", "N/A (empty named vectors map)")
            elif hasattr(vectors_config, 'size'):
                results["collection_vector_config_type"] = "single_unnamed_vector"
                vector_size, distance_metric, on_disk_payload = vectors_config.size, str(vectors_config.distance), getattr(vectors_config, 'on_disk', None)
                named_vectors_params["_default"] = {
                    "size": vector_size, "distance": distance_metric, "on_disk": on_disk_payload,
                    "hnsw_config": getattr(vectors_config.hnsw_config, 'model_dump', lambda mode: str(vectors_config.hnsw_config))(mode='json'),
                    "quantization_config": str(vectors_config.quantization_config) if vectors_config.quantization_config else None,
                    "multivector_config": getattr(vectors_config.multivector_config, 'model_dump', lambda mode: str(vectors_config.multivector_config))(mode='json')
                }
            else:
                results["collection_vector_config_type"] = "unknown_format"
                vector_size, distance_metric, on_disk_payload = "Unknown", "Unknown", "Unknown"

            results["collection"] = {
                "name": collection_name,
                "status": str(collection_info.status),
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": getattr(collection_info, 'indexed_vectors_count', None),
                "points_count": getattr(collection_info, 'points_count', None),
                "segments_count": getattr(collection_info, 'segments_count', None),
                "config": {
                    "params": {
                        "vector_size": vector_size, "distance": distance_metric,
                        "shard_number": collection_info.config.params.shard_number,
                        "replication_factor": collection_info.config.params.replication_factor,
                        "write_consistency_factor": collection_info.config.params.write_consistency_factor,
                        "on_disk_payload": on_disk_payload,
                        "read_fan_out_factor": getattr(collection_info.config.params, 'read_fan_out_factor', None),
                        "sparse_vectors": getattr(getattr(collection_info.config.params, 'sparse_vectors', None), 'model_dump', lambda mode: str(getattr(collection_info.config.params, 'sparse_vectors', None)))(mode='json'),
                    },
                    "hnsw_config": getattr(collection_info.config.hnsw_config, 'model_dump', lambda mode: str(collection_info.config.hnsw_config))(mode='json'),
                    "optimizer_config": getattr(collection_info.config.optimizer_config, 'model_dump', lambda mode: str(collection_info.config.optimizer_config))(mode='json'),
                    "wal_config": getattr(collection_info.config.wal_config, 'model_dump', lambda mode: str(collection_info.config.wal_config))(mode='json'),
                    "quantization_config": str(collection_info.config.quantization_config) if collection_info.config.quantization_config else None
                },
                "payload_schema": {str(k): getattr(v, 'model_dump', lambda mode: str(v))(mode='json') for k, v in collection_info.payload_schema.items()} if collection_info.payload_schema else {},
                "optimizer_status": getattr(collection_info.optimizer_status, 'model_dump', lambda mode: str(collection_info.optimizer_status))(mode='json')
            }
            if named_vectors_params: results["collection"]["config"]["params"]["named_vectors_params"] = named_vectors_params
            if include_segments: results["collection"]["segments"] = "Segments info not implemented in this version"
        else:
            collections_response = await qdrant_client.get_collections()
            results["collections_overview"] = []
            for col_desc in collections_response.collections:
                try:
                    detailed_info = await qdrant_client.get_collection(collection_name=col_desc.name)
                    col_vectors_config = detailed_info.config.params.vectors
                    col_vector_size, col_distance = "N/A", "N/A"
                    if isinstance(col_vectors_config, dict):
                        default_vec_cfg = col_vectors_config.get("default", next(iter(col_vectors_config.values())) if col_vectors_config else None)
                        if default_vec_cfg: col_vector_size, col_distance = default_vec_cfg.size, str(default_vec_cfg.distance)
                    elif hasattr(col_vectors_config, 'size'):
                        col_vector_size, col_distance = col_vectors_config.size, str(col_vectors_config.distance)
                    results["collections_overview"].append({
                        "name": col_desc.name, "status": str(detailed_info.status),
                        "points_count": getattr(detailed_info, 'points_count', None),
                        "vectors_count": detailed_info.vectors_count,
                        "vector_size": col_vector_size, "distance": col_distance,
                        "optimizer_status_ok": getattr(detailed_info.optimizer_status, 'ok', None)
                    })
                except Exception as e_detail:
                    results["collections_overview"].append({"name": col_desc.name, "error": f"Could not retrieve detailed info: {str(e_detail)}"})
        results["success"] = True
    except Exception as e:
        results["success"] = False
        results["error"] = f"Failed to get collection stats for '{collection_name if collection_name else 'all collections'}': {str(e)}"
        results["error_type"] = type(e).__name__
        results["traceback"] = traceback.format_exc()
    return results 