"""
Qdrant vector field name constants.

This module defines constants for vector field names used across the Qdrant collections
to ensure consistency and avoid magic strings throughout the codebase.
"""

# Dense vector field name used in Qdrant collections
# This corresponds to the embedding vectors stored for semantic search
DENSE_VECTOR_NAME = "text-dense"

# Sparse vector field name used in RRF (Reciprocal Rank Fusion) collections
# This corresponds to the sparse vectors for keyword-based search
SPARSE_VECTOR_NAME = "text-sparse"
