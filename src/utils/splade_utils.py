"""
SPLADE sparse vector encoding utilities using FastEmbed.
FastEmbed provides optimized ONNX models that are smaller and faster than raw transformers.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from fastembed import SparseTextEmbedding, SparseEmbedding
import os

logger = logging.getLogger(__name__)

def _check_intel_gpu_support():
    """Check if Intel GPU support is available."""
    try:
        import intel_extension_for_pytorch as ipex
        
        # Check if Intel GPU is available
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            logger.info(f"Intel GPU detected: {torch.xpu.get_device_name()}")
            return True
        else:
            logger.info("Intel GPU not available, falling back to CPU")
            return False
    except ImportError:
        logger.info("Intel Extension for PyTorch not available, using CPU")
        return False
    except Exception as e:
        logger.warning(f"Error checking Intel GPU support: {e}")
        return False

class SPLADEEncoder:
    """SPLADE encoder using FastEmbed's optimized models."""
    
    def __init__(self, model_name: str = "prithivida/Splade_PP_en_v1"):
        """
        Initialize SPLADE encoder with FastEmbed.
        
        Args:
            model_name: FastEmbed SPLADE model name. Default is the recommended SPLADE++ model.
        """
        self.model_name = model_name
        self.model: Optional[SparseTextEmbedding] = None
        self.intel_gpu_available = _check_intel_gpu_support()
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the FastEmbed SPLADE model."""
        try:
            logger.info(f"Initializing FastEmbed SPLADE model: {self.model_name}")
            
            # Detect GPU availability and set device preference for FastEmbed
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"SPLADE device selected: {device}")
                logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            elif self.intel_gpu_available:
                device = "cpu"  # FastEmbed doesn't support Intel GPU directly yet
                logger.info("Intel GPU available but FastEmbed uses CPU - considering custom implementation")
            else:
                device = "cpu"
                logger.info("SPLADE device selected: cpu (no GPU available)")
            
            # Initialize with device preference (FastEmbed will use CUDA if available)
            self.model = SparseTextEmbedding(model_name=self.model_name)
            logger.info(f"FastEmbed SPLADE model initialized successfully on {device}")
        except Exception as e:
            logger.error(f"Failed to initialize FastEmbed SPLADE model: {e}")
            raise
    
    def encode_documents(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        Encode documents to sparse vectors.
        
        Args:
            texts: List of documents to encode
            batch_size: Batch size for processing
            
        Returns:
            List of sparse vector dictionaries with 'indices' and 'values'
        """
        if not self.model:
            raise RuntimeError("SPLADE model not initialized")
        
        try:
            # Use smaller batch sizes for better memory management
            effective_batch_size = min(batch_size, 16) if self.intel_gpu_available else batch_size
            logger.info(f"Encoding {len(texts)} documents with batch_size={effective_batch_size}")
            
            # Generate sparse embeddings
            sparse_embeddings: List[SparseEmbedding] = list(
                self.model.embed(texts, batch_size=effective_batch_size)
            )
            
            # Convert to our expected format
            results = []
            for embedding in sparse_embeddings:
                sparse_vector = {
                    'indices': embedding.indices.tolist(),
                    'values': embedding.values.tolist()
                }
                results.append(sparse_vector)
            
            logger.info(f"Successfully encoded {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"Error encoding documents: {e}")
            raise
    
    def encode_query(self, query: str) -> Dict[str, Any]:
        """
        Encode a single query to sparse vector.
        
        Args:
            query: Query text to encode
            
        Returns:
            Sparse vector dictionary with 'indices' and 'values'
        """
        results = self.encode_documents([query], batch_size=1)
        return results[0] if results else {'indices': [], 'values': []}
    
    @staticmethod
    def list_available_models() -> List[Dict[str, Any]]:
        """
        List all available FastEmbed SPLADE models.
        
        Returns:
            List of model information dictionaries
        """
        try:
            return SparseTextEmbedding.list_supported_models()
        except Exception as e:
            logger.error(f"Error listing FastEmbed models: {e}")
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Model information dictionary
        """
        models = self.list_available_models()
        for model in models:
            if model.get('model') == self.model_name:
                return model
        return {'model': self.model_name, 'status': 'unknown'}

# Global encoder instance
_global_encoder: Optional[SPLADEEncoder] = None

def get_splade_encoder(model_name: str = "prithivida/Splade_PP_en_v1") -> SPLADEEncoder:
    """
    Get or create a global SPLADE encoder instance.
    
    Args:
        model_name: FastEmbed SPLADE model name
        
    Returns:
        SPLADEEncoder instance
    """
    global _global_encoder
    
    if _global_encoder is None or _global_encoder.model_name != model_name:
        logger.info(f"Creating new SPLADE encoder with model: {model_name}")
        _global_encoder = SPLADEEncoder(model_name)
    
    return _global_encoder

def encode_documents_for_qdrant(texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
    """
    Convenience function to encode documents for Qdrant storage.
    
    Args:
        texts: List of documents to encode
        batch_size: Batch size for processing
        
    Returns:
        List of sparse vector dictionaries
    """
    encoder = get_splade_encoder()
    return encoder.encode_documents(texts, batch_size)

def encode_query_for_qdrant(query: str) -> Dict[str, Any]:
    """
    Convenience function to encode a query for Qdrant search.
    
    Args:
        query: Query text to encode
        
    Returns:
        Sparse vector dictionary
    """
    encoder = get_splade_encoder()
    return encoder.encode_query(query) 