# backend/embeddings.py
"""
Embeddings wrapper with support for multiple models and improved error handling.
Supports sentence-transformers and E5 models with caching and batch processing.
"""

import logging
import time
from typing import List, Optional, Union, Dict, Any
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configurations with metadata
MODELS = {
    'all-MiniLM-L6-v2': {
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'dimensions': 384,
        'max_seq_length': 256,
        'description': 'Fast, lightweight model for general use'
    },
    'e5-small': {
        'model_name': 'intfloat/e5-small-v2',
        'dimensions': 384,
        'max_seq_length': 512,
        'description': 'Small E5 model, good balance of speed/quality'
    },
    'e5-base': {
        'model_name': 'intfloat/e5-base-v2',
        'dimensions': 768,
        'max_seq_length': 512,
        'description': 'Base E5 model, higher quality'
    },
    'e5-large': {
        'model_name': 'intfloat/e5-large-v2',
        'dimensions': 1024,
        'max_seq_length': 512,
        'description': 'Large E5 model, best quality but slower'
    },
    'bge-small': {
        'model_name': 'BAAI/bge-small-en-v1.5',
        'dimensions': 384,
        'max_seq_length': 512,
        'description': 'BGE small model, good for retrieval tasks'
    }
}

# Global model cache
_model_cache: Dict[str, SentenceTransformer] = {}

class EmbeddingError(Exception):
    """Custom exception for embedding-related errors."""
    pass

def get_available_models() -> List[str]:
    """Get list of available model names."""
    return list(MODELS.keys())

def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get information about a specific model."""
    if model_name not in MODELS:
        raise EmbeddingError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    return MODELS[model_name].copy()

def get_model(model_name: str = 'all-MiniLM-L6-v2', 
              cache_folder: Optional[str] = None,
              device: Optional[str] = None) -> SentenceTransformer:
    """
    Get or load a sentence transformer model with caching.
    
    Args:
        model_name: Name of the model to load
        cache_folder: Optional custom cache folder for model files
        device: Device to load model on ('cpu', 'cuda', 'mps', etc.)
        
    Returns:
        Loaded SentenceTransformer model
        
    Raises:
        EmbeddingError: If model loading fails
    """
    if model_name not in MODELS:
        raise EmbeddingError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    
    cache_key = f"{model_name}_{device or 'auto'}_{cache_folder or 'default'}"
    
    if cache_key in _model_cache:
        logger.debug(f"Using cached model: {model_name}")
        return _model_cache[cache_key]
    
    try:
        logger.info(f"Loading model: {model_name}")
        start_time = time.time()
        
        model_path = MODELS[model_name]['model_name']
        model = SentenceTransformer(
            model_path,
            cache_folder=cache_folder,
            device=device
        )
        
        load_time = time.time() - start_time
        logger.info(f"Model {model_name} loaded in {load_time:.2f}s")
        logger.info(f"Model device: {model.device}")
        
        _model_cache[cache_key] = model
        return model
        
    except Exception as e:
        raise EmbeddingError(f"Failed to load model {model_name}: {str(e)}")

def preprocess_texts(texts: List[str], model_name: str) -> List[str]:
    """
    Preprocess texts based on model requirements.
    E5 models benefit from query/passage prefixes.
    """
    if not isinstance(texts, list):
        texts = [texts]
    
    # E5 models benefit from prefixes for different text types
    if model_name.startswith('e5'):
        # For general embedding, use passage prefix
        # In a real system, you might want to distinguish query vs passage
        processed = [f"passage: {text}" for text in texts]
    else:
        processed = texts
    
    return processed

def embed_texts(texts: Union[str, List[str]], 
                model_name: str = 'all-MiniLM-L6-v2',
                batch_size: int = 32,
                normalize: bool = True,
                show_progress: bool = True,
                convert_to_tensor: bool = False,
                device: Optional[str] = None) -> np.ndarray:
    """
    Generate embeddings for input texts with improved error handling and options.
    
    Args:
        texts: Single text string or list of texts to embed
        model_name: Name of the embedding model to use
        batch_size: Batch size for processing
        normalize: Whether to L2 normalize embeddings
        show_progress: Whether to show progress bar
        convert_to_tensor: Return torch tensor instead of numpy array
        device: Device to use for inference
        
    Returns:
        Embeddings as numpy array or torch tensor
        
    Raises:
        EmbeddingError: If embedding generation fails
    """
    if isinstance(texts, str):
        texts = [texts]
    
    if not texts:
        raise EmbeddingError("No texts provided for embedding")
    
    if len(texts) == 0:
        raise EmbeddingError("Empty text list provided")
    
    try:
        model = get_model(model_name, device=device)
        
        # Preprocess texts if needed
        processed_texts = preprocess_texts(texts, model_name)
        
        logger.info(f"Embedding {len(processed_texts)} texts with {model_name}")
        start_time = time.time()
        
        # Generate embeddings
        embeddings = model.encode(
            processed_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_tensor=convert_to_tensor,
            normalize_embeddings=False  # We'll handle normalization manually
        )
        
        # Convert to numpy if needed
        if convert_to_tensor:
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        else:
            embeddings = np.array(embeddings, dtype=np.float32)
            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
                embeddings = embeddings / norms
        
        embed_time = time.time() - start_time
        logger.info(f"Generated {len(embeddings)} embeddings in {embed_time:.2f}s")
        
        return embeddings
        
    except Exception as e:
        raise EmbeddingError(f"Failed to generate embeddings: {str(e)}")

def compute_similarity(embeddings1: np.ndarray, 
                      embeddings2: np.ndarray,
                      metric: str = 'cosine') -> np.ndarray:
    """
    Compute similarity between two sets of embeddings.
    
    Args:
        embeddings1: First set of embeddings
        embeddings2: Second set of embeddings  
        metric: Similarity metric ('cosine', 'dot', 'euclidean')
        
    Returns:
        Similarity matrix
    """
    if metric == 'cosine':
        # Assuming embeddings are normalized
        return np.dot(embeddings1, embeddings2.T)
    elif metric == 'dot':
        return np.dot(embeddings1, embeddings2.T)
    elif metric == 'euclidean':
        # Compute negative euclidean distance (higher = more similar)
        dists = np.linalg.norm(embeddings1[:, None] - embeddings2[None, :], axis=2)
        return -dists
    else:
        raise EmbeddingError(f"Unknown similarity metric: {metric}")

def clear_cache():
    """Clear the model cache to free up memory."""
    global _model_cache
    logger.info(f"Clearing {len(_model_cache)} cached models")
    for model in _model_cache.values():
        if hasattr(model, 'cpu'):
            model.cpu()
        del model
    _model_cache.clear()
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Convenience functions for common use cases
def embed_query(query: str, model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """Embed a single query text."""
    if model_name.startswith('e5'):
        query = f"query: {query}"
    return embed_texts([query], model_name=model_name)[0]

def embed_documents(documents: List[str], 
                   model_name: str = 'all-MiniLM-L6-v2',
                   batch_size: int = 32) -> np.ndarray:
    """Embed multiple documents with appropriate preprocessing."""
    return embed_texts(documents, model_name=model_name, batch_size=batch_size)

if __name__ == "__main__":
    # Example usage and testing
    test_texts = [
        "How do I reset my password?",
        "What are the payment methods available?",
        "How to contact customer support?"
    ]
    
    print("Available models:", get_available_models())
    
    for model_name in ['all-MiniLM-L6-v2', 'e5-small']:
        print(f"\nTesting {model_name}:")
        try:
            embeddings = embed_texts(test_texts, model_name=model_name)
            print(f"Shape: {embeddings.shape}")
            print(f"Sample embedding (first 5 dims): {embeddings[0][:5]}")
        except EmbeddingError as e:
            print(f"Error: {e}")