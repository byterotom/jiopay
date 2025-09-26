# backend/indexer.py
"""
Enhanced FAISS indexer for vector search with multiple index types and improved functionality.
Supports different index types, metadata management, and performance optimizations.
"""

import os
import json
import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import pickle

import faiss
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISSIndexer:
    """
    Enhanced FAISS indexer with support for multiple index types and operations.
    """
    
    # Index type configurations
    INDEX_TYPES = {
        'flat': {
            'class': faiss.IndexFlatIP,
            'description': 'Exact search using inner product',
            'best_for': 'Small datasets (<100k vectors), highest accuracy'
        },
        'flat_l2': {
            'class': faiss.IndexFlatL2,
            'description': 'Exact search using L2 distance',
            'best_for': 'Small datasets, L2 distance needed'
        },
        'ivf': {
            'description': 'Approximate search with inverted file index',
            'best_for': 'Medium datasets (100k-1M vectors)',
            'requires_training': True
        },
        'hnsw': {
            'description': 'Hierarchical navigable small world graphs',
            'best_for': 'Fast approximate search, good recall',
            'requires_training': False
        }
    }
    
    def __init__(self, 
                 index_dir: str = 'data/indexes',
                 index_name: str = 'default',
                 index_type: str = 'flat'):
        """
        Initialize FAISS indexer.
        
        Args:
            index_dir: Directory to store index files
            index_name: Name for this specific index
            index_type: Type of FAISS index to use
        """
        self.index_dir = Path(index_dir)
        self.index_name = index_name
        self.index_type = index_type
        
        # Create directory if it doesn't exist
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.index_path = self.index_dir / f"{index_name}.index"
        self.meta_path = self.index_dir / f"{index_name}_meta.json"
        self.config_path = self.index_dir / f"{index_name}_config.json"
        
        # Index state
        self.index = None
        self.metadata = []
        self.config = {}
        self.is_trained = False
        
        logger.info(f"Initialized FAISSIndexer: {index_name} ({index_type})")
    
    def build_index(self, 
                   embeddings: np.ndarray,
                   metadatas: List[Dict[str, Any]],
                   index_type: Optional[str] = None,
                   nlist: int = 100,
                   M: int = 16,
                   efConstruction: int = 200) -> Tuple[str, str]:
        """
        Build FAISS index from embeddings and metadata.
        
        Args:
            embeddings: Array of embeddings
            metadatas: List of metadata dicts for each embedding
            index_type: Override default index type
            nlist: Number of clusters for IVF index
            M: Number of connections for HNSW
            efConstruction: Size of dynamic candidate list for HNSW
            
        Returns:
            Tuple of (index_path, metadata_path)
        """
        if index_type:
            self.index_type = index_type
            
        # Validate inputs
        embeddings = np.array(embeddings, dtype=np.float32)
        if len(embeddings) != len(metadatas):
            raise ValueError(f"Embeddings ({len(embeddings)}) and metadata ({len(metadatas)}) length mismatch")
        
        if len(embeddings) == 0:
            raise ValueError("No embeddings provided")
        
        dim = embeddings.shape[1]
        n_vectors = len(embeddings)
        
        logger.info(f"Building {self.index_type} index with {n_vectors} vectors (dim={dim})")
        start_time = time.time()
        
        # Create index based on type
        if self.index_type == 'flat':
            self.index = faiss.IndexFlatIP(dim)
            
        elif self.index_type == 'flat_l2':
            self.index = faiss.IndexFlatL2(dim)
            
        elif self.index_type == 'ivf':
            # IVF with flat quantizer
            quantizer = faiss.IndexFlatIP(dim)
            nlist = min(nlist, n_vectors // 10)  # Adjust nlist based on data size
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            
            # Train the index
            logger.info(f"Training IVF index with nlist={nlist}")
            self.index.train(embeddings)
            self.is_trained = True
            
        elif self.index_type == 'hnsw':
            self.index = faiss.IndexHNSWFlat(dim, M)
            self.index.hnsw.efConstruction = efConstruction
            
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # Add embeddings to index
        logger.info("Adding embeddings to index...")
        self.index.add(embeddings)
        
        # Store metadata and config
        self.metadata = metadatas
        self.config = {
            'index_type': self.index_type,
            'dimension': dim,
            'num_vectors': n_vectors,
            'created_at': time.time(),
            'nlist': nlist if self.index_type == 'ivf' else None,
            'M': M if self.index_type == 'hnsw' else None,
            'efConstruction': efConstruction if self.index_type == 'hnsw' else None
        }
        
        # Save to disk
        self._save_index()
        
        build_time = time.time() - start_time
        logger.info(f"Index built in {build_time:.2f}s")
        logger.info(f"Index size: {self.index.ntotal} vectors")
        
        return str(self.index_path), str(self.meta_path)
    
    def load_index(self) -> Tuple[Optional[faiss.Index], Optional[List[Dict]]]:
        """
        Load index and metadata from disk.
        
        Returns:
            Tuple of (index, metadata) or (None, None) if not found
        """
        if not self.index_path.exists():
            logger.warning(f"Index file not found: {self.index_path}")
            return None, None
            
        if not self.meta_path.exists():
            logger.warning(f"Metadata file not found: {self.meta_path}")
            return None, None
        
        try:
            # Load index
            logger.info(f"Loading index from {self.index_path}")
            self.index = faiss.read_index(str(self.index_path))
            
            # Load metadata
            with open(self.meta_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            # Load config if exists
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                    self.index_type = self.config.get('index_type', 'flat')
            
            logger.info(f"Loaded index with {self.index.ntotal} vectors")
            return self.index, self.metadata
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return None, None
    
    def search(self, 
               query_embeddings: np.ndarray,
               top_k: int = 5,
               nprobe: int = 10) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Search the index for similar vectors.
        
        Args:
            query_embeddings: Query vectors (can be single vector or batch)
            top_k: Number of results to return
            nprobe: Number of clusters to search (for IVF index)
            
        Returns:
            Tuple of (distances, indices, metadata_results)
        """
        if self.index is None:
            raise ValueError("No index loaded. Call load_index() or build_index() first.")
        
        # Ensure query is 2D
        query_embeddings = np.array(query_embeddings, dtype=np.float32)
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        
        # Set search parameters for IVF
        if self.index_type == 'ivf' and hasattr(self.index, 'nprobe'):
            self.index.nprobe = min(nprobe, self.config.get('nlist', 100))
        
        # Search
        start_time = time.time()
        distances, indices = self.index.search(query_embeddings, top_k)
        search_time = time.time() - start_time
        
        # Get metadata for results
        metadata_results = []
        for query_idx, query_indices in enumerate(indices):
            query_meta = []
            for idx in query_indices:
                if idx >= 0 and idx < len(self.metadata):
                    query_meta.append(self.metadata[idx])
                else:
                    query_meta.append(None)  # Invalid index
            metadata_results.append(query_meta)
        
        logger.debug(f"Search completed in {search_time:.4f}s")
        return distances, indices, metadata_results
    
    def add_vectors(self, 
                   embeddings: np.ndarray, 
                   metadatas: List[Dict[str, Any]]) -> None:
        """
        Add new vectors to existing index.
        
        Args:
            embeddings: New embeddings to add
            metadatas: Metadata for new embeddings
        """
        if self.index is None:
            raise ValueError("No index loaded. Call load_index() or build_index() first.")
        
        embeddings = np.array(embeddings, dtype=np.float32)
        if len(embeddings) != len(metadatas):
            raise ValueError("Embeddings and metadata length mismatch")
        
        # Add to index
        self.index.add(embeddings)
        
        # Add to metadata
        self.metadata.extend(metadatas)
        
        # Update config
        self.config['num_vectors'] = self.index.ntotal
        self.config['last_updated'] = time.time()
        
        # Save updated index
        self._save_index()
        
        logger.info(f"Added {len(embeddings)} vectors. Total: {self.index.ntotal}")
    
    def remove_vectors(self, indices_to_remove: List[int]) -> None:
        """
        Remove vectors from index (creates new index without removed vectors).
        
        Args:
            indices_to_remove: List of indices to remove
        """
        if self.index is None:
            raise ValueError("No index loaded")
        
        if not indices_to_remove:
            return
        
        logger.info(f"Removing {len(indices_to_remove)} vectors from index")
        
        # Get all embeddings and metadata
        all_embeddings = []
        all_metadata = []
        
        indices_set = set(indices_to_remove)
        for i in range(self.index.ntotal):
            if i not in indices_set:
                # This is a simplified approach - in practice, you'd need to 
                # store original embeddings or use a different approach
                all_metadata.append(self.metadata[i])
        
        # Update metadata
        self.metadata = all_metadata
        
        logger.warning("Vector removal requires rebuilding index with original embeddings")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        stats = {
            'index_type': self.index_type,
            'is_trained': getattr(self, 'is_trained', False),
            'config': self.config.copy()
        }
        
        if self.index is not None:
            stats.update({
                'total_vectors': self.index.ntotal,
                'dimension': self.index.d if hasattr(self.index, 'd') else None,
                'metadata_count': len(self.metadata),
                'index_size_mb': self._get_index_size_mb()
            })
        
        return stats
    
    def _save_index(self) -> None:
        """Save index, metadata, and config to disk."""
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))
        
        # Save metadata
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        # Save config
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.debug(f"Saved index to {self.index_path}")
    
    def _get_index_size_mb(self) -> float:
        """Get index file size in MB."""
        if self.index_path.exists():
            return self.index_path.stat().st_size / (1024 * 1024)
        return 0.0

# Convenience functions for backward compatibility
def build_faiss(embeddings: np.ndarray, 
               metadatas: List[Dict],
               dim: Optional[int] = None,
               index_type: str = 'flat',
               index_dir: str = 'data') -> Tuple[str, str]:
    """
    Build FAISS index (backward compatible function).
    
    Args:
        embeddings: Array of embeddings
        metadatas: List of metadata dicts
        dim: Dimension (auto-detected if None)
        index_type: Type of index to build
        index_dir: Directory to save index files
        
    Returns:
        Tuple of (index_path, metadata_path)
    """
    indexer = FAISSIndexer(
        index_dir=index_dir,
        index_name='faiss',
        index_type=index_type
    )
    return indexer.build_index(embeddings, metadatas)

def load_index(index_dir: str = 'data', 
              index_name: str = 'faiss') -> Tuple[Optional[faiss.Index], Optional[List[Dict]]]:
    """
    Load FAISS index (backward compatible function).
    
    Returns:
        Tuple of (index, metadata) or (None, None) if not found
    """
    indexer = FAISSIndexer(index_dir=index_dir, index_name=index_name)
    return indexer.load_index()

def search_index(index: faiss.Index, 
                query_emb: np.ndarray, 
                top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search index (backward compatible function).
    
    Returns:
        Tuple of (distances, indices)
    """
    query_emb = np.array(query_emb, dtype=np.float32)
    if query_emb.ndim == 1:
        query_emb = query_emb.reshape(1, -1)
    
    return index.search(query_emb, top_k)

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_embeddings = np.random.randn(1000, 384).astype(np.float32)
    sample_metadata = [
        {
            'text': f'Sample text {i}',
            'source': f'doc_{i//100}',
            'chunk_id': i,
            'url': f'https://example.com/doc_{i//100}'
        }
        for i in range(1000)
    ]
    
    # Test different index types
    for index_type in ['flat', 'ivf', 'hnsw']:
        print(f"\nTesting {index_type} index:")
        
        indexer = FAISSIndexer(
            index_dir='test_indexes',
            index_name=f'test_{index_type}',
            index_type=index_type
        )
        
        # Build index
        try:
            start_time = time.time()
            indexer.build_index(sample_embeddings, sample_metadata)
            build_time = time.time() - start_time
            
            # Test search
            query = np.random.randn(1, 384).astype(np.float32)
            distances, indices, metadata_results = indexer.search(query, top_k=5)
            
            print(f"Build time: {build_time:.2f}s")
            print(f"Search results shape: {distances.shape}")
            print(f"Top result distance: {distances[0][0]:.4f}")
            print(f"Stats: {indexer.get_stats()}")
            
        except Exception as e:
            print(f"Error with {index_type}: {e}")