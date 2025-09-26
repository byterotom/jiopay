# backend/build_index.py
"""
Enhanced index builder with support for multiple chunking strategies,
embedding models, and comprehensive ablation studies.
"""

import os
import json
import glob
import time
import argparse
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass, asdict

# Import chunking strategies
from chunking import (
    fixed_chunk, 
    semantic_chunk, 
    structural_chunk, 
    recursive_chunk,
    llm_chunk
)
from embeddings import embed_texts, get_available_models
from indexer import FAISSIndexer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BuildConfig:
    """Configuration for index building."""
    data_dir: str = "data/jiopay"
    output_dir: str = "data/indexes"
    chunking_strategy: str = "semantic"
    embedding_model: str = "all-MiniLM-L6-v2"
    index_type: str = "flat"
    chunk_size: int = 512
    chunk_overlap: int = 64
    max_chunks_per_doc: Optional[int] = None
    batch_size: int = 32
    index_name: Optional[str] = None

class IndexBuilder:
    """Enhanced index builder with ablation study support."""
    
    # Available chunking strategies
    CHUNKING_STRATEGIES = {
        'fixed': fixed_chunk,
        'semantic': semantic_chunk,
        'structural': structural_chunk,
        'recursive': recursive_chunk,
        'llm': llm_chunk
    }
    
    def __init__(self, config: BuildConfig):
        self.config = config
        self.stats = {
            'documents_loaded': 0,
            'documents_processed': 0,
            'chunks_generated': 0,
            'embeddings_created': 0,
            'build_time': 0,
            'errors': []
        }
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized IndexBuilder with config: {asdict(config)}")
    
    def load_documents(self) -> List[Dict[str, Any]]:
        """
        Load documents from various formats with enhanced error handling.
        
        Returns:
            List of loaded documents
        """
        logger.info(f"Loading documents from {self.config.data_dir}")
        
        docs = []
        supported_formats = ['*.json', '*.txt', '*.md']
        
        for pattern in supported_formats:
            files = glob.glob(os.path.join(self.config.data_dir, pattern))
            
            for file_path in files:
                try:
                    doc = self._load_single_file(file_path)
                    if doc:
                        docs.append(doc)
                        
                except Exception as e:
                    error_msg = f"Error loading {file_path}: {str(e)}"
                    logger.warning(error_msg)
                    self.stats['errors'].append(error_msg)
        
        self.stats['documents_loaded'] = len(docs)
        logger.info(f"Successfully loaded {len(docs)} documents")
        
        return docs
    
    def _load_single_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load a single file and return document dict."""
        file_ext = Path(file_path).suffix.lower()
        
        with open(file_path, encoding='utf-8') as f:
            if file_ext == '.json':
                doc = json.load(f)
                # Ensure required fields
                if 'text' not in doc:
                    logger.warning(f"No 'text' field in {file_path}")
                    return None
                return doc
                
            elif file_ext in ['.txt', '.md']:
                content = f.read().strip()
                if not content:
                    return None
                
                # Create document structure
                filename = Path(file_path).stem
                return {
                    'text': content,
                    'title': filename,
                    'url': f'file://{file_path}',
                    'source_file': file_path
                }
        
        return None
    
    def create_chunks(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create chunks from documents using the specified strategy.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            List of chunk dictionaries with metadata
        """
        logger.info(f"Creating chunks using {self.config.chunking_strategy} strategy")
        
        if self.config.chunking_strategy not in self.CHUNKING_STRATEGIES:
            raise ValueError(f"Unknown chunking strategy: {self.config.chunking_strategy}")
        
        chunk_func = self.CHUNKING_STRATEGIES[self.config.chunking_strategy]
        all_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            if not doc.get('text'):
                logger.warning(f"Document {doc_idx} has no text content")
                continue
            
            try:
                # Get chunks based on strategy
                if self.config.chunking_strategy == 'fixed':
                    chunks = chunk_func(
                        doc['text'], 
                        chunk_size=self.config.chunk_size,
                        overlap=self.config.chunk_overlap
                    )
                elif self.config.chunking_strategy in ['semantic', 'structural', 'recursive']:
                    chunks = chunk_func(doc['text'])
                elif self.config.chunking_strategy == 'llm':
                    chunks = chunk_func(doc['text'], max_chunks=self.config.max_chunks_per_doc)
                else:
                    chunks = chunk_func(doc['text'])
                
                # Add metadata to each chunk
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_dict = {
                        'text': chunk.get('text', chunk) if isinstance(chunk, dict) else chunk,
                        'meta': {
                            'url': doc.get('url', ''),
                            'title': doc.get('title', ''),
                            'source_file': doc.get('source_file', ''),
                            'doc_index': doc_idx,
                            'chunk_index': chunk_idx,
                            'chunking_strategy': self.config.chunking_strategy,
                            'chunk_size': len(chunk.get('text', chunk) if isinstance(chunk, dict) else chunk)
                        }
                    }
                    
                    # Add chunk-specific metadata if available
                    if isinstance(chunk, dict):
                        chunk_dict['meta'].update({
                            k: v for k, v in chunk.items() 
                            if k not in ['text', 'meta']
                        })
                    
                    all_chunks.append(chunk_dict)
                
                self.stats['documents_processed'] += 1
                
                # Limit chunks per document if specified
                if (self.config.max_chunks_per_doc and 
                    len(all_chunks) >= self.config.max_chunks_per_doc):
                    logger.info(f"Reached max chunks limit: {self.config.max_chunks_per_doc}")
                    break
                    
            except Exception as e:
                error_msg = f"Error chunking document {doc_idx}: {str(e)}"
                logger.error(error_msg)
                self.stats['errors'].append(error_msg)
        
        self.stats['chunks_generated'] = len(all_chunks)
        logger.info(f"Generated {len(all_chunks)} chunks from {self.stats['documents_processed']} documents")
        
        return all_chunks
    
    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> Tuple[List, List[Dict]]:
        """
        Create embeddings for chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Tuple of (embeddings, chunk_metadata)
        """
        logger.info(f"Creating embeddings using {self.config.embedding_model}")
        
        if not chunks:
            raise ValueError("No chunks provided for embedding")
        
        # Extract texts
        texts = []
        metadatas = []
        
        for chunk in chunks:
            text = chunk.get('text', '')
            if not text or not text.strip():
                logger.warning("Skipping empty chunk")
                continue
                
            texts.append(text.strip())
            metadatas.append(chunk.get('meta', {}))
        
        if not texts:
            raise ValueError("No valid texts found in chunks")
        
        try:
            # Create embeddings
            start_time = time.time()
            embeddings = embed_texts(
                texts, 
                model_name=self.config.embedding_model,
                batch_size=self.config.batch_size
            )
            embedding_time = time.time() - start_time
            
            self.stats['embeddings_created'] = len(embeddings)
            logger.info(f"Created {len(embeddings)} embeddings in {embedding_time:.2f}s")
            
            return embeddings, metadatas
            
        except Exception as e:
            error_msg = f"Error creating embeddings: {str(e)}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)
            raise
    
    def build_index(self, 
                   embeddings, 
                   metadatas: List[Dict],
                   index_name: Optional[str] = None) -> str:
        """
        Build and save FAISS index.
        
        Args:
            embeddings: Embedding vectors
            metadatas: Metadata for each embedding
            index_name: Custom index name
            
        Returns:
            Path to saved index
        """
        if index_name is None:
            index_name = (self.config.index_name or 
                         f"{self.config.chunking_strategy}_{self.config.embedding_model}_{self.config.index_type}")
        
        logger.info(f"Building {self.config.index_type} index: {index_name}")
        
        try:
            # Create indexer
            indexer = FAISSIndexer(
                index_dir=self.config.output_dir,
                index_name=index_name,
                index_type=self.config.index_type
            )
            
            # Build index
            start_time = time.time()
            index_path, meta_path = indexer.build_index(embeddings, metadatas)
            build_time = time.time() - start_time
            
            self.stats['build_time'] = build_time
            logger.info(f"Index built and saved to {index_path} in {build_time:.2f}s")
            
            # Save build statistics
            stats_path = Path(self.config.output_dir) / f"{index_name}_build_stats.json"
            with open(stats_path, 'w') as f:
                json.dump({
                    'config': asdict(self.config),
                    'stats': self.stats,
                    'timestamp': time.time()
                }, f, indent=2)
            
            return index_path
            
        except Exception as e:
            error_msg = f"Error building index: {str(e)}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)
            raise
    
    def build_complete_index(self) -> str:
        """
        Complete index building pipeline.
        
        Returns:
            Path to built index
        """
        logger.info("Starting complete index building pipeline")
        total_start_time = time.time()
        
        try:
            # Load documents
            documents = self.load_documents()
            if not documents:
                raise ValueError("No documents loaded")
            
            # Create chunks
            chunks = self.create_chunks(documents)
            if not chunks:
                raise ValueError("No chunks created")
            
            # Create embeddings
            embeddings, metadatas = self.create_embeddings(chunks)
            
            # Build index
            index_path = self.build_index(embeddings, metadatas)
            
            total_time = time.time() - total_start_time
            logger.info(f"✅ Complete index building finished in {total_time:.2f}s")
            logger.info(f"Final stats: {self.stats}")
            
            return index_path
            
        except Exception as e:
            logger.error(f"❌ Index building failed: {str(e)}")
            raise

def run_ablation_study():
    """
    Run comprehensive ablation study testing different configurations.
    """
    logger.info("Starting ablation study")
    
    # Define parameter grids
    chunking_strategies = ['fixed', 'semantic', 'structural', 'recursive']
    embedding_models = ['all-MiniLM-L6-v2', 'e5-small', 'e5-base']
    chunk_sizes = [256, 512, 1024]  # For fixed chunking
    overlaps = [0, 64, 128]  # For fixed chunking
    
    results = []
    
    for chunking in chunking_strategies:
        for model in embedding_models:
            if chunking == 'fixed':
                # Test different sizes and overlaps for fixed chunking
                for size in chunk_sizes:
                    for overlap in overlaps:
                        config = BuildConfig(
                            chunking_strategy=chunking,
                            embedding_model=model,
                            chunk_size=size,
                            chunk_overlap=overlap,
                            index_name=f"ablation_{chunking}_{model}_{size}_{overlap}"
                        )
                        
                        try:
                            builder = IndexBuilder(config)
                            index_path = builder.build_complete_index()
                            
                            results.append({
                                'config': asdict(config),
                                'stats': builder.stats,
                                'index_path': index_path,
                                'success': True
                            })
                            
                        except Exception as e:
                            logger.error(f"Ablation failed for {config.index_name}: {e}")
                            results.append({
                                'config': asdict(config),
                                'error': str(e),
                                'success': False
                            })
            else:
                # Test other chunking strategies without size variations
                config = BuildConfig(
                    chunking_strategy=chunking,
                    embedding_model=model,
                    index_name=f"ablation_{chunking}_{model}"
                )
                
                try:
                    builder = IndexBuilder(config)
                    index_path = builder.build_complete_index()
                    
                    results.append({
                        'config': asdict(config),
                        'stats': builder.stats,
                        'index_path': index_path,
                        'success': True
                    })
                    
                except Exception as e:
                    logger.error(f"Ablation failed for {config.index_name}: {e}")
                    results.append({
                        'config': asdict(config),
                        'error': str(e),
                        'success': False
                    })
    
    # Save ablation results
    results_path = "data/ablation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✅ Ablation study completed. Results saved to {results_path}")
    
    # Print summary
    successful = sum(1 for r in results if r['success'])
    logger.info(f"Summary: {successful}/{len(results)} configurations successful")

def main():
    """Main function with CLI support."""
    parser = argparse.ArgumentParser(description="Build FAISS index for RAG system")
    parser.add_argument("--data-dir", default="data/jiopay", help="Data directory")
    parser.add_argument("--output-dir", default="data/indexes", help="Output directory")
    parser.add_argument("--chunking", default="semantic", 
                       choices=['fixed', 'semantic', 'structural', 'recursive', 'llm'],
                       help="Chunking strategy")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2",
                       help="Embedding model name")
    parser.add_argument("--index-type", default="flat",
                       choices=['flat', 'flat_l2', 'ivf', 'hnsw'],
                       help="FAISS index type")
    parser.add_argument("--chunk-size", type=int, default=512,
                       help="Chunk size for fixed chunking")
    parser.add_argument("--chunk-overlap", type=int, default=64,
                       help="Chunk overlap for fixed chunking")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for embedding creation")
    parser.add_argument("--index-name", help="Custom index name")
    parser.add_argument("--ablation", action="store_true",
                       help="Run ablation study")
    parser.add_argument("--max-chunks", type=int, help="Max chunks per document")
    
    args = parser.parse_args()
    
    if args.ablation:
        run_ablation_study()
    else:
        # Build single index
        config = BuildConfig(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            chunking_strategy=args.chunking,
            embedding_model=args.embedding_model,
            index_type=args.index_type,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=args.batch_size,
            index_name=args.index_name,
            max_chunks_per_doc=args.max_chunks
        )
        
        builder = IndexBuilder(config)
        index_path = builder.build_complete_index()
        print(f"✅ Index built successfully: {index_path}")

# Backward compatibility function
def build_index():
    """Original build_index function for backward compatibility."""
    config = BuildConfig()
    builder = IndexBuilder(config)
    return builder.build_complete_index()

if __name__ == "__main__":
    main()