# retriever.py
from embeddings import embed_texts
from indexer import FAISSIndexer
import numpy as np

def retrieve(query, model_name='all-MiniLM-L6-v2', top_k=5, index_name=None):
    indexer = FAISSIndexer(
        index_dir="data/indexes",
        index_name=index_name or "semantic_all-MiniLM-L6-v2_flat"
    )
    idx, meta = indexer.load_index()
    if idx is None:
        raise RuntimeError("Index not found. Build it first.")

    # Embed the query
    q_emb = embed_texts([query], model_name=model_name)

    # Use the new search method (returns distances, indices, metadata_results)
    D, I, meta_results = indexer.search(q_emb.astype("float32"), top_k=top_k)

    # Build results list using returned metadata
    results = []
    for score, m in zip(D[0], meta_results[0]):
        if m is None:
            continue
        results.append({"score": float(score), "meta": m})

    return results


def get_retriever_stats():
    """Return stats about the current FAISS index."""
    indexer = FAISSIndexer(index_dir="data/indexes", index_name="semantic_all-MiniLM-L6-v2_flat")
    idx, meta = indexer.load_index()
    if idx is None or meta is None:
        return {"status": "index not loaded"}
    return {
        "status": "ok",
        "ntotal": idx.ntotal,
        "dimension": idx.d,
        "metadata_count": len(meta),
    }
