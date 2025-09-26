# backend/app.py
"""
Enhanced FastAPI application for JioPay RAG chatbot with comprehensive features.
"""

import os
import time
import logging
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Import with error handling
try:
    from retriever import retrieve, get_retriever_stats
except ImportError as e:
    logging.error(f"Failed to import retriever: {e}")
    retrieve = None
    get_retriever_stats = None

try:
    from generator import HuggingFaceGenerator, create_generator, generate_answer
    GENERATOR_AVAILABLE = True
except ImportError as e:
    logging.error(f"Failed to import generator: {e}")
    HuggingFaceGenerator = None
    create_generator = None
    generate_answer = None
    GENERATOR_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
app_state = {
    "generator": None,
    "startup_time": None,
    "request_count": 0,
    "error_count": 0
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting JioPay RAG application...")
    app_state["startup_time"] = time.time()
    
    # Initialize generator if available
    if GENERATOR_AVAILABLE:
        try:
            app_state["generator"] = HuggingFaceGenerator(model_name='flan-t5-large')
            logger.info("Generator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize generator: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down JioPay RAG application...")

# Create FastAPI app with enhanced configuration
app = FastAPI(
    title="JioPay RAG Chatbot",
    description="Production-grade Retrieval-Augmented Generation chatbot for JioPay customer support",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enhanced CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Request/Response Models
class ChatRequest(BaseModel):
    """Chat request model with validation."""
    query: str = Field(..., min_length=1, max_length=1000, description="User's question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of retrieved chunks")
    embed_model: str = Field(
        default="all-MiniLM-L6-v2", 
        description="Embedding model to use",
        pattern="^[a-zA-Z0-9_-]+$"
    )
    generator_provider: str = Field(
        default="huggingface",
        description="Generator provider (huggingface or openai)"
    )
    generator_model: str = Field(
        default="flan-t5-base",
        description="Generator model name"
    )
    include_debug: bool = Field(
        default=False,
        description="Include debug information in response"
    )

    @validator('query')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

class Citation(BaseModel):
    """Citation model for source references."""
    index: int = Field(..., description="Citation index")
    url: str = Field(..., description="Source URL")
    title: str = Field(..., description="Source title")
    snippet: str = Field(..., description="Relevant text snippet")
    score: float = Field(..., description="Relevance score")

class RetrievalResult(BaseModel):
    """Individual retrieval result."""
    score: float = Field(..., description="Similarity score")
    meta: Dict[str, Any] = Field(..., description="Chunk metadata")

class ChatResponse(BaseModel):
    """Enhanced chat response model."""
    answer: str = Field(..., description="Generated answer")
    citations: List[Citation] = Field(default=[], description="Source citations")
    retrieved: List[RetrievalResult] = Field(default=[], description="Retrieved chunks")
    
    # Metadata
    generation_time: float = Field(..., description="Generation time in seconds")
    retrieval_time: float = Field(..., description="Retrieval time in seconds")
    total_time: float = Field(..., description="Total processing time")
    model_info: Dict[str, str] = Field(..., description="Model information")
    
    # Debug info (optional)
    debug: Optional[Dict[str, Any]] = Field(None, description="Debug information")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    uptime: float
    version: str
    models_loaded: Dict[str, bool]
    request_count: int
    error_count: int

class StatsResponse(BaseModel):
    """Statistics response."""
    retriever_stats: Optional[Dict[str, Any]] = None
    app_stats: Dict[str, Any]

# Middleware for request tracking
@app.middleware("http")
async def track_requests(request, call_next):
    """Track request metrics."""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        app_state["request_count"] += 1
        return response
    except Exception as e:
        app_state["error_count"] += 1
        logger.error(f"Request failed: {str(e)}")
        raise
    finally:
        process_time = time.time() - start_time
        logger.info(f"{request.method} {request.url.path} - {process_time:.3f}s")

# Main chat endpoint
@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Main chat endpoint with retrieval and generation.
    
    - Retrieves relevant chunks from knowledge base
    - Generates contextual answer using LLM
    - Returns answer with citations and metadata
    """
    if not retrieve:
        raise HTTPException(
            status_code=503, 
            detail="Retriever not available. Please check index files."
        )
    
    if not GENERATOR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Generator not available. Please install required dependencies."
        )
    
    total_start_time = time.time()
    
    try:
        # Retrieval phase
        retrieval_start = time.time()
        retrieved_results = retrieve(
            request.query, 
            model_name=request.embed_model, 
            top_k=request.top_k
        )
        retrieval_time = time.time() - retrieval_start
        
        if not retrieved_results:
            return ChatResponse(
                answer="I couldn't find relevant information to answer your question. Please contact JioPay support directly.",
                citations=[],
                retrieved=[],
                generation_time=0,
                retrieval_time=retrieval_time,
                total_time=time.time() - total_start_time,
                model_info={
                    "embedding_model": request.embed_model,
                    "generator": "none"
                }
            )
        
        # Generation phase
        generation_start = time.time()
        
        # Use the enhanced generator if available
        if app_state["generator"] and request.generator_provider == "huggingface":
            result = app_state["generator"].generate_answer(request.query, retrieved_results)
            answer = result["answer"]
            citations = result.get("citations", [])
            generation_time = result.get("generation_time", time.time() - generation_start)
        else:
            # Fallback to backward compatible function
            answer = generate_answer(request.query, retrieved_results)
            citations = []
            generation_time = time.time() - generation_start
        
        total_time = time.time() - total_start_time
        
        # Convert retrieved results to response format
        retrieved_formatted = [
            RetrievalResult(
                score=r.get("score", 0.0),
                meta=r.get("meta", {})
            )
            for r in retrieved_results
        ]
        
        # Create citations if not already provided
        if not citations:
            citations = [
                Citation(
                    index=i,
                    url=r.get("meta", {}).get("url", ""),
                    title=r.get("meta", {}).get("title", ""),
                    snippet=r.get("meta", {}).get("text", "")[:200] + "...",
                    score=r.get("score", 0.0)
                )
                for i, r in enumerate(retrieved_results)
            ]
        
        # Debug information
        debug_info = None
        if request.include_debug:
            debug_info = {
                "query_length": len(request.query),
                "retrieved_count": len(retrieved_results),
                "avg_retrieval_score": sum(r.get("score", 0) for r in retrieved_results) / len(retrieved_results) if retrieved_results else 0,
                "processing_steps": {
                    "retrieval_time": retrieval_time,
                    "generation_time": generation_time,
                    "total_time": total_time
                }
            }
        
        # Log successful request
        background_tasks.add_task(
            log_chat_request, 
            request.query, 
            len(retrieved_results), 
            total_time
        )
        
        return ChatResponse(
            answer=answer,
            citations=citations,
            retrieved=retrieved_formatted,
            generation_time=generation_time,
            retrieval_time=retrieval_time,
            total_time=total_time,
            model_info={
                "embedding_model": request.embed_model,
                "generator_provider": request.generator_provider,
                "generator_model": request.generator_model
            },
            debug=debug_info
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

# Additional endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with basic info."""
    return {
        "message": "✅ JioPay RAG backend is running!",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Comprehensive health check."""
    uptime = time.time() - app_state["startup_time"] if app_state["startup_time"] else 0
    
    models_loaded = {
        "retriever": retrieve is not None,
        "generator": app_state["generator"] is not None,
        "embeddings": True  # Assuming embeddings are always available
    }
    
    return HealthResponse(
        status="healthy" if all(models_loaded.values()) else "degraded",
        uptime=uptime,
        version="1.0.0",
        models_loaded=models_loaded,
        request_count=app_state["request_count"],
        error_count=app_state["error_count"]
    )

@app.get("/stats", response_model=StatsResponse, tags=["Stats"])
async def get_stats():
    """Get application and model statistics."""
    retriever_stats = None
    if get_retriever_stats:
        try:
            retriever_stats = get_retriever_stats()
        except Exception as e:
            logger.warning(f"Could not get retriever stats: {e}")
    
    app_stats = {
        "uptime": time.time() - app_state["startup_time"] if app_state["startup_time"] else 0,
        "request_count": app_state["request_count"],
        "error_count": app_state["error_count"],
        "error_rate": app_state["error_count"] / max(app_state["request_count"], 1)
    }
    
    return StatsResponse(
        retriever_stats=retriever_stats,
        app_stats=app_stats
    )

@app.get("/models", tags=["Models"])
async def list_models():
    """List available models."""
    return {
        "embedding_models": ["all-MiniLM-L6-v2", "e5-small", "e5-base", "e5-large"],
        "generator_providers": ["huggingface", "openai"],
        "generator_models": {
            "huggingface": ["flan-t5-small", "flan-t5-base", "flan-t5-large"],
            "openai": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        }
    }

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    return JSONResponse(
        status_code=422,
        content={"detail": f"Validation error: {str(exc)}"}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Enhanced HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "timestamp": time.time(),
            "path": str(request.url.path)
        }
    )

# Background tasks
async def log_chat_request(query: str, retrieved_count: int, processing_time: float):
    """Log chat request for analytics."""
    logger.info(f"Chat request processed: query_len={len(query)}, retrieved={retrieved_count}, time={processing_time:.3f}s")

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )
