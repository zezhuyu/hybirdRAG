import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status as http_status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add HybirdRAG to path
sys.path.insert(0, str(Path(__file__).parent))

from HybirdRAG.pipeline import HybridRAGPipeline
from clean import (
    clear_all_databases,
    clear_milvus_collection,
    clear_neo4j_database,
    get_database_status as get_db_status
)

# Initialize FastAPI app
app = FastAPI(
    title="HybirdRAG API",
    description="REST API for HybirdRAG document management and retrieval",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (initialized lazily)
_pipeline: Optional[HybridRAGPipeline] = None


def get_pipeline() -> HybridRAGPipeline:
    """Get or create the HybridRAG pipeline instance."""
    global _pipeline
    if _pipeline is None:
        try:
            _pipeline = HybridRAGPipeline()
        except Exception as e:
            raise HTTPException(
                status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize pipeline: {str(e)}"
            ) from e
    return _pipeline


# Request/Response Models
class Document(BaseModel):
    """Document model for adding documents."""
    content: str = Field(..., description="Document text content")
    id: Optional[str] = Field(None, description="Optional document ID")
    title: Optional[str] = Field(None, description="Optional document title")
    page: Optional[int] = Field(None, description="Optional page number")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class DocumentsRequest(BaseModel):
    """Request model for adding multiple documents."""
    documents: List[Document] = Field(..., description="List of documents to add")


class QueryRequest(BaseModel):
    """Request model for querying."""
    query: str = Field(..., description="Search query")
    rewrite: bool = Field(True, description="Enable query rewriting")
    chat_history: Optional[List[str]] = Field(None, description="Chat history for context")
    broaden_query: bool = Field(True, description="Enable query broadening")
    broaden_retry_limit: int = Field(3, description="Query broadening retry limit")
    context_chunk_size: int = Field(256, description="Context chunk size")
    rerank: bool = Field(True, description="Enable reranking")
    compress: bool = Field(False, description="Enable result compression")
    limit: int = Field(30, description="Maximum number of results")


class QueryResponse(BaseModel):
    """Response model for queries."""
    results: List[str] = Field(..., description="Retrieved document chunks")
    count: int = Field(..., description="Number of results returned")


class CleanRequest(BaseModel):
    """Request model for cleaning database."""
    collection_name: str = Field("vector_rag", description="Collection name to clean")
    skip_neo4j_on_auth_failure: bool = Field(True, description="Skip Neo4j if auth fails")


class CleanResponse(BaseModel):
    """Response model for clean operations."""
    success: bool = Field(..., description="Whether the operation succeeded")
    milvus: Dict[str, Any] = Field(..., description="Milvus operation result")
    neo4j: Dict[str, Any] = Field(..., description="Neo4j operation result")


class StatusResponse(BaseModel):
    """Response model for database status."""
    milvus: Dict[str, Any] = Field(..., description="Milvus status")
    neo4j: Dict[str, Any] = Field(..., description="Neo4j status")


# Health Check
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "HybirdRAG API"}


# Document Management Endpoints
@app.post("/api/v1/documents", tags=["Documents"], status_code=http_status.HTTP_201_CREATED)
async def add_document(document: Document):
    """
    Add a single document to the RAG system.
    
    The document will be:
    - Chunked and added to vector store (Milvus)
    - Processed and added to knowledge graph (Neo4j) if GraphRAG is enabled
    """
    try:
        pipeline = get_pipeline()
        
        # Convert Pydantic model to dict
        doc_dict = {
            "content": document.content,
            **({"id": document.id} if document.id else {}),
            **({"title": document.title} if document.title else {}),
            **({"page": document.page} if document.page is not None else {}),
            **(document.metadata or {})
        }
        
        # Add document
        pipeline.add_document(doc_dict)
        
        return {
            "success": True,
            "message": "Document added successfully",
            "document_id": document.id or "auto-generated"
        }
    except Exception as e:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add document: {str(e)}"
        ) from e


@app.post("/api/v1/documents/batch", tags=["Documents"], status_code=http_status.HTTP_201_CREATED)
async def add_documents(request: DocumentsRequest):
    """
    Add multiple documents to the RAG system in batch.
    
    This is more efficient than adding documents one by one.
    """
    try:
        pipeline = get_pipeline()
        
        # Convert Pydantic models to dicts
        documents = []
        for doc in request.documents:
            doc_dict = {
                "content": doc.content,
                **({"id": doc.id} if doc.id else {}),
                **({"title": doc.title} if doc.title else {}),
                **({"page": doc.page} if doc.page is not None else {}),
                **(doc.metadata or {})
            }
            documents.append(doc_dict)
        
        # Add documents in batch
        pipeline.add_documents(documents)
        
        return {
            "success": True,
            "message": f"Successfully added {len(documents)} documents",
            "count": len(documents)
        }
    except Exception as e:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add documents: {str(e)}"
        ) from e


# Retrieval Endpoints
@app.post("/api/v1/query", tags=["Retrieval"], response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system to retrieve relevant documents.
    
    Supports:
    - Query rewriting for better specificity
    - Query broadening for better coverage
    - Reranking for better relevance
    - Result compression for efficiency
    - Multi-hop question handling
    """
    try:
        pipeline = get_pipeline()
        
        # Execute query
        results = pipeline.query(
            query=request.query,
            rewrite=request.rewrite,
            chat_history=request.chat_history,
            broaden_query=request.broaden_query,
            broaden_retry_limit=request.broaden_retry_limit,
            context_chunk_size=request.context_chunk_size,
            rerank=request.rerank,
            compress=request.compress,
            limit=request.limit
        )
        
        # Ensure results is a list
        if not isinstance(results, list):
            results = [results] if results else []
        
        return QueryResponse(
            results=results,
            count=len(results)
        )
    except Exception as e:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        ) from e


@app.get("/api/v1/query", tags=["Retrieval"], response_model=QueryResponse)
async def query_get(
    q: str,
    rewrite: bool = True,
    broaden_query: bool = True,
    rerank: bool = True,
    compress: bool = False,
    limit: int = 30
):
    """
    Simple GET endpoint for querying (convenience method).
    
    Query parameters:
    - q: Search query (required)
    - rewrite: Enable query rewriting (default: true)
    - broaden_query: Enable query broadening (default: true)
    - rerank: Enable reranking (default: true)
    - compress: Enable compression (default: false)
    - limit: Maximum results (default: 30)
    """
    try:
        pipeline = get_pipeline()
        
        results = pipeline.query(
            query=q,
            rewrite=rewrite,
            broaden_query=broaden_query,
            rerank=rerank,
            compress=compress,
            limit=limit
        )
        
        if not isinstance(results, list):
            results = [results] if results else []
        
        return QueryResponse(
            results=results,
            count=len(results)
        )
    except Exception as e:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        ) from e


# Database Management Endpoints
@app.post("/api/v1/database/clean", tags=["Database"], response_model=CleanResponse)
async def clean_database(request: CleanRequest):
    """
    Clean/clear all data from the databases.
    
    This will:
    - Clear all data from Milvus collection
    - Clear all data from Neo4j database (for the specified collection)
    
    WARNING: This operation is irreversible!
    """
    try:
        result = clear_all_databases(
            collection_name=request.collection_name,
            skip_neo4j_on_auth_failure=request.skip_neo4j_on_auth_failure
        )
        
        return CleanResponse(
            success=result["success"],
            milvus=result["milvus"],
            neo4j=result["neo4j"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clean database: {str(e)}"
        ) from e


@app.delete("/api/v1/database/clean", tags=["Database"], response_model=CleanResponse)
async def clean_database_delete(collection_name: str = "vector_rag"):
    """
    Clean database using DELETE method (convenience endpoint).
    
    WARNING: This operation is irreversible!
    """
    try:
        result = clear_all_databases(
            collection_name=collection_name,
            skip_neo4j_on_auth_failure=True
        )
        
        return CleanResponse(
            success=result["success"],
            milvus=result["milvus"],
            neo4j=result["neo4j"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clean database: {str(e)}"
        ) from e


@app.post("/api/v1/database/clean/milvus", tags=["Database"])
async def clean_milvus(collection_name: str = "vector_rag"):
    """
    Clean only Milvus collection.
    
    WARNING: This operation is irreversible!
    """
    try:
        result = clear_milvus_collection(collection_name=collection_name)
        
        if result["success"]:
            return {
                "success": True,
                "message": f"Cleared {result['deleted_count']} entities from Milvus",
                "deleted_count": result["deleted_count"]
            }
        else:
            raise HTTPException(
                status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Unknown error")
            )
    except Exception as e:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clean Milvus: {str(e)}"
        ) from e


@app.post("/api/v1/database/clean/neo4j", tags=["Database"])
async def clean_neo4j(collection_name: Optional[str] = None):
    """
    Clean Neo4j database.
    
    If collection_name is provided, only deletes nodes with that collection.
    If None, deletes ALL data in Neo4j.
    
    WARNING: This operation is irreversible!
    """
    try:
        result = clear_neo4j_database(collection_name=collection_name)
        
        if result["success"]:
            return {
                "success": True,
                "message": f"Cleared {result['deleted_nodes']} nodes and {result['deleted_relationships']} relationships from Neo4j",
                "deleted_nodes": result["deleted_nodes"],
                "deleted_relationships": result["deleted_relationships"]
            }
        else:
            raise HTTPException(
                status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Unknown error")
            )
    except Exception as e:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clean Neo4j: {str(e)}"
        ) from e


@app.get("/api/v1/database/status", tags=["Database"], response_model=StatusResponse)
async def get_database_status():
    """
    Get status information about both databases.
    
    Returns:
    - Milvus connection status and collections
    - Neo4j connection status, node count, and relationship count
    """
    try:
        db_status = get_db_status()
        return StatusResponse(
            milvus=db_status["milvus"],
            neo4j=db_status["neo4j"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get database status: {str(e)}"
        ) from e


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "HybirdRAG API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "add_document": "/api/v1/documents",
            "add_documents_batch": "/api/v1/documents/batch",
            "query": "/api/v1/query",
            "clean_database": "/api/v1/database/clean",
            "database_status": "/api/v1/database/status"
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HybirdRAG REST API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

