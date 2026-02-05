# HybirdRAG API Documentation

HybirdRAG provides two server interfaces:
1. **REST API** (`rest-server.py`) - HTTP/REST endpoints via FastAPI
2. **MCP Server** (`mcp-server.py`) - Model Context Protocol for AI assistants

**Default Base URL:** `http://localhost:8000`

**Interactive Docs (Swagger UI):** `http://localhost:8000/docs`

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Document Management](#document-management)
3. [Query/Retrieval](#queryretrieval)
4. [Database Management](#database-management)
5. [Job Management](#job-management)
6. [MCP Tools](#mcp-tools)
7. [Data Models](#data-models)
8. [Configuration](#configuration)

---

## Quick Start

### Start the REST Server

```bash
cd HybirdRAG
python rest-server.py --host 0.0.0.0 --port 8000
```

### Start the MCP Server (for AI assistants)

```bash
cd HybirdRAG
python mcp-server.py
```

### Basic Usage

```bash
# Add a document
curl -X POST http://localhost:8000/api/v1/documents \
  -H "Content-Type: application/json" \
  -d '{"content": "Machine learning is a subset of AI..."}'

# Query documents
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?"}'
```

---

## Document Management

### Add Single Document

**POST** `/api/v1/documents`

Add a single document to the RAG system. The document will be chunked and indexed in both:
- Vector store (Milvus) for semantic search
- Knowledge graph (Neo4j) for entity extraction (if enabled)

**Request Body:**
```json
{
  "content": "The text content of the document",
  "id": "optional-unique-id",
  "title": "Optional Document Title",
  "page": 1,
  "metadata": {
    "author": "John Doe",
    "source": "textbook"
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `content` | string | ✅ Yes | Document text content |
| `id` | string | No | Unique document identifier |
| `title` | string | No | Document title |
| `page` | integer | No | Page number |
| `metadata` | object | No | Additional key-value metadata |

**Response (201 Created):**
```json
{
  "success": true,
  "message": "Document added successfully",
  "document_id": "auto-generated"
}
```

---

### Add Multiple Documents (Batch)

**POST** `/api/v1/documents/batch`

Add multiple documents in batch. This endpoint processes documents asynchronously and returns immediately with a job ID for tracking.

**Request Body:**
```json
{
  "documents": [
    {
      "content": "First document content...",
      "title": "Document 1"
    },
    {
      "content": "Second document content...",
      "title": "Document 2",
      "page": 5
    }
  ]
}
```

**Response (202 Accepted):**
```json
{
  "success": true,
  "message": "Accepted 2 documents for processing",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "count": 2
}
```

---

### Upload PDF Document

**POST** `/api/v1/documents/upload`

Upload a PDF file for processing. The server extracts text and adds it to the RAG system.

**Content-Type:** `multipart/form-data`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | file | ✅ Yes | PDF file to upload |
| `use_ocr` | boolean | No | Enable OCR for scanned PDFs (default: false) |
| `collection_name` | string | No | Collection name for organization |

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@/path/to/document.pdf" \
  -F "use_ocr=false"
```

**Response (202 Accepted):**
```json
{
  "success": true,
  "message": "Accepted file document.pdf for processing",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "count": 1,
  "filename": "document.pdf"
}
```

---

### List Documents

**GET** `/api/v1/documents`

Retrieve documents from the vector store.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collection_name` | string | vector_rag | Collection to query |
| `limit` | integer | 1000 | Maximum documents to return |

**Response:**
```json
{
  "documents": [
    {
      "id": 123456789,
      "content": "Document text content...",
      "title": "Document Title",
      "page": 1,
      "source": "textbook",
      "source_path": "/path/to/file.pdf",
      "add_at": 1706824800
    }
  ]
}
```

---

### Get Document Status

**GET** `/api/v1/documents/status`

Get document count without retrieving content.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collection_name` | string | vector_rag | Collection to query |

**Response:**
```json
{
  "collection_name": "vector_rag",
  "document_count": 1542
}
```

---

## Query/Retrieval

### Query Documents (POST)

**POST** `/api/v1/query`

Query the RAG system to retrieve relevant documents. Supports advanced features like query rewriting, broadening, and reranking.

**Request Body:**
```json
{
  "query": "What is machine learning?",
  "rewrite": true,
  "chat_history": ["Previous question", "Previous answer"],
  "broaden_query": true,
  "broaden_retry_limit": 3,
  "context_chunk_size": 256,
  "rerank": true,
  "compress": false,
  "limit": 30
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | - | ✅ **Required.** Search query |
| `rewrite` | boolean | true | Enable query rewriting for better specificity |
| `chat_history` | array | null | Previous conversation for context-aware rewriting |
| `broaden_query` | boolean | true | Enable query broadening for better coverage |
| `broaden_retry_limit` | integer | 3 | Max retries for query broadening |
| `context_chunk_size` | integer | 256 | Context chunk size for processing |
| `rerank` | boolean | true | Enable result reranking for relevance |
| `compress` | boolean | false | Enable result compression |
| `limit` | integer | 30 | Maximum number of results |

**Response:**
```json
{
  "results": [
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data...",
    "The key difference between supervised and unsupervised learning is..."
  ],
  "count": 2
}
```

---

### Query Documents (GET)

**GET** `/api/v1/query`

Simple GET endpoint for quick queries.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `q` | string | - | ✅ **Required.** Search query |
| `rewrite` | boolean | true | Enable query rewriting |
| `broaden_query` | boolean | true | Enable query broadening |
| `rerank` | boolean | true | Enable reranking |
| `compress` | boolean | false | Enable compression |
| `limit` | integer | 30 | Maximum results |

**Example:**
```bash
curl "http://localhost:8000/api/v1/query?q=What%20is%20machine%20learning&limit=5"
```

---

## Database Management

### Get Database Status

**GET** `/api/v1/database/status`

Get connection status and statistics for both databases.

**Response:**
```json
{
  "milvus": {
    "connected": true,
    "collections": ["vector_rag", "another_collection"],
    "collection_info": {
      "vector_rag": {"row_count": 1542}
    }
  },
  "neo4j": {
    "connected": true,
    "node_count": 3500,
    "relationship_count": 8200
  }
}
```

---

### Clean All Databases

**POST** `/api/v1/database/clean`

⚠️ **WARNING: This operation is irreversible!**

Clear all data from both Milvus and Neo4j.

**Request Body:**
```json
{
  "collection_name": "vector_rag",
  "skip_neo4j_on_auth_failure": true
}
```

**Response:**
```json
{
  "success": true,
  "milvus": {
    "success": true,
    "deleted_count": 1542
  },
  "neo4j": {
    "success": true,
    "deleted_nodes": 3500,
    "deleted_relationships": 8200
  }
}
```

---

### Clean Milvus Only

**POST** `/api/v1/database/clean/milvus`

⚠️ **WARNING: Irreversible operation!**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collection_name` | string | vector_rag | Collection to clean |

**Response:**
```json
{
  "success": true,
  "message": "Cleared 1542 entities from Milvus",
  "deleted_count": 1542
}
```

---

### Clean Neo4j Only

**POST** `/api/v1/database/clean/neo4j`

⚠️ **WARNING: Irreversible operation!**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collection_name` | string | null | If provided, only deletes nodes with that collection. If null, deletes ALL data. |

**Response:**
```json
{
  "success": true,
  "message": "Cleared 3500 nodes and 8200 relationships from Neo4j",
  "deleted_nodes": 3500,
  "deleted_relationships": 8200
}
```

---

## Job Management

Async operations (batch document upload, PDF processing) return job IDs for tracking progress.

### Get Job Status

**GET** `/api/v1/jobs/{job_id}`

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string | Job ID from async operation |

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "count": 15,
  "progress": 100,
  "created_at": "2026-01-22T10:30:00",
  "started_at": "2026-01-22T10:30:01",
  "completed_at": "2026-01-22T10:32:45",
  "error": null
}
```

**Job Status Values:**
| Status | Description |
|--------|-------------|
| `pending` | Job created, waiting to start |
| `processing` | Job is currently running |
| `completed` | Job finished successfully |
| `failed` | Job failed (check `error` field) |

---

### List Jobs

**GET** `/api/v1/jobs`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 50 | Maximum jobs to return |
| `status` | string | null | Filter by status (pending/processing/completed/failed) |

**Response:**
```json
{
  "jobs": [
    {
      "job_id": "550e8400-e29b-41d4-a716-446655440000",
      "status": "completed",
      "count": 15,
      "progress": 100
    }
  ],
  "count": 1,
  "total": 5
}
```

---

## MCP Tools

The MCP server (`mcp-server.py`) exposes tools for AI assistants like Claude.

### Available Tools

| Tool | Description |
|------|-------------|
| `add_document` | Add a single document |
| `add_documents` | Add multiple documents (batch) |
| `query` | Query the RAG system |
| `clean_database` | Clean both databases |
| `clean_milvus` | Clean Milvus only |
| `clean_neo4j` | Clean Neo4j only |
| `get_database_status` | Get database status |
| `get_job_status` | Get async job status |

### MCP Configuration Example

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "hybirdrag": {
      "command": "python",
      "args": ["/path/to/HybirdRAG/mcp-server.py"],
      "env": {
        "MILVUS_URI": "http://localhost:19530",
        "NEO4J_URI": "bolt://localhost:7687"
      }
    }
  }
}
```

---

## Data Models

### Document Schema (Milvus)

| Field | Type | Description |
|-------|------|-------------|
| `id` | INT64 | Auto-generated primary key |
| `content` | VARCHAR(10000) | Document text content |
| `title` | VARCHAR(1000) | Document title |
| `page` | INT64 | Page number |
| `source` | VARCHAR(500) | Source identifier |
| `source_path` | VARCHAR(1000) | Source file path |
| `vector` | FLOAT_VECTOR(1024) | Embedding vector |
| `sparse` | SPARSE_FLOAT_VECTOR | BM25 sparse vector |
| `add_at` | INT64 | Timestamp when added |
| `chapter` | INT64 | Chapter number |
| `section` | INT64 | Section number |

### Knowledge Graph (Neo4j)

The GraphRAG component extracts entities and relationships:

- **Nodes:** Entities extracted from documents (concepts, people, places, etc.)
- **Relationships:** Connections between entities
- **Properties:** Metadata including collection name, source, etc.

---

## Configuration

### Environment Variables

Create a `.env` file in the HybirdRAG directory:

```bash
# Milvus Configuration
MILVUS_URI=http://localhost:19530
COLLECTION_NAME=vector_rag

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Ollama (for embeddings and LLM)
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
LLM_MODEL=llama3.2

# Optional: OpenAI for reranking/compression
OPENAI_API_KEY=sk-...
```

### Docker Compose (Recommended)

```bash
cd HybirdRAG
docker-compose up -d  # Starts Milvus and Neo4j
```

---

## Error Handling

All endpoints return errors in a consistent format:

**Error Response (4xx/5xx):**
```json
{
  "detail": "Error message describing what went wrong"
}
```

**Common HTTP Status Codes:**

| Code | Description |
|------|-------------|
| `200` | Success |
| `201` | Created (document added) |
| `202` | Accepted (async job queued) |
| `400` | Bad Request (invalid input) |
| `404` | Not Found (job/resource not found) |
| `500` | Internal Server Error |

---

## Examples

### Complete Workflow: Add PDF and Query

```bash
# 1. Upload a PDF document
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@lecture_notes.pdf" \
  -F "collection_name=csc101"

# Response: {"job_id": "abc123..."}

# 2. Check job status
curl http://localhost:8000/api/v1/jobs/abc123...

# 3. Query when job is completed
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is recursion?",
    "rewrite": true,
    "rerank": true,
    "limit": 5
  }'
```

### Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000"

# Add document
response = requests.post(f"{BASE_URL}/api/v1/documents", json={
    "content": "Machine learning is a subset of AI that enables systems to learn from data.",
    "title": "Introduction to ML"
})
print(response.json())

# Query
response = requests.post(f"{BASE_URL}/api/v1/query", json={
    "query": "What is machine learning?",
    "limit": 5
})
results = response.json()
for result in results["results"]:
    print(result)
```

---

## Health Check

**GET** `/health`

Simple health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "HybirdRAG API"
}
```

---

## API Info

**GET** `/`

Get API information and available endpoints.

**Response:**
```json
{
  "service": "HybirdRAG API",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "docs": "/docs",
    "add_document": "/api/v1/documents",
    "add_documents_batch": "/api/v1/documents/batch",
    "upload_document": "/api/v1/documents/upload",
    "document_status": "/api/v1/documents/status",
    "query": "/api/v1/query",
    "clean_database": "/api/v1/database/clean",
    "database_status": "/api/v1/database/status",
    "job_status": "/api/v1/jobs/{job_id}"
  }
}
```
