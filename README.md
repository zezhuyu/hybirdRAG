# HybirdRAG

A **hybrid RAG (Retrieval-Augmented Generation)** system that combines **vector search** (Milvus) and **graph-based retrieval** (Neo4j) to improve answer quality for complex, multi-hop questions. It supports optional query rewriting, query broadening, reranking, and context compression.

---

## General Idea

HybirdRAG retrieves context in two ways:

1. **Vector RAG** — Dense embeddings (e.g. BGE-M3) stored in Milvus for semantic similarity search.
2. **Graph RAG** — Documents are chunked, summarized, and stored as a knowledge graph in Neo4j. Community detection and summarization help answer questions that need relationships and multi-hop reasoning.

The pipeline can use both retrieval paths and merge or prefer results (e.g. use vector results and optionally rerank them; use graph answers when available). Optional steps include:

- **Query rewriting** — Make the query more specific using an LLM.
- **Query broadening** — Generate related queries for better recall.
- **Reranking** — Score and reorder retrieved chunks (gRPC ML service or HTTP vLLM reranker).
- **Compression** — Compress retrieved context with an LLM (e.g. LLMLingua) before use.

---

## Components

| Component | Description |
|-----------|-------------|
| **HybirdRAG (pipeline)** | Main orchestration: `HybridRAGPipeline` in `HybirdRAG/pipeline.py`. Wires Vector RAG, Graph RAG, rewriting, broadening, reranking, and compression. |
| **VectorRAG** | Milvus-based vector store, chunking, embedding (OpenAI-compatible or gRPC). See `HybirdRAG/VectorRAG/`. |
| **GraphRAG** | Neo4j knowledge graph, community building, summarization. See `HybirdRAG/GraphRAG/`. |
| **comp** | gRPC client (`HybirdRAG/comp/client.py`) for the ML service: embeddings, rerank, rewrite, compress. Used when the gRPC server is running and (for rerank) when `RERANKER_MODEL_NAME` is not set. |
| **OpenAI-style clients** | `OpenAIEmbeddingClient`, `OpenAIRerankerClient` in `HybirdRAG/VectorRAG/convertObj.py` for HTTP-based embedding/rerank (e.g. Ollama, vLLM). |
| **REST API** | FastAPI app in `rest-server.py`; unified entrypoint is `api.py` (REST + MCP). |
| **MCP** | MCP-over-HTTP tools in `mcp-server.py` for document add, query, status, clean. |
| **ML gRPC server** | Optional Docker service in `docker/` that runs embedding, reranker, rewriter, and compressor models. |

**Reranker selection:**

- If **`RERANKER_MODEL_NAME`** is set in the environment (non-empty), the pipeline uses **`OpenAIRerankerClient`** (HTTP vLLM rerank endpoint, e.g. `http://localhost:8080/engines/vllm/rerank`).
- Otherwise it uses **`self.service`** (gRPC `MLModelClient.rerank_documents`).

---

## Deployment

### Prerequisites

- Docker and Docker Compose
- Python 3.8+ (for local run)
- (Optional) GPU for Ollama and/or ML gRPC server

### 1. Start databases (Milvus + Neo4j)

Create the shared network and start Milvus (etcd, MinIO, Milvus) and Neo4j:

```bash
cd dependencies
docker compose -f db-compose.yml up -d
```

Wait until Milvus and Neo4j are healthy. Defaults:

- Milvus: `localhost:19530`
- Neo4j: `localhost:7474` (browser), `localhost:7687` (Bolt). Default auth: `neo4j` / `GraphRAG`.

### 2. (Optional) Start Ollama for embeddings/LLMs

If you use Ollama for embeddings and/or chat:

```bash
docker compose -f dependencies/ollama-compose.yml up -d
```

Default: `http://localhost:11434`. Set `OPENAI_BASE_URL` (and optionally `OPENAI_API_KEY`) to point to it.

### 3. (Optional) Start ML gRPC server

If you use the gRPC ML service for embedding/rerank/rewrite/compress:

```bash
docker compose -f dependencies/docker-model-runner.yml up -d
```

Or build/run from `docker/`:

```bash
cd docker
docker build -t ml-grpc-server .
docker run -p 50051:50051 --gpus all ml-grpc-server
```

Set `SERVICE_HOST=localhost:50051` (or `ml_grpc_server:50051` when the app runs in Docker on the same Compose network).

### 4. Run HybirdRAG API (Docker)

From the repo root, using the main Compose file:

```bash
# Ensure db-network exists (from step 1)
docker network create db-network  # if not already created

docker compose up -d
```

This builds and runs the HybirdRAG service on port **8000**. Health: `GET http://localhost:8000/health`.

### 5. Run HybirdRAG locally (no Docker for app)

```bash
# Create virtualenv and install
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e .

# Copy env and edit (see below)
cp .env.example .env

# Run API
python api.py --host 0.0.0.0 --port 8000
```

Or use the CLI:

```bash
hybrid-rag --env .env --query "Your question here" --limit 10
```

---

## Environment (.env) setup

Copy the example and adjust for your setup:

```bash
cp .env.example .env
```

### Core services

| Variable | Description | Example |
|----------|-------------|---------|
| `SERVICE_HOST` | gRPC ML service (embedding/rerank/rewrite/compress). Omit or leave unused if you use only HTTP (Ollama/vLLM). | `localhost:50051` |
| `MILVUS_HOST` | Milvus server. | `localhost:19530` |
| `NEO4J_HOST` | Neo4j host. | `localhost` |
| `NEO4J_BOLT_PORT` | Neo4j Bolt port. | `7687` |
| `NEO4J_USERNAME` | Neo4j user. | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password. | `GraphRAG` (match db-compose) |
| `NEO4J_DATABASE` | Neo4j database. | `neo4j` |
| `NEO4J_URL` | Full Bolt URL (optional). If not set, built from host and port. | `bolt://localhost:7687` |

### LLM / embedding / reranker (OpenAI-compatible HTTP)

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_BASE_URL` | Base URL for OpenAI-compatible API (Ollama, vLLM, etc.). | `http://localhost:11434/v1` or `http://localhost:8080` |
| `OPENAI_API_KEY` | API key (use `ollama` for Ollama if required). | `ollama` or your key |
| `EMBEDDING_MODEL_NAME` | Model name for embeddings. | `bge-m3:latest` or `ai/qwen3-embedding-vllm` |
| `RERANKER_MODEL_NAME` | If **set and non-empty**, reranking uses HTTP `OpenAIRerankerClient` (e.g. vLLM `/engines/vllm/rerank`). If **unset or empty**, reranking uses gRPC `self.service`. | `ai/qwen3-reranker-vllm:0.6B` or leave empty for gRPC |
| `PROMPT_REWRITE_MODEL` | Model for query rewriting. | `qwen3:0.6b` |
| `GRAPH_CREATE_MODEL` | Model used for Graph RAG (summarization, etc.). | `gpt-oss:latest` |

### Pipeline behavior

| Variable | Description | Example |
|----------|-------------|---------|
| `COLLECTION_NAME` | Milvus collection name. | `vector_rag` |
| `GRAPHRAG_ENABLED` | Enable Graph RAG indexing and retrieval. | `True` or `False` |
| `RERANK_ENABLED` | When `true`, rerank retrieved chunks by relevance before returning (uses gRPC or HTTP reranker per `RERANKER_MODEL_NAME`). | `true` or `false` |
| `COMPRESS_ENABLED` | When `true`, compress retrieved context with an LLM (e.g. LLMLingua) before use. | `true` or `false` |
| `GRAPH_VECTOR_DIM` | Embedding dimension used for the Graph RAG vector store (e.g. in Milvus for graph nodes). | `1024` |
| `EMBEDDING_DIMENSION` | Dimension of embedding vectors produced by the embedding model (must match model output). | `1024` |
| `CHUNK_SIZE` | Target size (in characters) for text chunks when splitting documents. | `256` |
| `CHUNK_OVERLAP` | Number of characters to overlap between consecutive chunks for continuity. | `64` |
| `MIN_CHUNK_TOKENS` | Minimum token count for a chunk; smaller chunks may be merged or dropped. | `50` |

### Minimal .env examples

**Local with Ollama + vLLM reranker (no gRPC):**

```env
MILVUS_HOST=localhost:19530
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
EMBEDDING_MODEL_NAME=bge-m3:latest
RERANKER_MODEL_NAME=ai/qwen3-reranker-vllm:0.6B
NEO4J_HOST=localhost
NEO4J_PASSWORD=GraphRAG
NEO4J_URL=bolt://localhost:7687
COLLECTION_NAME=vector_rag
GRAPHRAG_ENABLED=True
```

Note: Both the embedding client and `OpenAIRerankerClient` use `OPENAI_BASE_URL`. If embeddings and HTTP reranker run on different ports, set `OPENAI_BASE_URL` to the one you use for rerank when `RERANKER_MODEL_NAME` is set (e.g. vLLM at `http://localhost:8080`).

**With gRPC ML service (no HTTP reranker):**

```env
SERVICE_HOST=localhost:50051
MILVUS_HOST=localhost:19530
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
# Leave RERANKER_MODEL_NAME unset or empty to use gRPC rerank
NEO4J_HOST=localhost
NEO4J_PASSWORD=GraphRAG
NEO4J_URL=bolt://localhost:7687
COLLECTION_NAME=vector_rag
GRAPHRAG_ENABLED=True
```

---

## API overview

- **Health:** `GET /health`
- **Add document:** `POST /api/v1/documents` (body: `content`, optional `id`, `title`, `page`, `metadata`)
- **Add multiple:** `POST /api/v1/documents/bulk` (body: `documents: [{ content, ... }]`)
- **Query:** `POST /api/v1/query` (body: `query`, optional `rewrite`, `chat_history`, `broaden_query`, `rerank`, `compress`, `limit`)
- **Document status:** `GET /api/v1/documents/status?collection_name=...`
- **List documents:** `GET /api/v1/documents?collection_name=...&limit=...`
- **MCP tools:** `GET /mcp/tools`, `POST /mcp/tools/call`

See `rest-server.py` and `API_DOCUMENTATION.md` (if present) for full request/response shapes.

---

## Project layout (summary)

```text
HybirdRAG/
├── api.py                 # Unified REST + MCP server entrypoint
├── rest-server.py         # FastAPI REST endpoints
├── mcp-server.py          # MCP tool definitions and handlers
├── compose.yml            # HybirdRAG service (port 8000)
├── Dockerfile             # Image for HybirdRAG API
├── .env.example            # Example environment variables
├── dependencies/
│   ├── db-compose.yml     # Milvus + Neo4j
│   ├── ollama-compose.yml # Ollama (optional)
│   ├── com-compose.yml    # (optional)
│   └── docker-model-runner.yml  # ML gRPC server (optional)
├── docker/                # ML gRPC server (Dockerfile, server.py, run.py)
└── HybirdRAG/
    ├── pipeline.py        # HybridRAGPipeline
    ├── VectorRAG/         # Milvus, chunking, OpenAI/gRPC embedding & rerank
    ├── GraphRAG/          # Neo4j, communities, summarization
    └── comp/              # gRPC client for ML service
```

---

## Quick start checklist

1. Start DBs: `dependencies/db-compose.yml` → Milvus + Neo4j.
2. Copy and edit `.env` from `.env.example` (Milvus, Neo4j, OpenAI base URL, models, `RERANKER_MODEL_NAME` if using HTTP reranker).
3. (Optional) Start Ollama and/or vLLM for embeddings/rerank; or start the gRPC ML server and set `SERVICE_HOST`.
4. Run the API: `docker compose up -d` or `python api.py --host 0.0.0.0 --port 8000`.
5. Add documents via `POST /api/v1/documents` or bulk, then query with `POST /api/v1/query`.

For reranker: set `RERANKER_MODEL_NAME` to use the HTTP reranker (e.g. vLLM); leave it unset to use the gRPC reranker when `SERVICE_HOST` is set.
