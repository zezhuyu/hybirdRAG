"""Shared helpers for CLI scripts interacting with the Hybrid RAG stack."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from .pipeline import HybridRAGPipeline


def load_env_file(env_path: Path) -> None:
    """Populate ``os.environ`` from a simple ``.env`` file."""

    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ[key.strip()] = value.strip()  # Override existing values


def _require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Environment variable '{key}' must be set.")
    return value


def _parse_host_port(value: str, default_port: Optional[str] = None) -> Dict[str, str]:
    if ":" in value:
        host, port = value.split(":", 1)
        return {"host": host, "port": port}
    if default_port:
        return {"host": value, "port": default_port}
    return {"host": value}


def build_pipeline_from_env() -> HybridRAGPipeline:
    """Instantiate ``HybridRAGPipeline`` using environment variables."""

    service_host = _require_env("SERVICE_HOST")
    milvus_host = _require_env("MILVUS_HOST")
    openai_host = _require_env("OPENAI_BASE_URL")
    openai_key = _require_env("OPENAI_API_KEY")

    neo4j_host = _require_env("NEO4J_HOST")
    neo4j_user = _require_env("NEO4J_USERNAME")
    neo4j_password = _require_env("NEO4J_PASSWORD")
    neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")

    bolt_port = os.getenv("NEO4J_BOLT_PORT", "7687")
    neo4j_url = os.getenv("NEO4J_URL")
    if not neo4j_url:
        host_parts = _parse_host_port(neo4j_host, default_port="7474")
        neo4j_url = f"bolt://{host_parts['host']}:{bolt_port}"

    graph_collection = os.getenv("GRAPH_VECTOR_COLLECTION", "graph_rag_embeddings")
    graph_dim = int(os.getenv("GRAPH_VECTOR_DIM", "1024"))

    milvus_parts = _parse_host_port(milvus_host)
    if "host" not in milvus_parts:
        raise RuntimeError("MILVUS_HOST must include a hostname")

    # Create URI format for MilvusClient
    if "port" in milvus_parts:
        milvus_uri = f"http://{milvus_parts['host']}:{milvus_parts['port']}"
    else:
        milvus_uri = f"http://{milvus_parts['host']}:19530"  # Default port
    
    milvus_kwargs: Dict[str, Any] = {"uri": milvus_uri}
    milvus_user = os.getenv("MILVUS_USERNAME")
    milvus_password = os.getenv("MILVUS_PASSWORD")
    if milvus_user:
        milvus_kwargs["user"] = milvus_user
    if milvus_password:
        milvus_kwargs["password"] = milvus_password

    graph_vector_store_kwargs: Dict[str, Any] = {
        "host": milvus_parts["host"],
        "collection_name": graph_collection,
        "dim": graph_dim,
    }
    if "port" in milvus_parts:
        graph_vector_store_kwargs["port"] = milvus_parts["port"]
        graph_vector_store_kwargs["uri"] = f"{milvus_parts['host']}:{milvus_parts['port']}"
    if milvus_user:
        graph_vector_store_kwargs["user"] = milvus_user
    if milvus_password:
        graph_vector_store_kwargs["password"] = milvus_password

    return HybridRAGPipeline(
        service_host=service_host,
        milvus_host=milvus_parts["host"],
        openai_host=openai_host,
        openai_api_key=openai_key,
        neo4j_config={
            "url": neo4j_url,
            "username": neo4j_user,
            "password": neo4j_password,
            "database": neo4j_database,
        },
        graph_vector_store_kwargs=graph_vector_store_kwargs,
        milvus_client_kwargs=milvus_kwargs,
    )


def main():
    """Main CLI entry point for ReqQuest."""
    import sys
    print("ReqQuest CLI - Hybrid RAG System")
    print("Available commands:")
    print("  load_pdf: Load and process PDF documents")
    print("  query_docs: Query documents")
    print("  hybrid-rag: Main hybrid RAG pipeline")
    print("\nUsage examples:")
    print("  python -m HybirdRAG.load_pdf book.pdf --env .env")
    print("  python -m HybirdRAG.query_docs --env .env")
    print("  python -m HybirdRAG.pipeline --env .env")
    return 0

__all__ = ["load_env_file", "build_pipeline_from_env", "main"]
