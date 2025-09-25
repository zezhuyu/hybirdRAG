from typing import Any, Dict, List, Optional
import sys
import os

from openai import OpenAI
from pymilvus import MilvusClient

from .VectorRAG.pipeline import VectorRAGPipeline
from .VectorRAG.text_processing import ContextualChunker, LateChunker

from .GraphRAG.convertObj import OpenAILLMWrapper, MLModelEmbeddingWrapper
from .GraphRAG.pipeline import GraphRAGPipeline
from .GraphRAG.store import GraphRAGStore, NEO4J_AVAILABLE

from comp import MLModelClient
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

rewrite_prompt = """
You are a helpful assistant that rewrites the prompt to be more specific and concise.
given the user chat history and query, rewrite the prompt to be more specific and concise.

# Chat History:
{chat_history}
# Query:
"""

class HybridRAGPipeline:
    def __init__(
        self,
        service_host: str,
        milvus_host: str,
        openai_host: str,
        openai_api_key: str,
        *,
        neo4j_config: Optional[Dict[str, Any]] = None,
        graph_vector_store_kwargs: Optional[Dict[str, Any]] = None,
        milvus_client_kwargs: Optional[Dict[str, Any]] = None,
        vector_collection_name: str = "vector_rag",
        splitter_chunk_size: int = 1024,
        splitter_overlap: int = 50,
    ) -> None:
        if neo4j_config is None:
            raise ValueError(
                "neo4j_config is required to initialise the GraphRAG store with Neo4j credentials."
            )
        if graph_vector_store_kwargs is None:
            raise ValueError(
                "graph_vector_store_kwargs is required so graph embeddings persist in Milvus."
            )

        self.service = MLModelClient(host=service_host)

        milvus_kwargs: Dict[str, Any] = {"host": milvus_host}
        if milvus_client_kwargs:
            milvus_kwargs.update(milvus_client_kwargs)
        self.milvus = MilvusClient(**milvus_kwargs)

        self.vector_rag = VectorRAGPipeline(
            milvus=self.milvus,
            embedding=self.service,
            collection_name=vector_collection_name,
        )

        # Ensure base_url has /v1 suffix for Ollama compatibility
        base_url = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1/"
        if not base_url.endswith('/v1') and not base_url.endswith('/v1/'):
            base_url = base_url.rstrip('/') + '/v1/'
        self.openai = OpenAI(api_key=openai_api_key, base_url=base_url)
        self.graph_llm = OpenAILLMWrapper(client=self.openai)
        
        # Set the LLM in global settings for all llama-index operations
        from llama_index.core import Settings
        Settings.llm = self.graph_llm

        # Configure Neo4j graph store
        if NEO4J_AVAILABLE:
            # Parse Neo4j configuration
            neo4j_url = neo4j_config.get("url", "bolt://localhost:7687")
            neo4j_username = neo4j_config.get("username", "neo4j")
            neo4j_password = neo4j_config.get("password", "password")
            neo4j_database = neo4j_config.get("database", "neo4j")
            
            self.graph_store = GraphRAGStore(
                uri=neo4j_url,
                username=neo4j_username,
                password=neo4j_password,
                database=neo4j_database,
                summarizer_llm=self.graph_llm
            )
        else:
            # Fallback to simple graph store
            self.graph_store = GraphRAGStore(summarizer_llm=self.graph_llm)

        # Create embedding wrapper for GraphRAG
        embedding_wrapper = MLModelEmbeddingWrapper(self.service)
        
        # Use the same Milvus collection for GraphRAG as the main vector store
        graph_milvus_kwargs = {
            "host": milvus_host,
            "collection_name": vector_collection_name,  # Use same collection
            "dim": 768,  # Same dimension as main vector store
        }
        if milvus_client_kwargs:
            graph_milvus_kwargs.update(milvus_client_kwargs)
        
        self.graph_rag = GraphRAGPipeline(
            llm=self.graph_llm,
            graph_store=self.graph_store,
            milvus_kwargs=graph_milvus_kwargs,
            embedding_model=embedding_wrapper,
            milvus_client=self.milvus,  # Pass the Milvus client for syncing
        )

        self.splitter = SentenceSplitter(
            chunk_size=splitter_chunk_size,
            chunk_overlap=splitter_overlap,
        )
        self.late_chunker = LateChunker()
        self.context_chunker = ContextualChunker()

    def query(
        self,
        query: str,
        chat_history: Optional[List[str]] = None,
        context_chunk_size: int = 128,
        rerank: bool = True,
        compress: bool = True,
        limit: int = 10,
    ):
        chat_history = chat_history or []
        rewritten_prompt = rewrite_prompt.format(
            chat_history="\n".join(chat_history), query=query
        )
        try:
            rewritten = self.openai.chat.completions.create(
                model="qwen3:0.6b",
                messages=[
                    {"role": "system", "content": rewritten_prompt},
                    {"role": "user", "content": query},
                ],
            )
            query_text = (
                rewritten.choices[0].message.content
                if rewritten and rewritten.choices
                else query
            )
        except Exception:
            query_text = query

        vector_results = self.vector_rag.query(query_text, limit=limit)
        
        # Try to get graph answer, but handle gracefully if it fails
        try:
            graph_answer = self.graph_rag.query(query_text)
        except Exception as e:
            print(f"⚠️  GraphRAG query failed: {e}")
            print("Continuing with vector-only search...")
            graph_answer = ""

        # Handle HybridHits objects from Milvus
        vector_texts = []
        for hit in vector_results:
            # HybridHits is iterable and contains Hit objects
            for hit_item in hit:
                if hasattr(hit_item, 'entity') and hasattr(hit_item.entity, 'content'):
                    vector_texts.append(hit_item.entity.content)
                elif isinstance(hit_item, dict) and 'entity' in hit_item:
                    vector_texts.append(hit_item['entity'].get('content', ''))
        
        vector_texts = [text for text in vector_texts if text]
        
        # Debug output
        print(f"🔍 Debug: Found {len(vector_texts)} vector texts")
        if vector_texts:
            print(f"🔍 Debug: First vector text: {vector_texts[0][:100]}...")

        if context_chunk_size > 0 and vector_texts:
            self.context_chunker.max_chunk_size = context_chunk_size
            refined_chunks = self.context_chunker.process_results(vector_texts)
            vector_texts = [chunk["text"] for chunk in refined_chunks if "text" in chunk]

        if rerank and vector_texts:
            vector_texts = self.service.rerank_documents(vector_texts, query_text)

        results = [graph_answer] + vector_texts

        if compress:
            compressed_result = self.service.compress_prompt(query_text, results)
            print(f"🔍 Debug: Compression result: {compressed_result[:100] if compressed_result else 'None'}...")
            # If compression fails or returns empty, return the original results
            if compressed_result and compressed_result.strip():
                return [compressed_result]
            else:
                print("⚠️  Compression failed or returned empty, returning original results")
                return results
        return results

    def add_document(self, document: Dict[str, Any]) -> None:
        metadata = {k: v for k, v in document.items() if k != "content"}
        text = document.get("content", "")
        if not text:
            return

        vector_docs = [
            {**metadata, "content": chunk}
            for chunk in self.late_chunker.chunk_document(text)
        ]
        if vector_docs:
            self.vector_rag.add_document(vector_docs)

        graph_nodes = self._build_graph_nodes(text, metadata)
        if graph_nodes:
            self.graph_rag.build_index(graph_nodes)
            self.graph_rag.build_communities()

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        all_vector_docs: List[Dict[str, Any]] = []
        all_nodes: List[TextNode] = []

        for document in documents:
            metadata = {k: v for k, v in document.items() if k != "content"}
            text = document.get("content", "")
            if not text:
                continue

            all_vector_docs.extend(
                {**metadata, "content": chunk}
                for chunk in self.late_chunker.chunk_document(text)
            )
            all_nodes.extend(self._build_graph_nodes(text, metadata))

        if all_vector_docs:
            self.vector_rag.add_document(all_vector_docs)

        if all_nodes:
            self.graph_rag.build_index(all_nodes)
            self.graph_rag.build_communities()

    def _build_graph_nodes(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[TextNode]:
        chunks = self.splitter.split_text(text)
        return [TextNode(text=chunk, metadata=metadata) for chunk in chunks if chunk]


def main():
    """Main CLI entry point for HybridRAG pipeline."""
    import argparse
    import sys
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="HybridRAG Pipeline")
    parser.add_argument("--env", type=Path, help="Environment file path")
    parser.add_argument("--query", type=str, help="Query to process")
    parser.add_argument("--limit", type=int, default=10, help="Number of results to return")
    
    args = parser.parse_args()
    
    if args.env:
        from .cli_utils import load_env_file
        load_env_file(args.env)
    
    try:
        from .cli_utils import build_pipeline_from_env
        pipeline = build_pipeline_from_env()
        
        if args.query:
            print(f"Processing query: {args.query}")
            results = pipeline.query(args.query, limit=args.limit)
            print(f"Found {len(results)} results")
            for i, result in enumerate(results):
                print(f"{i+1}. {result}")
        else:
            print("HybridRAG Pipeline initialized successfully!")
            print("Use --query 'your question' to search documents")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure all required services (Milvus, Neo4j) are running and environment variables are set.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
