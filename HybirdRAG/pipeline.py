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
        service_host: str = None,
        milvus_host: str = None,
        openai_host: str = None,
        openai_api_key: str = None,
        *,
        neo4j_config: Optional[Dict[str, Any]] = None,
        graph_vector_store_kwargs: Optional[Dict[str, Any]] = None,
        milvus_client_kwargs: Optional[Dict[str, Any]] = None,
        vector_collection_name: str = "vector_rag",
        splitter_chunk_size: int = 1024,
        splitter_overlap: int = 50,
    ) -> None:
        # Set default values from environment variables
        service_host = service_host or os.getenv("SERVICE_HOST", "localhost:50051")
        milvus_host = milvus_host or os.getenv("MILVUS_HOST", "localhost:19530")
        openai_host = openai_host or os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "ollama")
        # Set default Neo4j configuration from environment variables
        if neo4j_config is None:
            neo4j_config = {
                "host": os.getenv("NEO4J_HOST", "localhost"),
                "username": os.getenv("NEO4J_USERNAME", "neo4j"),
                "password": os.getenv("NEO4J_PASSWORD", "password"),
                "database": os.getenv("NEO4J_DATABASE", "neo4j"),
                "bolt_port": int(os.getenv("NEO4J_BOLT_PORT", "7687")),
                "url": os.getenv("NEO4J_URL", "bolt://localhost:7687"),
            }
        
        # Set default graph vector store configuration from environment variables
        if graph_vector_store_kwargs is None:
            graph_collection = os.getenv("GRAPH_VECTOR_COLLECTION", "graph_rag_embeddings")
            graph_dim = int(os.getenv("GRAPH_VECTOR_DIM", "1024"))
            
            # Parse milvus host for graph store
            if ":" in milvus_host:
                milvus_host_parts = milvus_host.split(":", 1)
                milvus_host_name = milvus_host_parts[0]
                milvus_port = milvus_host_parts[1]
            else:
                milvus_host_name = milvus_host
                milvus_port = "19530"
            
            graph_vector_store_kwargs = {
                "host": milvus_host_name,
                "collection_name": graph_collection,
                "dim": graph_dim,
                "port": milvus_port,
                "uri": f"{milvus_host_name}:{milvus_port}",
            }
            
            # Add Milvus credentials if available
            milvus_user = os.getenv("MILVUS_USERNAME")
            milvus_password = os.getenv("MILVUS_PASSWORD")
            if milvus_user:
                graph_vector_store_kwargs["user"] = milvus_user
            if milvus_password:
                graph_vector_store_kwargs["password"] = milvus_password
        
        # Set default Milvus client configuration from environment variables
        if milvus_client_kwargs is None:
            # Parse milvus host for client
            if ":" in milvus_host:
                milvus_host_parts = milvus_host.split(":", 1)
                milvus_host_name = milvus_host_parts[0]
                milvus_port = milvus_host_parts[1]
            else:
                milvus_host_name = milvus_host
                milvus_port = "19530"
            
            milvus_uri = f"http://{milvus_host_name}:{milvus_port}"
            milvus_client_kwargs = {"uri": milvus_uri}
            
            # Add Milvus credentials if available
            milvus_user = os.getenv("MILVUS_USERNAME")
            milvus_password = os.getenv("MILVUS_PASSWORD")
            if milvus_user:
                milvus_client_kwargs["user"] = milvus_user
            if milvus_password:
                milvus_client_kwargs["password"] = milvus_password
        self.service = MLModelClient(host=service_host)
        # Use the configured milvus_client_kwargs directly
        self.milvus = MilvusClient(**milvus_client_kwargs)

        self.vector_rag = VectorRAGPipeline(
            milvus=self.milvus,
            embedding=self.service,
            collection_name=vector_collection_name,
        )

        base_url = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1/"
        if not base_url.endswith('/v1') and not base_url.endswith('/v1/'):
            base_url = base_url.rstrip('/') + '/v1/'
        self.openai = OpenAI(api_key=openai_api_key, base_url=base_url)
        self.graph_llm = OpenAILLMWrapper(client=self.openai)
        
        # Set the LLM in global settings for all llama-index operations
        from llama_index.core import Settings
        Settings.llm = self.graph_llm

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
            model = os.getenv("PROMPT_REWRITE_MODEL", "qwen3:0.6b")
            rewritten = self.openai.chat.completions.create(
                model=model,
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
            # Only include graph answer if it's not empty
            if not graph_answer or not graph_answer.strip():
                graph_answer = None
        except Exception as e:
            print(f"⚠️  GraphRAG query failed: {e}")
            print("Continuing with vector-only search...")
            graph_answer = None

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
        

        if context_chunk_size > 0 and vector_texts:
            self.context_chunker.max_chunk_size = context_chunk_size
            refined_chunks = self.context_chunker.process_results(vector_texts)
            vector_texts = [chunk["text"] for chunk in refined_chunks if "text" in chunk]

        if rerank and vector_texts:
            vector_texts = self.service.rerank_documents(vector_texts, query_text)

        # Prioritize GraphRAG answer if available
        if graph_answer is not None and graph_answer.strip() and "No relevant information found" not in graph_answer:
            # GraphRAG found a good answer - use it as primary result
            results = [graph_answer]
            # Add top vector results as supplementary information
            if vector_texts:
                results.extend(vector_texts[:2])  # Top 2 vector results as supplement
        else:
            # Fallback to vector results only
            results = vector_texts

        if compress:
            # Only compress the vector texts, not the GraphRAG answer (which can be garbled)
            texts_to_compress = vector_texts if vector_texts else results
            compressed_result = self.service.compress_prompt(query_text, texts_to_compress)
            # If compression fails or returns empty, return the original results
            if compressed_result and compressed_result.strip():
                # Return compressed summary + GraphRAG answer (if available) + top vector results
                final_results = [compressed_result]
                if graph_answer is not None:
                    final_results.append(graph_answer)
                final_results.extend(vector_texts[:2])  # Top 2 vector results
                return final_results
            else:
                return results
        return results

    # def fast_query(
    #     self, 
    #     query: str, 
    #     limit: int = 5,
    #     use_vector: bool = True,
    #     use_graph: bool = True,
    #     max_communities: int = 3
    # ) -> Dict[str, Any]:
    #     """
    #     🚀 Ultra-fast hybrid query optimized for 1-2 second retrieval.
    #     Returns raw data perfect for downstream LLM processing.
    #     Skips expensive LLM processing during retrieval.
    #     """
    #     import time
    #     from typing import Dict, Any
        
    #     start_time = time.time()
        
    #     result = {
    #         "query": query,
    #         "vector_results": [],
    #         "graph_results": [],
    #         "total_retrieval_time": 0,
    #         "performance_breakdown": {},
    #         "optimization_note": "Fast retrieval - raw data for LLM processing"
    #     }
        
    #     # 🚀 Vector RAG (skip expensive query rewriting)
    #     vector_time = 0
    #     if use_vector:
    #         vector_start = time.time()
    #         try:
    #             # Use original query directly for speed
    #             vector_results = self.vector_rag.query(query, limit=min(limit, 8))
                
    #             # Process vector results efficiently
    #             vector_texts = []
    #             for hit in vector_results:
    #                 for hit_item in hit:
    #                     if hasattr(hit_item, 'entity') and hasattr(hit_item.entity, 'content'):
    #                         vector_texts.append({
    #                             "content": hit_item.entity.content,
    #                             "score": getattr(hit_item, 'score', 0.0),
    #                             "source": getattr(hit_item.entity, 'source', 'unknown')
    #                         })
    #                     elif isinstance(hit_item, dict) and 'content' in hit_item:
    #                         vector_texts.append({
    #                             "content": hit_item['content'],
    #                             "score": hit_item.get('score', 0.0),
    #                             "source": hit_item.get('source', 'unknown')
    #                         })
                
    #             result["vector_results"] = vector_texts[:limit]
    #             vector_time = time.time() - vector_start
                
    #         except Exception as e:
    #             print(f"⚠️  Vector RAG failed: {e}")
    #             result["vector_results"] = []
        
    #     # 🚀 Graph RAG (fast mode - no LLM processing)
    #     graph_time = 0
    #     if use_graph and self.graph_rag:
    #         graph_start = time.time()
    #         try:
    #             graph_data = self.graph_rag.fast_query(query, max_communities=max_communities)
    #             result["graph_results"] = graph_data.get("communities", [])
    #             result["graph_entities"] = graph_data.get("entities", [])
                
    #             graph_time = time.time() - graph_start
                
    #         except Exception as e:
    #             print(f"⚠️  Graph RAG failed: {e}")
    #             result["graph_results"] = []
        
    #     result["total_retrieval_time"] = time.time() - start_time
    #     result["performance_breakdown"] = {
    #         "vector_time": vector_time,
    #         "graph_time": graph_time,
    #         "total_time": result["total_retrieval_time"]
    #     }
        
    #     return result

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
            # 🚀 For single documents, use smart rebuilding to avoid unnecessary work
            self._rebuild_communities_if_needed()

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
            # 🚀 Generate communities immediately during document loading (not query time!)
            self.graph_rag.build_communities()

    def _build_graph_nodes(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[TextNode]:
        chunks = self.splitter.split_text(text)
        return [TextNode(text=chunk, metadata=metadata) for chunk in chunks if chunk]

    def _rebuild_communities_if_needed(self):
        """Only rebuild communities if the graph has changed significantly."""
        try:
            if hasattr(self.graph_rag.graph_store, '_should_rebuild_communities'):
                if self.graph_rag.graph_store._should_rebuild_communities():
                    # Rebuilding communities silently
                    self.graph_rag.graph_store.invalidate_communities()
                    self.graph_rag.build_communities()
            else:
                # Fallback to always rebuild if change detection not available
                print("⚠️  Change detection not available - rebuilding communities")
                self.graph_rag.build_communities()
        except Exception as e:
            print(f"⚠️  Error in community rebuild check: {e}")
            # Fallback to rebuild
            self.graph_rag.build_communities()


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
