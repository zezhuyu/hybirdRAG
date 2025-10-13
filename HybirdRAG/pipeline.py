from typing import Any, Dict, List, Optional
import sys
import os
import ast
import re

from openai import OpenAI
from pymilvus import MilvusClient

from .VectorRAG.pipeline import VectorRAGPipeline, DIMENSION
from .VectorRAG.text_processing import ContextualChunker, LateChunker

from .GraphRAG.convertObj import OpenAILLMWrapper, MLModelEmbeddingWrapper, OpenAIEmbeddingsWrapper
from .VectorRAG.convertObj import OpenAIEmbeddingClient
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

        base_url = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1/"
        if not base_url.endswith('/v1') and not base_url.endswith('/v1/'):
            base_url = base_url.rstrip('/') + '/v1/'
        self.openai = OpenAI(api_key=openai_api_key, base_url=base_url)
        self.graph_llm = OpenAILLMWrapper(client=self.openai)

        self.vector_rag = VectorRAGPipeline(
            milvus=self.milvus,
            embedding=OpenAIEmbeddingClient(self.openai),
            collection_name=vector_collection_name,
        )
        
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

        embedding_wrapper = OpenAIEmbeddingsWrapper(self.openai)
        
        # Use the same Milvus collection for GraphRAG as the main vector store
        graph_milvus_kwargs = {
            "host": milvus_host,
            "collection_name": vector_collection_name,  # Use same collection
            "dim": DIMENSION,  # Same dimension as main vector store
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

    def _broaden_query_with_retry(self, query_text: str, retry_limit: int = 3) -> Optional[List[str]]:
        """Broaden query with retry mechanism for better reliability."""
        broaden_prompt = """Create 3-5 related search queries for: "{query}"

Examples:
"What happened to the main character?" → ["How did the main character develop?", "What challenges did the main character face?", "What was the main character's background?", "How did the main character change?"]

"Tell me about the setting" → ["What is the environment like?", "Where does the story take place?", "What are the key locations?", "How does the setting affect the plot?"]

"How did the character feel?" → ["What was the character's emotional reaction?", "How did the character's feelings change?", "What did the character experience?", "What were the character's thoughts?"]

Return ONLY a Python list of strings, no explanations, no other text: ["query1", "query2", "query3"]"""
        
        for attempt in range(retry_limit):
            try:
                broadened_prompt = broaden_prompt.format(query=query_text)
                model = os.getenv("PROMPT_REWRITE_MODEL", "qwen3:0.6b")
                rewritten = self.openai.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a query expansion assistant. Return ONLY a Python list of strings. No thinking, no explanations, no other text."},
                        {"role": "user", "content": broadened_prompt},
                    ],
                )
                response_text = (
                    rewritten.choices[0].message.content
                    if rewritten and rewritten.choices
                    else None
                )
                
                if not response_text:
                    print(f"⚠️ Attempt {attempt + 1}: Empty response from LLM")
                    continue
                
                # Clean and parse the response
                broadened_queries = self._parse_broadened_queries(response_text)
                if broadened_queries:
                    return broadened_queries
                else:
                    print(f"⚠️ Attempt {attempt + 1}: Failed to parse valid queries")
                    
            except Exception as e:
                print(f"⚠️ Attempt {attempt + 1}: Exception - {e}")
                if attempt == retry_limit - 1:
                    print(f"❌ All {retry_limit} attempts failed")
                continue
        
        return None

    def _parse_broadened_queries(self, response_text: str) -> Optional[List[str]]:
        """Parse broadened queries from LLM response."""
        try:
            # Clean the response text
            response_text = response_text.strip()
            
            # Remove thinking tags if present
            if '<think>' in response_text and '</think>' in response_text:
                start = response_text.find('</think>') + len('</think>')
                response_text = response_text[start:].strip()
            
            # Remove any markdown code blocks if present
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            
            # Extract the list from the response using regex
            list_pattern = r'\[.*?\]'
            matches = re.findall(list_pattern, response_text, re.DOTALL)
            if matches:
                response_text = matches[0]
            
            broadened_queries = ast.literal_eval(response_text)
            if isinstance(broadened_queries, list) and all(isinstance(q, str) for q in broadened_queries):
                return broadened_queries
            else:
                return None
                
        except (ValueError, SyntaxError) as e:
            print(f"⚠️ Parsing error: {e}")
            return None

    def _clean_query_text(self, query_text: str) -> str:
        """Clean query text to ensure it's safe for search."""
        if not query_text or not isinstance(query_text, str):
            return ""
        
        # Remove any thinking tags that might have leaked through
        if '<think>' in query_text and '</think>' in query_text:
            start = query_text.find('</think>') + len('</think>')
            query_text = query_text[start:].strip()
        
        # Remove any markdown code blocks
        if query_text.startswith('```'):
            lines = query_text.split('\n')
            query_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else ""
        
        query_text = re.sub(r'\([^)]*shortened[^)]*\)', '', query_text, flags=re.IGNORECASE)
        query_text = re.sub(r'\([^)]*concise[^)]*\)', '', query_text, flags=re.IGNORECASE)
        
        # Replace multiple line breaks and spaces with single spaces
        query_text = re.sub(r'\s+', ' ', query_text)
        
        # Remove any non-printable characters except basic punctuation
        query_text = ''.join(char for char in query_text if char.isprintable() or char.isspace())
        query_text = query_text.strip()
        
        # Limit query length to prevent issues (reduce from 10000 to 2000)
        if len(query_text) > 2000:
            query_text = query_text[:2000]
        
        # Ensure query ends properly (remove trailing punctuation issues)
        query_text = query_text.rstrip('.,;:!?')
        
        return query_text

    def query(
        self,
        query: str,
        rewrite: bool = True,
        chat_history: Optional[List[str]] = None,
        broaden_query: bool = True,
        broaden_retry_limit: int = 3,
        context_chunk_size: int = 256,
        rerank: bool = True,
        compress: bool = False,
        limit: int = 20,
    ):
        query_text = query
        if rewrite:
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

        if broaden_query:
            # Store original query as backup
            original_query_text = query_text
            broadened_queries = self._broaden_query_with_retry(query_text, broaden_retry_limit)
            if broadened_queries:
                # Add original query and broadened queries as a list
                all_queries = [original_query_text] + broadened_queries
                # Keep queries as a list instead of concatenating
                query_text = all_queries
            else:
                print("⚠️ Query broadening failed after all retries, using original query")
                # Ensure we use the original clean query
                query_text = original_query_text

        # Ensure query_text is a list and clean each query
        if isinstance(query_text, str):
            query_text = [self._clean_query_text(query_text)]
        elif isinstance(query_text, list):
            query_text = [self._clean_query_text(q) for q in query_text if q]
        else:
            query_text = [self._clean_query_text(str(query_text))]

        
        vector_results = self.vector_rag.query(query_text, limit=limit)
        
        # Try to get graph answer, but handle gracefully if it fails
        try:
            # GraphRAG query expects a list of strings - query_text is already a list
            graph_answer = self.graph_rag.query(query_text)
            # Only include graph answer if it's not empty
            if not graph_answer or (isinstance(graph_answer, list) and not any(g.strip() for g in graph_answer)):
                graph_answer = None
        except Exception as e:
            print(f"⚠️  GraphRAG query failed: {e}")
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

        results = []
        if graph_answer is not None:
            # GraphRAG found a good answer - use it as primary result
            results = graph_answer
            # Add top vector results as supplementary information
            if vector_texts:
                results.extend(vector_texts)  # Top vector results as supplement
        else:
            # Fallback to vector results only
            results = vector_texts

        if rerank and vector_texts:
            # Ensure query_text is a string for rerank
            rerank_query = query_text if isinstance(query_text, str) else " ".join(query_text) if isinstance(query_text, list) else str(query_text)
            vector_texts = self.service.rerank_documents(vector_texts, rerank_query)
            # Update results if we used vector_texts
            if not (graph_answer is not None and isinstance(graph_answer, str) and graph_answer.strip() and "No relevant information found" not in graph_answer):
                results = vector_texts

        results = results[:limit]

        if compress:
            # Ensure query_text is a string for compress
            compress_query = query_text if isinstance(query_text, str) else " ".join(query_text) if isinstance(query_text, list) else str(query_text)
            compressed_result = self.service.compress_prompt(compress_query, results)
            # If compression fails or returns empty, return the original results
            if compressed_result and compressed_result.strip():
                return [compressed_result]
            else:
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
