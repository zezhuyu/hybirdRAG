from typing import Any, Dict, List, Optional
import sys
import os

from openai import OpenAI
from pymilvus import MilvusClient

from .VectorRAG.pipeline import VectorRAGPipeline
from .VectorRAG.text_processing import ContextualChunker, LateChunker

from .GraphRAG.convertObj import OpenAILLMWrapper
from .GraphRAG.pipeline import GraphRAGPipeline
from .GraphRAG.store import GraphRAGStore

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

        self.openai = OpenAI(api_key=openai_api_key, base_url=os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1/" )
        self.graph_llm = OpenAILLMWrapper(client=self.openai)

        graph_store_settings = dict(neo4j_config)
        graph_store_settings.setdefault("summarizer_llm", self.graph_llm)
        self.graph_store = GraphRAGStore(**graph_store_settings)

        self.graph_rag = GraphRAGPipeline(
            llm=self.graph_llm,
            graph_store=self.graph_store,
            milvus_kwargs=dict(graph_vector_store_kwargs),
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

        vector_results = self.vector_rag.query(query_text)
        graph_answer = self.graph_rag.query(query_text)

        vector_texts = [
            hit.get("entity", {}).get("content")
            for hit in vector_results
            if isinstance(hit, dict)
        ]
        vector_texts = [text for text in vector_texts if text]

        if context_chunk_size > 0 and vector_texts:
            self.context_chunker.max_chunk_size = context_chunk_size
            refined_chunks = self.context_chunker.process_results(vector_texts)
            vector_texts = [chunk["text"] for chunk in refined_chunks if "text" in chunk]

        if rerank and vector_texts:
            vector_texts = self.service.rerank_documents(vector_texts, query_text)

        results = [graph_answer] + vector_texts

        if compress:
            return self.service.compress_prompt(query_text, results)
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
