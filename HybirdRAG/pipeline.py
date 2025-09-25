from VectorRAG.pipeline import VectorRAGPipeline
from GraphRAG.pipeline import GraphRAGPipeline
from pymilvus import MilvusClient
from comp import MLModelClient
from GraphRAG.convertObj import OpenAILLMWrapper
from GraphRAG.store import GraphRAGStore
from VectorRAG.text_processing import ContextualChunker   
from VectorRAG.text_processing import Formatter
from VectorRAG.text_processing import LateChunker
from openai import OpenAI
from GraphRAG.document import Document
from GraphRAG.text_splitter import SentenceSplitter
from typing import List

rewrite_prompt = """
You are a helpful assistant that rewrites the prompt to be more specific and concise.
given the user chat history and query, rewrite the prompt to be more specific and concise.

# Chat History:
{chat_history}
# Query:
"""

class HybridRAGPipeline:
    def __init__(self, service_host: str, milvus_host: str, neo4j_host: str, openai_host: str, openai_api_key: str):
        self.vector_rag = vector_rag
        self.graph_rag = graph_rag
        self.openai = OpenAI(api_key=openai_api_key, base_url=openai_host)
        self.service = MLModelClient(host=service_host)
        self.milvus = MilvusClient(host=milvus_host)
        self.vector_rag = VectorRAGPipeline(milvus=self.milvus, embedding=self.service)
        self.graph_rag = GraphRAGPipeline(llm=OpenAILLMWrapper(client=self.openai), graph_store=GraphRAGStore())
        
    def query(self, query: str, chat_history: List[str] = [], context_chunk_size: int = 128, rerank: bool = True, compress: bool = True):

        prompt = rewrite_prompt.format(chat_history=chat_history.join("\n"), query=query)
        prompt = self.openai.chat.completions.create(
            model="qwen3:0.6b",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ]
        )
        query = prompt.choices[0].message.content
        vector_results = self.vector_rag.query(query)
        graph_results = self.graph_rag.query(query)

        vector_results = [result["entity"]["content"] for result in vector_results]
        if context_chunk_size > 0:
            chunker = ContextualChunker(mode="sentence", max_chunk_size=context_chunk_size)
            refined_chunks = chunker.process_results(retrieved_texts)
            vector_results = [c["text"] for c in refined_chunks]
        if rerank:
            vector_results = self.service.rerank_documents(vector_results, query)
        vector_results.insert(0, graph_results)
        if compress:
            return self.service.compress_prompt(query, vector_results)
        return vector_results

    def add_document(self, document: dict):
        chunker = LateChunker()
        chunks = chunker.chunk_document(document["content"])
        chunked_docs = [{**{k: v for k, v in document.items() if k != "content"}, "content": chunk} for chunk in chunks]
        vector_docs = Formatter.to_custom_rag(chunked_docs)
        self.vector_rag.add_document(vector_docs)
        graph_docs = Formatter.to_llamaindex([document])
        nodes = []
        for doc in graph_docs:
            doc_obj = Document(page_content=doc["content"], metadata=doc)
            nodes.extend(self.splitter.split_text(doc_obj.page_content))
        self.graph_rag.build_index(nodes)
        self.graph_rag.build_communities()

    def add_documents(self, documents: list):
        chunker = LateChunker()
        expanded_docs = []
        for document in documents:
            chunks = chunker.chunk_document(document["content"])
            expanded_docs.extend([{**{k: v for k, v in document.items() if k != "content"}, "content": chunk} for chunk in chunks])
        vector_docs = Formatter.to_custom_rag(expanded_docs)
        self.vector_rag.add_document(vector_docs)
        graph_docs = Formatter.to_llamaindex(documents)
        nodes = []
        for doc in graph_docs:
            doc_obj = Document(page_content=doc["content"], metadata=doc)
            nodes.extend(self.splitter.split_text(doc_obj.page_content))
        self.graph_rag.build_index(nodes)
        self.graph_rag.build_communities()