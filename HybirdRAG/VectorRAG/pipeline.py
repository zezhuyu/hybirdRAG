from pymilvus import MilvusClient, DataType, Function, FunctionType, AnnSearchRequest, RRFRanker
from datetime import datetime, timezone
import os
import time
import globals
import schedule
import asyncio

from comp import MLModelClient
from langchain.schema import Document
from langchain.text_splitter import SentenceSplitter
from VectorRAG.text_processing import Formatter

schema = MilvusClient.create_schema()

index_params = MilvusClient.prepare_index_params()

bm25_function = Function(
    name="title_bm25_emb", # Function name
    input_field_names=["title"], # Name of the VARCHAR field containing raw text data
    output_field_names=["sparse"], # Name of the SPARSE_FLOAT_VECTOR field reserved to store generated embeddings
    function_type=FunctionType.BM25, # Set to `BM25`
)

DIMENSION = 1024

schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=10000, enable_analyzer=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=DIMENSION)
schema.add_field(field_name="add_at", datatype=DataType.INT64)
schema.add_field(field_name="chapter", datatype=DataType.INT64)
schema.add_field(field_name="section", datatype=DataType.INT64)
schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)
schema.add_function(bm25_function)

index_params.add_index(
    field_name="sparse",
    index_type="SPARSE_INVERTED_INDEX",
    metric_type="BM25",
    params={
        "inverted_index_algo": "DAAT_MAXSCORE",
        "bm25_k1": 1.2,
        "bm25_b": 0.75
    }
)

index_params.add_index(
    field_name="vector",
    index_type="IVF_FLAT",
    metric_type="COSINE",
    params={
        "nlist": 128
    }
)

index_params.add_index(
    field_name="text_vector",
    index_type="IVF_FLAT",
    metric_type="COSINE",
    params={
        "nlist": 128
    }
)

class VectorRAGPipeline:
    def __init__(self, milvus: MilvusClient, embedding: Embedding, collection_name: str = "vector_rag"):
        self.milvus = milvus
        self.index = index_params
        self.embedding = embedding
        self.collection_name = collection_name
        if not self.milvus.has_collection(collection_name=collection_name):
            self.milvus.create_collection(collection_name="vector_rag", schema=schema, index_params=self.index)

    def add_document(self, document: List[dict]):
        for doc in document:
            doc["vector"] = self.embedding.embed_sentence(doc["content"])
            doc['add_at'] = float(doc.get("add_at", time.time()))
        self.milvus.insert(collection_name=self.collection_name, data=document)

    def query(self, query: str, limit: int = 10, filters: List[str] = None):
        query_embedding = self.embedding.embed_sentence(query)
        search_params = {
            "output_fields": ["id", "content", "add_at", "chapter", "section"]
        }
        if filters:
            search_params["filter"] = " AND ".join(filter)
        sparse_search_params = search_params.copy()
        sparse_search_params["params"] = {'drop_ratio_search': 0.6}
        full_text_search_params = {"metric_type": "BM25"}
        full_text_search_req = AnnSearchRequest(
            [query], "sparse", full_text_search_params, limit=limit
        )

        dense_search_params = {"metric_type": "COSINE"}
        dense_req = AnnSearchRequest(
            [query_embedding], "vector", dense_search_params, limit=limit
        )
        results = self.milvus.hybrid_search(
                    self.collection_name,
                    [full_text_search_req, dense_req],
                    ranker=RRFRanker(),
                    limit=limit,
                    **sparse_search_params
                )

        return results