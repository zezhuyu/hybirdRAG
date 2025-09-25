import grpc
from typing import List

# Import generated protobuf classes
try:
    # Try relative imports first (when used as part of package)
    from . import ml_models_pb2 as pb2
    from . import ml_models_pb2_grpc as pb2_grpc
    PROTO_AVAILABLE = True
except ImportError:
    try:
        # Fall back to absolute imports (when run directly)
        import ml_models_pb2 as pb2
        import ml_models_pb2_grpc as pb2_grpc
        PROTO_AVAILABLE = True
    except ImportError:
        print("Error: Protobuf files not found.")
        print("Please run: python -m grpc_tools.protoc --python_out=. --grpc_python_out=. ml_models.proto")
        PROTO_AVAILABLE = False
        raise ImportError("Protobuf files not found. Please run: python -m grpc_tools.protoc --python_out=. --grpc_python_out=. ml_models.proto")
    
class MLModelClient:
    def __init__(self, host: str):
        self.channel = grpc.insecure_channel(f'{host}')
        self.stub = pb2_grpc.MLModelServiceStub(self.channel)

    def embed_sentence(self, sentence: str) -> List[float]:
        response = self.stub.EmbedSentence(pb2.EmbedSentenceRequest(sentence=sentence))
        return response.embedding

    def embed_batch(self, sentences: List[str]) -> List[List[float]]:
        response = self.stub.EmbedBatch(pb2.EmbedBatchRequest(sentences=sentences))
        return [r.values for r in response.embeddings]

    def rerank_documents(self, documents: List[str], query: str) -> List[str]:
        response = self.stub.RerankDocuments(pb2.RerankDocumentsRequest(documents=documents, query=query))
        sorted_docs = sorted([{'doc': doc, 'score': score} for doc, score in zip(documents, response.scores)], key=lambda x: x['score'], reverse=True)
        return [item['doc'] for item in sorted_docs]

    def rewrite_prompt(self, prompt: str) -> str:
        response = self.stub.RewritePrompt(pb2.RewritePromptRequest(prompt=prompt))
        return response.rewritten_prompt

    def compress_prompt(self, prompt: str, documents: List[str]) -> str:
        response = self.stub.CompressPrompt(pb2.CompressPromptRequest(prompt=prompt, documents=documents))
        return response.compressed_prompt
    
    def get_memory_info(self) -> pb2.MemoryInfo:
        response = self.stub.GetMemoryInfo(pb2.GetMemoryInfoRequest())
        return response.memory_info
    
    def cleanup_models(self):
        response = self.stub.CleanupModels(pb2.CleanupModelsRequest())
        return response.success
    
    def warmup_models(self):
        response = self.stub.WarmupModels(pb2.WarmupModelsRequest())
        return response.success
    
    def get_server_status(self) -> pb2.ServerStatus:
        response = self.stub.GetServerStatus(pb2.GetServerStatusRequest())
        return response.status