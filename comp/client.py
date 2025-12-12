import grpc
import time
from typing import List, Optional, Callable, TypeVar

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

T = TypeVar('T')

def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    operation_name: str = "Operation"
) -> Optional[T]:
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        backoff_factor: Multiplier for delay after each retry
        operation_name: Name of the operation for logging
    
    Returns:
        Result of the function call, or None if all retries failed
    """
    delay = initial_delay
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except grpc.RpcError as e:
            if attempt == max_retries:
                print(f"❌ {operation_name} failed after {max_retries} retries: {e.code()}")
                raise
            
            # Check if it's a retryable error
            if e.code() in [
                grpc.StatusCode.UNAVAILABLE,
                grpc.StatusCode.DEADLINE_EXCEEDED,
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                grpc.StatusCode.ABORTED,
                grpc.StatusCode.UNKNOWN
            ]:
                print(f"⚠️  {operation_name} failed (attempt {attempt + 1}/{max_retries + 1}): {e.code()}")
                print(f"   Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= backoff_factor
            else:
                # Non-retryable error, fail immediately
                print(f"❌ {operation_name} failed with non-retryable error: {e.code()}")
                raise
        except Exception as e:
            if attempt == max_retries:
                print(f"❌ {operation_name} failed after {max_retries} retries: {e}")
                raise
            
            print(f"⚠️  {operation_name} failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
            print(f"   Retrying in {delay:.1f}s...")
            time.sleep(delay)
            delay *= backoff_factor
    
    return None
    
class MLModelClient:
    def __init__(self, host: str, max_retries: int = 3):
        self.channel = grpc.insecure_channel(f'{host}')
        self.stub = pb2_grpc.MLModelServiceStub(self.channel)
        self.max_retries = max_retries

    def embed_sentence(self, sentence: str) -> List[float]:
        def _call():
            response = self.stub.EmbedSentence(pb2.EmbedSentenceRequest(sentence=sentence))
            return response.embedding
        
        return retry_with_backoff(
            _call,
            max_retries=self.max_retries,
            operation_name="Embed sentence"
        )

    def embed_sentences(self, sentences: List[str]) -> List[List[float]]:
        """
        🚀 Batch embedding generation - optimized for multiple sentences.
        Falls back to individual calls if batch method fails.
        """
        try:
            # Try batch method first
            return self.embed_batch(sentences)
        except Exception as e:
            print(f"⚠️  Batch embedding failed, using individual calls: {e}")
            # Fallback to individual calls with retry
            embeddings = []
            for sentence in sentences:
                embedding = self.embed_sentence(sentence)
                embeddings.append(embedding)
            return embeddings

    def embed_batch(self, sentences: List[str]) -> List[List[float]]:
        def _call():
            response = self.stub.EmbedBatch(pb2.EmbedBatchRequest(sentences=sentences))
            return [r.values for r in response.embeddings]
        
        return retry_with_backoff(
            _call,
            max_retries=self.max_retries,
            operation_name=f"Embed batch ({len(sentences)} sentences)"
        )

    def rerank_documents(self, documents: List[str], query: str) -> List[str]:
        def _call():
            response = self.stub.RerankDocuments(pb2.RerankDocumentsRequest(documents=documents, query=query))
            if not response.scores or len(response.scores) == 0:
                # Silently return original documents if reranking isn't working
                # Only print warning once per client instance
                if not hasattr(self, '_rerank_warning_shown'):
                    print(f"⚠️  Warning: Rerank service not returning scores, using original document order")
                    self._rerank_warning_shown = True
                return documents  # Return original documents if no scores
            
            # Check if we have valid scores
            if len(response.scores) != len(documents):
                if not hasattr(self, '_rerank_mismatch_warning_shown'):
                    print(f"⚠️  Warning: Rerank scores count mismatch (got {len(response.scores)}, expected {len(documents)})")
                    self._rerank_mismatch_warning_shown = True
                return documents
            
            # Sort documents by scores (higher scores first)
            sorted_docs = sorted([{'doc': doc, 'score': score} for doc, score in zip(documents, response.scores)], 
                               key=lambda x: x['score'], reverse=True)
            return [item['doc'] for item in sorted_docs]
        
        return retry_with_backoff(
            _call,
            max_retries=self.max_retries,
            operation_name=f"Rerank documents ({len(documents)} docs)"
        )

    def rewrite_prompt(self, prompt: str) -> str:
        def _call():
            response = self.stub.RewritePrompt(pb2.RewritePromptRequest(prompt=prompt))
            return response.rewritten_prompt
        
        return retry_with_backoff(
            _call,
            max_retries=self.max_retries,
            operation_name="Rewrite prompt"
        )

    def compress_prompt(self, prompt: str, documents: List[str]) -> str:
        def _call():
            response = self.stub.CompressPrompt(pb2.CompressPromptRequest(prompt=prompt, documents=documents))
            return response.compressed_prompt
        
        return retry_with_backoff(
            _call,
            max_retries=self.max_retries,
            operation_name="Compress prompt"
        )
    
    def get_memory_info(self) -> pb2.MemoryInfo:
        def _call():
            response = self.stub.GetMemoryInfo(pb2.GetMemoryInfoRequest())
            return response.memory_info
        
        return retry_with_backoff(
            _call,
            max_retries=self.max_retries,
            operation_name="Get memory info"
        )
    
    def cleanup_models(self):
        def _call():
            response = self.stub.CleanupModels(pb2.CleanupModelsRequest())
            return response.success
        
        return retry_with_backoff(
            _call,
            max_retries=self.max_retries,
            operation_name="Cleanup models"
        )
    
    def warmup_models(self):
        def _call():
            response = self.stub.WarmupModels(pb2.WarmupModelsRequest())
            return response.success
        
        return retry_with_backoff(
            _call,
            max_retries=self.max_retries,
            operation_name="Warmup models"
        )
    
    def get_server_status(self) -> pb2.ServerStatus:
        def _call():
            response = self.stub.GetServerStatus(pb2.GetServerStatusRequest())
            return response.status
        
        return retry_with_backoff(
            _call,
            max_retries=self.max_retries,
            operation_name="Get server status"
        )