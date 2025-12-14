from typing import List
from openai import OpenAI
import HybirdRAG.comp as comp
import os
import requests

class OpenAIEmbeddingClient:
    def __init__(self, model: OpenAI, model_name: str = None):
        self._model = model
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME", "bge-m3:latest")


    def embed_sentence(self, sentence: str) -> List[float]:
        try:
            response = self._model.embeddings.create(
                input=sentence,
                model=self.model_name
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"❌ Embedding failed for sentence: {e}")
            print(f"   Model: {self.model_name}")
            print(f"   Sentence length: {len(sentence)} chars")
            raise  # Re-raise the exception instead of returning None

    def embed_sentences(self, sentences: List[str]) -> List[List[float]]:
        """
        🚀 Batch embedding generation - optimized for multiple sentences.
        Falls back to individual calls if batch method fails.
        """
        try:
            # Try batch method first
            response = self._model.embeddings.create(
                input=sentences,
                model=self.model_name
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"⚠️  Batch embedding failed, using individual calls: {e}")
            # Fallback to individual embeddings instead of returning None
            return [self.embed_sentence(s) for s in sentences]

    def embed_batch(self, sentences: List[str]) -> List[List[float]]:
        try:
            response = self._model.embeddings.create(
                input=sentences,
                model=self.model_name
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"⚠️  Batch embedding failed, using individual calls: {e}")
            # Fallback to individual embeddings instead of returning None
            return [self.embed_sentence(s) for s in sentences]

class OpenAIRerankerClient:
    def __init__(self, model_name: str = None):
        base_url = os.getenv("OPENAI_BASE_URL") or "http://localhost:8080"
        openai_api_key = os.getenv("OPENAI_API_KEY") or "ollama"
        # Ensure base_url doesn't have trailing slashes
        base_url = base_url.rstrip('/')
        self.base_url = base_url
        self.api_key = openai_api_key
        self.model_name = model_name or os.getenv("RERANKER_MODEL_NAME", "ai/qwen3-reranker-vllm:0.6B")
        # Construct rerank endpoint URL
        self.rerank_url = f"{self.base_url}/engines/vllm/rerank"
        # Score endpoint for pairwise scoring
        self.score_url = f"{self.base_url}/engines/vllm/score"

    def rerank_documents(self, documents: List[str], query: str) -> List[str]:
        """
        Rerank documents against a query and return reranked documents (sorted by score).
        
        Matches the signature and behavior of MLModelClient.rerank_documents.
        Returns original documents if reranking fails or scores are invalid.
        
        Args:
            documents: List of documents to rerank
            query: Query string to rank documents against
            
        Returns:
            List[str]: Reranked documents (sorted by relevance score, highest first)
        """
        try:
            # Prepare request payload following the curl pattern
            payload = {
                "model": self.model_name,
                "query": query,
                "documents": documents
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            # Add API key to headers if provided
            if self.api_key and self.api_key != "ollama":
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Make HTTP POST request
            response = requests.post(
                self.rerank_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse response JSON
            result = response.json()
            
            # Extract scores from response
            # The response format may vary, try common patterns
            if "scores" in result:
                scores = result["scores"]
            elif "data" in result and isinstance(result["data"], list):
                # If response has data array with score fields
                scores = [item.get("score", 0.0) for item in result["data"]]
            elif isinstance(result, list):
                # If response is directly a list of scores
                scores = result
            else:
                # Try to find scores in nested structure
                scores = result.get("results", {}).get("scores", [])
            
            # Validate scores - match gRPC client behavior
            if not scores or len(scores) == 0:
                # Silently return original documents if reranking isn't working
                # Only print warning once per client instance
                if not hasattr(self, '_rerank_warning_shown'):
                    print(f"⚠️  Warning: Rerank service not returning scores, using original document order")
                    self._rerank_warning_shown = True
                return documents  # Return original documents if no scores
            
            # Check if we have valid scores
            if len(scores) != len(documents):
                if not hasattr(self, '_rerank_mismatch_warning_shown'):
                    print(f"⚠️  Warning: Rerank scores count mismatch (got {len(scores)}, expected {len(documents)})")
                    self._rerank_mismatch_warning_shown = True
                return documents
            
            # Sort documents by scores (higher scores first)
            sorted_docs = sorted([{'doc': doc, 'score': float(score)} for doc, score in zip(documents, scores)], 
                               key=lambda x: x['score'], reverse=True)
            return [item['doc'] for item in sorted_docs]
            
        except requests.exceptions.RequestException as e:
            # On HTTP errors, return original documents (matching gRPC client behavior)
            if not hasattr(self, '_rerank_http_error_shown'):
                print(f"⚠️  Warning: HTTP request failed for reranking: {e}")
                print(f"   URL: {self.rerank_url}")
                print(f"   Model: {self.model_name}")
                print(f"   Returning original document order")
                self._rerank_http_error_shown = True
            return documents
        except (KeyError, ValueError, TypeError) as e:
            # On parsing errors, return original documents
            if not hasattr(self, '_rerank_parse_error_shown'):
                print(f"⚠️  Warning: Failed to parse rerank response: {e}")
                print(f"   Model: {self.model_name}")
                print(f"   Response: {response.text if 'response' in locals() else 'N/A'}")
                print(f"   Returning original document order")
                self._rerank_parse_error_shown = True
            return documents
        except Exception as e:
            # On any other error, return original documents
            if not hasattr(self, '_rerank_error_shown'):
                print(f"⚠️  Warning: Reranking failed: {e}")
                print(f"   Model: {self.model_name}")
                print(f"   Returning original document order")
                self._rerank_error_shown = True
            return documents
    