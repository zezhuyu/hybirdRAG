from typing import Optional, List, Mapping, Any

from llama_index.core import SimpleDirectoryReader, SummaryIndex, Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.embeddings import BaseEmbedding
from openai import OpenAI
from comp import MLModelClient
from llama_index.core.embeddings import BaseEmbedding


class OpenAILLMWrapper(CustomLLM):
    context_window: int = 3900
    num_output: int = 256
    model_name: str = "OpenAILLMWrapper"
    dummy_response: str = ""
    client: Optional[OpenAI] = None

    def __init__(self, client: OpenAI, **kwargs):
        super().__init__(client=client, **kwargs)
        # OpenAI client doesn't have these attributes, using defaults

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        try:
            # Use the correct OpenAI client method for Ollama
            response = self.client.chat.completions.create(
                model="llama3.1:8b",  # Ollama model name
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.1
            )
            return CompletionResponse(text=response.choices[0].message.content)
        except Exception as e:
            # Fallback to dummy response if LLM is unavailable
            print(f"LLM connection failed: {e}")
            print("Using fallback response for GraphRAG...")
            return CompletionResponse(text="GraphRAG processing unavailable - LLM connection failed.")

    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        try:
            # Use streaming for Ollama
            response = self.client.chat.completions.create(
                model="gpt-oss:latest",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256,
                temperature=0.1,
                stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield CompletionResponse(text=chunk.choices[0].delta.content, delta=chunk.choices[0].delta.content)
        except Exception as e:
            # Fallback to dummy response if LLM is unavailable
            print(f"LLM streaming failed: {e}")
            yield CompletionResponse(text="GraphRAG streaming unavailable - LLM connection failed.")

    async def apredict(self, prompt: str, **kwargs: Any) -> str:
        """Async version of predict method for llama-index compatibility."""
        try:
            # Handle PromptTemplate objects
            if hasattr(prompt, 'format'):
                # It's a PromptTemplate object, format it with kwargs
                prompt_text = prompt.format(**kwargs)
            else:
                # It's already a string
                prompt_text = str(prompt)
            
            # Use the correct OpenAI client method for Ollama
            response = self.client.chat.completions.create(
                model="llama3.1:8b",  # Ollama model name
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=kwargs.get('max_tokens', 256),
                temperature=kwargs.get('temperature', 0.1)
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback to dummy response if LLM is unavailable
            print(f"LLM connection failed: {e}")
            print("Using fallback response for GraphRAG...")
            return "GraphRAG processing unavailable - LLM connection failed."


class MLModelEmbeddingWrapper(BaseEmbedding):
    """Wrapper to make MLModelClient compatible with llama-index BaseEmbedding interface."""
    
    ml_model_client: MLModelClient
    
    def __init__(self, ml_model_client: MLModelClient, **kwargs):
        super().__init__(ml_model_client=ml_model_client, **kwargs)
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        return self.ml_model_client.embed_sentence(text)
    
    def _get_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts."""
        return self.ml_model_client.embed_batch(texts)
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a query (same as text embedding)."""
        return self._get_text_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async version of text embedding."""
        return self._get_text_embedding(text)
    
    async def _aget_text_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """Async version of batch text embedding."""
        return self._get_text_embedding_batch(texts)
    
    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async version of query embedding."""
        return self._get_query_embedding(query)



class OpenAIEmbeddingsWrapper(BaseEmbedding):
    def __init__(
        self,
        client: OpenAI,
        model_name: str = "bge-m3:latest ",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model = client

    def _get_query_embedding(self, query: str) -> List[float]:
        response = self._model.embeddings.create(
            input=query,
            model=self.model_name
        )
        return response.data[0].embedding

    def _get_text_embedding(self, text: str) -> List[float]:
        response = self._model.embeddings.create(
            input=text,
            model=self.model_name
        )
        return response.data[0].embedding

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        response = self._model.embeddings.create(
            input=texts,
            model=self.model_name
        )
        return [data.embedding for data in response.data]

    async def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)

class GRPCEmbeddingsWrapper(BaseEmbedding):
    def __init__(
        self,
        client: MLModelClient,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._model = client

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._model.embed_sentence(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._model.embed_sentence(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._model.embed_batch(texts)

    async def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    async def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._get_text_embeddings(texts)