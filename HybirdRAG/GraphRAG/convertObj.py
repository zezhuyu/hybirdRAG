from typing import Optional, List, Mapping, Any

from llama_index.core import SimpleDirectoryReader, SummaryIndex, Settings
from llama_index.core.callbacks import CallbackManager, llm_completion_callback
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.embeddings import BaseEmbedding
from openai import OpenAI
from comp import MLModelClient


class OpenAILLMWrapper(CustomLLM):
    context_window: int = 3900
    num_output: int = 256
    model_name: str = "OpenAILLMWrapper"
    dummy_response: str = ""

    def __init__(self, client: OpenAI):
        self.client = client
        self.model_name = client.model_name
        self.context_window = client.context_window
        self.num_output = client.num_output

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text=self.client.complete(prompt))

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        response = ""
        for token in self.dummy_response:
            response += token
            yield CompletionResponse(text=response, delta=token)



class OpenAIEmbeddingsWrapper(BaseEmbedding):
    def __init__(
        self,
        client: OpenAI,
        model_name: str = "text-embedding-3-small",
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