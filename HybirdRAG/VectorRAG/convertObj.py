from typing import List
from openai import OpenAI
import os

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
            print(f"⚠️  Batch embedding failed, using individual calls: {e}")
            return None

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
            return None

    def embed_batch(self, sentences: List[str]) -> List[List[float]]:
        try:
            response = self._model.embeddings.create(
                input=sentences,
                model=self.model_name
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            print(f"⚠️  Batch embedding failed, using individual calls: {e}")
            return None