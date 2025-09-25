import json
import re
import sys
from typing import Any, Dict, List, Optional

from llama_index.core import PropertyGraphIndex, StorageContext
from llama_index.core.llms import LLM
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.simple import SimpleVectorStore as MilvusVectorStore

from .extractor import GraphRAGExtractor
from .query import GraphRAGQueryEngine
from .store import GraphRAGStore




KG_TRIPLET_EXTRACT_TMPL = """
-Goal-
Given a text document, identify all entities and their entity types from the text and all relationships among the identified entities.
Given the text, extract up to {max_knowledge_triplets} entity-relation triplets.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type of the entity
- entity_description: Comprehensive description of the entity's attributes and activities

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relation: relationship between source_entity and target_entity
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other

3. Output Formatting:
- Return the result in valid JSON format with two keys: 'entities' (list of entity objects) and 'relationships' (list of relationship objects).
- Exclude any text outside the JSON structure (e.g., no explanations or comments).
- If no entities or relationships are identified, return empty lists: { "entities": [], "relationships": [] }.

-An Output Example-
{
  "entities": [
    {
      "entity_name": "Albert Einstein",
      "entity_type": "Person",
      "entity_description": "Albert Einstein was a theoretical physicist who developed the theory of relativity and made significant contributions to physics."
    },
    {
      "entity_name": "Theory of Relativity",
      "entity_type": "Scientific Theory",
      "entity_description": "A scientific theory developed by Albert Einstein, describing the laws of physics in relation to observers in different frames of reference."
    },
    {
      "entity_name": "Nobel Prize in Physics",
      "entity_type": "Award",
      "entity_description": "A prestigious international award in the field of physics, awarded annually by the Royal Swedish Academy of Sciences."
    }
  ],
  "relationships": [
    {
      "source_entity": "Albert Einstein",
      "target_entity": "Theory of Relativity",
      "relation": "developed",
      "relationship_description": "Albert Einstein is the developer of the theory of relativity."
    },
    {
      "source_entity": "Albert Einstein",
      "target_entity": "Nobel Prize in Physics",
      "relation": "won",
      "relationship_description": "Albert Einstein won the Nobel Prize in Physics in 1921."
    }
  ]
}

-Real Data-
######################
text: {text}
######################
output:"""

def parse_fn(response_str: str) -> Any:
    json_pattern = r"\{.*\}"
    match = re.search(json_pattern, response_str, re.DOTALL)
    entities = []
    relationships = []
    if not match:
        return entities, relationships
    json_str = match.group(0)
    try:
        data = json.loads(json_str)
        entities = [
            (
                entity["entity_name"],
                entity["entity_type"],
                entity["entity_description"],
            )
            for entity in data.get("entities", [])
        ]
        relationships = [
            (
                relation["source_entity"],
                relation["target_entity"],
                relation["relation"],
                relation["relationship_description"],
            )
            for relation in data.get("relationships", [])
        ]
        return entities, relationships
    except json.JSONDecodeError as e:
        print("Error parsing JSON:", e)
        return entities, relationships


class GraphRAGPipeline:
    def __init__(
        self,
        llm: LLM,
        graph_store: GraphRAGStore,
        *,
        vector_store: Optional[MilvusVectorStore] = None,
        milvus_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.llm = llm
        self.graph_store = graph_store
        if hasattr(self.graph_store, "set_summarizer_llm"):
            self.graph_store.set_summarizer_llm(llm)
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            if milvus_kwargs is None:
                raise ValueError(
                    "Provide either an initialised Milvus vector_store or milvus_kwargs "
                    "to construct one."
                )
            self.vector_store = MilvusVectorStore(**milvus_kwargs)

        self.storage_context = StorageContext.from_defaults(
            graph_store=self.graph_store,
            vector_store=self.vector_store,
        )

        self.kg_extractor = GraphRAGExtractor(
            llm=llm,
            extract_prompt=KG_TRIPLET_EXTRACT_TMPL,
            max_paths_per_chunk=2,
            parse_fn=parse_fn,
        )
        self.index: Optional[PropertyGraphIndex] = None
        self.query_engine: Optional[GraphRAGQueryEngine] = None

    def build_index(self, nodes: List[BaseNode]) -> PropertyGraphIndex:
        if not nodes:
            raise ValueError("No nodes supplied for index construction.")

        processed_nodes = self.kg_extractor(
            nodes, show_progress=True
        )

        if self.index is None:
            self.index = PropertyGraphIndex(
                nodes=processed_nodes,
                storage_context=self.storage_context,
                show_progress=True,
            )
        else:
            self.index.insert_nodes(processed_nodes)
        return self.index

    def load_index(self) -> PropertyGraphIndex:
        """Load an existing index definition from the persistent stores."""
        self.index = PropertyGraphIndex.from_storage(
            storage_context=self.storage_context
        )
        return self.index

    def build_communities(self) -> None:
        self.graph_store.build_communities()

    def query(self, query: str) -> str:
        if self.index is None:
            try:
                self.load_index()
            except ValueError as exc:
                raise ValueError("Index is not built and no persisted index found.") from exc

        if self.query_engine is None:
            self.query_engine = GraphRAGQueryEngine(
                graph_store=self.graph_store,
                llm=self.llm,
                index=self.index,
                similarity_top_k=10,
            )
        return self.query_engine.custom_query(query)
