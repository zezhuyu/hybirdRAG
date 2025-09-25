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
Extract entities and relationships from the text. Return ONLY valid JSON.

Extract up to {max_knowledge_triplets} entities and their relationships.

For each entity, extract: entity_name, entity_type, entity_description
For each relationship, extract: source_entity, target_entity, relation, relationship_description

Return ONLY this JSON format (no other text):
{
  "entities": [
    {"entity_name": "Entity Name", "entity_type": "Type", "entity_description": "Description"}
  ],
  "relationships": [
    {"source_entity": "Source", "target_entity": "Target", "relation": "Relation", "relationship_description": "Description"}
  ]
}

If none found, return: {"entities": [], "relationships": []}

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
    """Parse the LLM response to extract entities and relationships."""
    entities = []
    relationships = []
    
    # Try to find JSON in the response - look for opening brace
    json_start = response_str.find('{')
    if json_start == -1:
        print(f"❌ No JSON found in response")
        print(f"Raw response: {response_str[:200]}...")
        return entities, relationships
    
    # Extract everything from the first opening brace
    json_str = response_str[json_start:]
    
    # Try to fix common JSON truncation issues
    json_str = fix_truncated_json(json_str)
    
    try:
        data = json.loads(json_str)
        
        # Extract entities
        entities = [
            (
                entity.get("entity_name", ""),
                entity.get("entity_type", ""),
                entity.get("entity_description", ""),
            )
            for entity in data.get("entities", [])
        ]
        
        # Extract relationships
        relationships = [
            (
                relation.get("source_entity", ""),
                relation.get("target_entity", ""),
                relation.get("relation", ""),
                relation.get("relationship_description", ""),
            )
            for relation in data.get("relationships", [])
        ]
        
        print(f"✅ Successfully parsed {len(entities)} entities and {len(relationships)} relationships")
        return entities, relationships
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        print(f"Raw response: {response_str[:200]}...")
        
        # Try to extract partial data from malformed JSON
        try:
            partial_entities, partial_relationships = extract_partial_json(response_str)
            if partial_entities or partial_relationships:
                print(f"⚠️  Extracted partial data: {len(partial_entities)} entities, {len(partial_relationships)} relationships")
                return partial_entities, partial_relationships
        except:
            pass
        
        return entities, relationships


def fix_truncated_json(json_str: str) -> str:
    """Attempt to fix common JSON truncation issues."""
    # If the JSON appears to be truncated (doesn't end with }), try to close it
    if not json_str.strip().endswith('}'):
        # Find the last complete object or array
        brace_count = 0
        bracket_count = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(json_str):
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\':
                escape_next = True
                continue
                
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
                
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                elif char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
        
        # Try to close incomplete structures
        if bracket_count > 0:
            json_str += ']' * bracket_count
        if brace_count > 0:
            json_str += '}' * brace_count
        elif brace_count < 0:
            # Remove extra closing braces
            json_str = json_str.rstrip('}')
    
    return json_str


def extract_partial_json(response_str: str) -> tuple:
    """Extract entities and relationships from malformed JSON using regex."""
    entities = []
    relationships = []
    
    try:
        # Extract entities using regex patterns
        entity_pattern = r'"entity_name":\s*"([^"]*)",\s*"entity_type":\s*"([^"]*)",\s*"entity_description":\s*"([^"]*)"'
        entity_matches = re.findall(entity_pattern, response_str)
        
        for match in entity_matches:
            entities.append((match[0], match[1], match[2]))
        
        # Extract relationships using regex patterns
        rel_pattern = r'"source_entity":\s*"([^"]*)",\s*"target_entity":\s*"([^"]*)",\s*"relation":\s*"([^"]*)",\s*"relationship_description":\s*"([^"]*)"'
        rel_matches = re.findall(rel_pattern, response_str)
        
        for match in rel_matches:
            relationships.append((match[0], match[1], match[2], match[3]))
            
    except Exception as e:
        print(f"⚠️  Partial extraction failed: {e}")
    
    return entities, relationships


class GraphRAGPipeline:
    def __init__(
        self,
        llm: LLM,
        graph_store: GraphRAGStore,
        *,
        vector_store: Optional[MilvusVectorStore] = None,
        milvus_kwargs: Optional[Dict[str, Any]] = None,
        embedding_model: Optional[Any] = None,
    ) -> None:
        self.llm = llm
        self.embedding_model = embedding_model
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

        # Set the LLM and embeddings in global settings for all llama-index operations
        from llama_index.core import Settings
        Settings.llm = llm
        if embedding_model:
            Settings.embed_model = embedding_model

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
                property_graph_store=self.graph_store,  # Explicitly pass our custom store
                show_progress=True,
            )
        else:
            self.index.insert_nodes(processed_nodes)
        return self.index

    def load_index(self) -> PropertyGraphIndex:
        """Load an existing index definition from the persistent stores."""
        self.index = PropertyGraphIndex.from_existing(
            property_graph_store=self.graph_store
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
