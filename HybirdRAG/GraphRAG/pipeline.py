import json
import re
import sys
from typing import Any, Dict, List, Optional

from llama_index.core import PropertyGraphIndex, StorageContext
from llama_index.core.llms import LLM
from llama_index.core.schema import BaseNode
# Use SimpleVectorStore as the base, but we'll connect it to Milvus data
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
        milvus_client: Optional[Any] = None,
    ) -> None:
        self.llm = llm
        self.embedding_model = embedding_model
        self.graph_store = graph_store
        if hasattr(self.graph_store, "set_summarizer_llm"):
            self.graph_store.set_summarizer_llm(llm)
        
        # Store the Milvus client for syncing data
        self.milvus_client = milvus_client
        
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
        
        # Try to load existing index first
        try:
            self.load_index()
            print("✅ Loaded existing GraphRAG index")
        except Exception as e:
            print(f"⚠️  No existing GraphRAG index found: {e}")
            self.index = None
            
            # If no existing index, try to build one from Milvus data
            if self.milvus_client is not None:
                print("🔄 Building new GraphRAG index from Milvus data...")
                self._build_index_from_milvus()

    def _sync_vector_store_with_milvus(self):
        """Sync the GraphRAG vector store with data from Milvus collection."""
        try:
            from llama_index.core.schema import TextNode
            
            # Get collection info
            collection_name = self.milvus_client.collection_name if hasattr(self.milvus_client, 'collection_name') else 'vector_rag'
            
            # Query all documents from Milvus
            results = self.milvus_client.query(
                collection_name=collection_name,
                filter="",  # Get all documents
                output_fields=["content", "title", "page", "source"],
                limit=1000  # Reasonable limit
            )
            
            print(f"🔄 Syncing GraphRAG vector store with {len(results)} documents from Milvus...")
            
            # Clear existing data
            self.vector_store._data.embedding_dict.clear()
            self.vector_store._data.text_id_to_ref_doc_id.clear()
            self.vector_store._data.metadata_dict.clear()
            
            # Add documents to vector store
            for i, result in enumerate(results):
                content = result.get('content', '')
                if content and len(content.strip()) > 50:  # Only substantial content
                    node = TextNode(
                        text=content,
                        metadata={
                            'title': result.get('title', ''),
                            'page': result.get('page', ''),
                            'source': result.get('source', ''),
                            'milvus_id': result.get('id', i)
                        }
                    )
                    
                    # Generate embedding and set it on the node
                    if self.embedding_model and hasattr(self.embedding_model, 'ml_model_client'):
                        try:
                            embedding = self.embedding_model.ml_model_client.embed_sentence(content)
                            # Set the embedding on the node
                            node.embedding = embedding
                            
                            # Add to both vector store and docstore
                            self.vector_store.add([node])
                            self.storage_context.docstore.add_documents([node])
                            
                            # Ensure proper mapping in vector store
                            if hasattr(self.vector_store, '_data'):
                                # Set the text_id_to_ref_doc_id mapping
                                self.vector_store._data.text_id_to_ref_doc_id[node.node_id] = node.node_id
                            
                        except Exception as embed_error:
                            print(f"⚠️  Failed to generate embedding for document {i}: {embed_error}")
                            # Skip this document if embedding fails
                            continue
                    else:
                        print(f"⚠️  No embedding model available, skipping document {i}")
                        continue
            
            print(f"✅ Synced {len(results)} documents to GraphRAG vector store")
            
        except Exception as e:
            print(f"⚠️  Failed to sync vector store with Milvus: {e}")
            import traceback
            traceback.print_exc()

    def _build_index_from_milvus(self):
        """Build GraphRAG index from Milvus documents."""
        try:
            from llama_index.core.schema import TextNode
            
            # Get collection info
            collection_name = self.milvus_client.collection_name if hasattr(self.milvus_client, 'collection_name') else 'vector_rag'
            
            # Query documents from Milvus
            results = self.milvus_client.query(
                collection_name=collection_name,
                filter="",  # Get all documents
                output_fields=["content", "title", "page", "source"],
                limit=100  # Reasonable limit for GraphRAG
            )
            
            print(f"🔄 Building GraphRAG index from {len(results)} Milvus documents...")
            
            # Convert to TextNodes
            nodes = []
            for i, result in enumerate(results):
                content = result.get('content', '')
                if content and len(content.strip()) > 100:  # Only substantial content
                    node = TextNode(
                        text=content,
                        metadata={
                            'title': result.get('title', ''),
                            'page': result.get('page', ''),
                            'source': result.get('source', ''),
                            'milvus_id': result.get('id', i)
                        }
                    )
                    nodes.append(node)
            
            print(f"🔄 Created {len(nodes)} text nodes, building GraphRAG index...")
            
            # Build the index
            if nodes:
                self.build_index(nodes)
                print("✅ GraphRAG index built successfully from Milvus data")
            else:
                print("⚠️  No suitable documents found in Milvus")
                
        except Exception as e:
            print(f"⚠️  Failed to build index from Milvus: {e}")
            import traceback
            traceback.print_exc()

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
        # First sync the vector store with Milvus data before loading the index
        if self.milvus_client is not None:
            print("🔄 Syncing vector store with Milvus data before loading index...")
            self._sync_vector_store_with_milvus()
        
        # Now try to load the index structure
        try:
            self.index = PropertyGraphIndex.from_existing(
                property_graph_store=self.graph_store,
                storage_context=self.storage_context  # Use our synced storage context
            )
                
        except Exception as e:
            print(f"⚠️  Failed to load existing index: {e}")
            raise
            
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
