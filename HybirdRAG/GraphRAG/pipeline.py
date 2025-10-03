import json
import re
import sys
from typing import Any, Dict, List, Optional

from llama_index.core import PropertyGraphIndex, StorageContext
from llama_index.core.llms import LLM
from llama_index.core.schema import BaseNode
# Use our fixed SimpleVectorStore that properly populates result.nodes
from .fixed_vector_store import FixedSimpleVectorStore as MilvusVectorStore

from .extractor import GraphRAGExtractor
from .query import GraphRAGQueryEngine
from .store import GraphRAGStore




KG_TRIPLET_EXTRACT_TMPL = """
Extract entities and relationships from the text. Return ONLY valid JSON.

Extract up to {max_knowledge_triplets} entities and their relationships.

For each entity, extract: entity_name, entity_type, entity_description
For each relationship, extract: source_entity, target_entity, relation, relationship_description

Return ONLY this JSON format (no other text):
{{
  "entities": [
    {{"entity_name": "Entity Name", "entity_type": "Type", "entity_description": "Description"}}
  ],
  "relationships": [
    {{"source_entity": "Source", "target_entity": "Target", "relation": "Relation", "relationship_description": "Description"}}
  ]
}}

If none found, return: {{"entities": [], "relationships": []}}

-An Output Example-
{{
  "entities": [
    {{
      "entity_name": "Albert Einstein",
      "entity_type": "Person",
      "entity_description": "Albert Einstein was a theoretical physicist who developed the theory of relativity and made significant contributions to physics."
    }},
    {{
      "entity_name": "Theory of Relativity",
      "entity_type": "Scientific Theory",
      "entity_description": "A scientific theory developed by Albert Einstein, describing the laws of physics in relation to observers in different frames of reference."
    }},
    {{
      "entity_name": "Nobel Prize in Physics",
      "entity_type": "Award",
      "entity_description": "A prestigious international award in the field of physics, awarded annually by the Royal Swedish Academy of Sciences."
    }}
  ],
  "relationships": [
    {{
      "source_entity": "Albert Einstein",
      "target_entity": "Theory of Relativity",
      "relation": "developed",
      "relationship_description": "Albert Einstein is the developer of the theory of relativity."
    }},
    {{
      "source_entity": "Albert Einstein",
      "target_entity": "Nobel Prize in Physics",
      "relation": "won",
      "relationship_description": "Albert Einstein won the Nobel Prize in Physics in 1921."
    }}
  ]
}}

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
        return entities, relationships
    
    # Extract everything from the first opening brace
    json_str = response_str[json_start:]
    
    # Try to fix common JSON truncation issues
    json_str = fix_truncated_json(json_str)
    
    # Clean common JSON formatting issues from LLM responses
    # Fix double braces at start/end
    if json_str.startswith('{{') and json_str.endswith('}}'):
        json_str = json_str[1:-1]  # Remove outer braces
    
    # Fix double braces within arrays (common LLM issue)
    import re
    json_str = re.sub(r'\{\{', '{', json_str)  # Replace {{ with {
    json_str = re.sub(r'\}\}', '}', json_str)  # Replace }} with }
    
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
        
        # Suppress parsing success messages to keep progress bar clean
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
        
        # If partial extraction also fails, return empty results (no retry to avoid infinite loops)
        print("⚠️  No valid data could be extracted from malformed JSON")
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
        
        # Ensure the vector store has access to the embedding model
        if embedding_model:
            self.vector_store._embed_model = embedding_model

        self.storage_context = StorageContext.from_defaults(
            graph_store=self.graph_store,
            vector_store=self.vector_store,
        )
        
        # Set the docstore on the vector store so it can populate result.nodes
        if hasattr(self.vector_store, 'set_docstore'):
            self.vector_store.set_docstore(self.storage_context.docstore)
        
        # Fix the VectorContextRetriever bug: patch the supports_vector_queries check
        self._patch_vector_context_retriever()

        # Set the LLM and embeddings in global settings for all llama-index operations
        from llama_index.core import Settings
        Settings.llm = llm
        if embedding_model:
            Settings.embed_model = embedding_model
            # Also set it on the vector store for direct access
            if hasattr(self.vector_store, '_embed_model'):
                self.vector_store._embed_model = embedding_model

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
        except Exception as e:
            print(f"⚠️  No existing GraphRAG index found: {e}")
            self.index = None
            
            # If no existing index, try to build one from Milvus data
            if self.milvus_client is not None:
                self._build_index_from_milvus()

    def _sync_vector_store_with_milvus(self):
        """Sync the GraphRAG vector store with data from Milvus collection."""
        try:
            from llama_index.core.schema import TextNode
            import time
            
            start_time = time.time()
            
            # Get collection info
            collection_name = self.milvus_client.collection_name if hasattr(self.milvus_client, 'collection_name') else 'vector_rag'
            
            # 🚀 Optimized query with progress indicator
            results = self.milvus_client.query(
                collection_name=collection_name,
                filter="",  # Get all documents
                output_fields=["content", "title", "page", "source", "vector"],  # 🚀 Include existing embeddings
                limit=1000  # Reasonable limit
            )
            
            query_time = time.time() - start_time
            
            # Clear existing data
            self.vector_store._data.embedding_dict.clear()
            self.vector_store._data.text_id_to_ref_doc_id.clear()
            self.vector_store._data.metadata_dict.clear()
            
            # Prepare nodes for batch processing
            nodes_to_add = []
            embeddings_needed = []
            contents_for_embedding = []
            
            for i, result in enumerate(results):
                content = result.get('content', '')
                existing_vector = result.get('vector')  # 🚀 Check for existing embedding
                
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
                    
                    if existing_vector and len(existing_vector) > 0:
                        # 🚀 Use existing embedding - no network call needed!
                        node.embedding = existing_vector
                        nodes_to_add.append(node)
                    else:
                        # Need to generate embedding
                        embeddings_needed.append((i, node))
                        contents_for_embedding.append(content)
            
            # Generate embeddings for documents that don't have them
            if contents_for_embedding:
                
                if self.embedding_model and hasattr(self.embedding_model, 'ml_model_client'):
                    try:
                        # Check if batch embedding is available
                        if hasattr(self.embedding_model.ml_model_client, 'embed_sentences'):
                            embeddings = self.embedding_model.ml_model_client.embed_sentences(contents_for_embedding)
                        else:
                            embeddings = []
                            for j, content in enumerate(contents_for_embedding):
                                embedding = self.embedding_model.ml_model_client.embed_sentence(content)
                                embeddings.append(embedding)
                        
                        # Add embeddings to nodes
                        for (i, node), embedding in zip(embeddings_needed, embeddings):
                            node.embedding = embedding
                            nodes_to_add.append(node)
                        
                            
                    except Exception as embed_error:
                        print(f"⚠️  Failed to generate embeddings: {embed_error}")
                        # Continue without these documents
                else:
                    print(f"⚠️  No embedding model available, skipping {len(contents_for_embedding)} documents")
            
            reused_count = len(nodes_to_add) - len(contents_for_embedding)
            
            # 🚀 Batch insert all nodes at once
            if nodes_to_add:
                self.vector_store.add(nodes_to_add)
                self.storage_context.docstore.add_documents(nodes_to_add)
                
                # Ensure proper mapping in vector store
                if hasattr(self.vector_store, '_data'):
                    for node in nodes_to_add:
                        self.vector_store._data.text_id_to_ref_doc_id[node.node_id] = node.node_id
                
                total_time = time.time() - start_time
            else:
                print("⚠️  No nodes to add")
            
        except Exception as e:
            print(f"⚠️  Failed to sync vector store with Milvus: {e}")
            import traceback
            traceback.print_exc()

    def _build_index_from_milvus(self):
        """Build GraphRAG index from Milvus documents."""
        try:
            self._sync_vector_store_with_milvus()
            
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
            
            # Build the index
            if nodes:
                self.build_index(nodes)
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
            nodes, show_progress=False
        )

        if self.index is None:
            self.index = PropertyGraphIndex(
                nodes=processed_nodes,
                storage_context=self.storage_context,
                property_graph_store=self.graph_store,  # Explicitly pass our custom store
                show_progress=False,
            )
        else:
            self.index.insert_nodes(processed_nodes)
        return self.index

    def load_index(self) -> PropertyGraphIndex:
        """Load an existing index definition from the persistent stores."""
        # First sync the vector store with Milvus data before loading the index
        if self.milvus_client is not None:
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

    def fast_query(self, query: str, max_communities: int = 5) -> Dict[str, Any]:
        """
        🚀 Ultra-fast query that returns raw data in 1-2 seconds.
        Skips expensive LLM processing during retrieval - perfect for downstream LLM processing.
        """
        import time
        from typing import Dict, Any, List
        
        start_time = time.time()
        
        if self.index is None:
            try:
                self.load_index()
            except ValueError as exc:
                raise ValueError("Index is not built and no persisted index found.") from exc
        
        try:
            # 🚀 Step 1: Fast entity extraction (simplified)
            entities = self._fast_extract_entities(query)
            step1_time = time.time() - start_time
            
            # 🚀 Step 2: Lightning-fast community lookup
            community_data = self._fast_get_communities(entities, max_communities)
            step2_time = time.time() - start_time
            
            # 🚀 Step 3: Return raw data (no LLM calls!)
            result = {
                "query": query,
                "entities": entities[:10],
                "communities": community_data,
                "total_retrieval_time": time.time() - start_time,
                "performance_breakdown": {
                    "entity_extraction": step1_time,
                    "community_lookup": step2_time - step1_time,
                    "data_formatting": time.time() - start_time - step2_time
                },
                "optimization_note": "Raw data returned - no LLM processing during retrieval"
            }
            
            
            return result
            
        except Exception as e:
            print(f"⚠️  Fast query failed: {e}")
            return {
                "query": query,
                "entities": [],
                "communities": [],
                "total_retrieval_time": time.time() - start_time,
                "error": str(e)
            }

    def _fast_extract_entities(self, query: str) -> List[str]:
        """Fast entity extraction optimized for speed."""
        import re
        
        try:
            # Use minimal similarity search for speed
            nodes_retrieved = self.index.as_retriever(similarity_top_k=5).retrieve(query)
            
            entities = set()
            
            # Extract entities from top nodes only
            for node in nodes_retrieved[:3]:  # Limit for speed
                # Structured extraction (original method)
                matches = re.findall(
                    r"^(\w+(?:\s+\w+)*)\s*->\s*([a-zA-Z\s]+?)\s*->\s*(\w+(?:\s+\w+)*)$",
                    node.text, re.MULTILINE | re.IGNORECASE
                )
                
                for match in matches:
                    entities.add(match[0].strip())
                    entities.add(match[2].strip())
            
            # Fallback: extract from query itself
            query_entities = re.findall(r'\b([A-Z][a-z]*(?:\s+[A-Z][a-z]*)*)\b', query)
            entities.update(query_entities)
            
            # Filter and limit
            entity_list = [e for e in entities if len(e) > 2][:15]
            return entity_list
            
        except Exception as e:
            # Emergency fallback
            return re.findall(r'\b([A-Z][a-z]+)\b', query)[:5]

    def _fast_get_communities(self, entities: List[str], max_communities: int = 5) -> List[Dict[str, Any]]:
        """Lightning-fast community retrieval."""
        try:
            # Get cached community summaries
            all_communities = self.graph_store.get_community_summaries()
            entity_info = self.graph_store.entity_info
            
            community_scores = {}
            
            # Score communities based on entity matches
            for entity in entities:
                if entity in entity_info:
                    for comm_id in entity_info[entity]:
                        if comm_id in all_communities:
                            if comm_id not in community_scores:
                                community_scores[comm_id] = {
                                    "score": 0,
                                    "matched_entities": []
                                }
                            community_scores[comm_id]["score"] += 1
                            community_scores[comm_id]["matched_entities"].append(entity)
            
            # Create result list with summaries
            results = []
            for comm_id, data in community_scores.items():
                summary = all_communities[comm_id]
                results.append({
                    "community_id": comm_id,
                    "summary": summary,
                    "score": data["score"],
                    "matched_entities": data["matched_entities"],
                    "summary_preview": summary[:200] + "..." if len(summary) > 200 else summary
                })
            
            # Sort by score and limit
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:max_communities]
            
        except Exception as e:
            print(f"⚠️  Community retrieval failed: {e}")
            return []
    
    def _patch_vector_context_retriever(self):
        """Patch the VectorContextRetriever to fix the supports_vector_queries bug."""
        try:
            from llama_index.core.indices.property_graph.sub_retrievers.vector import VectorContextRetriever
            
            # Store the original methods
            original_retrieve_from_graph = VectorContextRetriever.retrieve_from_graph
            original_aretrieve_from_graph = VectorContextRetriever.aretrieve_from_graph
            
            def patched_retrieve_from_graph(self, query_bundle, limit=None):
                """Patched version that correctly calls supports_vector_queries()."""
                from llama_index.core.vector_stores.types import VectorStoreQuery
                from llama_index.core.schema import NodeWithScore
                
                vector_store_query = self._get_vector_store_query(query_bundle)

                triplets = []
                kg_ids = []
                new_scores = []
                
                # FIXED: Call supports_vector_queries() instead of just checking if it exists
                if self._graph_store.supports_vector_queries():
                    result = self._graph_store.vector_query(vector_store_query)
                    if len(result) != 2:
                        raise ValueError("No nodes returned by vector_query")
                    kg_nodes, scores = result

                    kg_ids = [node.id for node in kg_nodes]
                    triplets = self._graph_store.get_rel_map(
                        kg_nodes,
                        depth=self._path_depth,
                        limit=limit or self._limit,
                        ignore_rels=["__kg_source__"],
                    )

                elif self._vector_store is not None:
                    query_result = self._vector_store.query(vector_store_query)
                    if query_result.nodes is not None and query_result.similarities is not None:
                        kg_ids = self._get_kg_ids(query_result.nodes)
                        scores = query_result.similarities
                        kg_nodes = self._graph_store.get(ids=kg_ids)
                        triplets = self._graph_store.get_rel_map(
                            kg_nodes,
                            depth=self._path_depth,
                            limit=limit or self._limit,
                            ignore_rels=["__kg_source__"],
                        )

                    elif query_result.ids is not None and query_result.similarities is not None:
                        kg_ids = query_result.ids
                        scores = query_result.similarities
                        kg_nodes = self._graph_store.get(ids=kg_ids)
                        triplets = self._graph_store.get_rel_map(
                            kg_nodes,
                            depth=self._path_depth,
                            limit=limit or self._limit,
                            ignore_rels=["__kg_source__"],
                        )

                # Rest of the method remains the same
                for triplet in triplets:
                    score1 = (
                        scores[kg_ids.index(triplet[0].id)] if triplet[0].id in kg_ids else 0.0
                    )
                    score2 = (
                        scores[kg_ids.index(triplet[2].id)] if triplet[2].id in kg_ids else 0.0
                    )
                    new_scores.append((score1 + score2) / 2.0)

                nodes = []
                for i, triplet in enumerate(triplets):
                    node = NodeWithScore(
                        node=triplet[1], score=new_scores[i]
                    )  # triplet[1] is the relation
                    nodes.append(node)

                return nodes
            
            async def patched_aretrieve_from_graph(self, query_bundle, limit=None):
                """Patched async version that correctly calls supports_vector_queries()."""
                from llama_index.core.vector_stores.types import VectorStoreQuery
                from llama_index.core.schema import NodeWithScore
                
                vector_store_query = self._get_vector_store_query(query_bundle)

                triplets = []
                kg_ids = []
                new_scores = []
                
                # FIXED: Call supports_vector_queries() instead of just checking if it exists
                if self._graph_store.supports_vector_queries():
                    result = self._graph_store.vector_query(vector_store_query)
                    if len(result) != 2:
                        raise ValueError("No nodes returned by vector_query")
                    kg_nodes, scores = result

                    kg_ids = [node.id for node in kg_nodes]
                    triplets = self._graph_store.get_rel_map(
                        kg_nodes,
                        depth=self._path_depth,
                        limit=limit or self._limit,
                        ignore_rels=["__kg_source__"],
                    )

                elif self._vector_store is not None:
                    query_result = self._vector_store.query(vector_store_query)
                    if query_result.nodes is not None and query_result.similarities is not None:
                        kg_ids = self._get_kg_ids(query_result.nodes)
                        scores = query_result.similarities
                        kg_nodes = self._graph_store.get(ids=kg_ids)
                        triplets = self._graph_store.get_rel_map(
                            kg_nodes,
                            depth=self._path_depth,
                            limit=limit or self._limit,
                            ignore_rels=["__kg_source__"],
                        )

                    elif query_result.ids is not None and query_result.similarities is not None:
                        kg_ids = query_result.ids
                        scores = query_result.similarities
                        kg_nodes = self._graph_store.get(ids=kg_ids)
                        triplets = self._graph_store.get_rel_map(
                            kg_nodes,
                            depth=self._path_depth,
                            limit=limit or self._limit,
                            ignore_rels=["__kg_source__"],
                        )

                # Rest of the method remains the same
                for triplet in triplets:
                    score1 = (
                        scores[kg_ids.index(triplet[0].id)] if triplet[0].id in kg_ids else 0.0
                    )
                    score2 = (
                        scores[kg_ids.index(triplet[2].id)] if triplet[2].id in kg_ids else 0.0
                    )
                    new_scores.append((score1 + score2) / 2.0)

                nodes = []
                for i, triplet in enumerate(triplets):
                    node = NodeWithScore(
                        node=triplet[1], score=new_scores[i]
                    )  # triplet[1] is the relation
                    nodes.append(node)

                return nodes
            
            # Apply the patches
            VectorContextRetriever.retrieve_from_graph = patched_retrieve_from_graph
            VectorContextRetriever.aretrieve_from_graph = patched_aretrieve_from_graph
            
        except Exception as e:
            print(f"⚠️  Failed to patch VectorContextRetriever: {e}")
