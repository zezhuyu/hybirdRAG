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

IMPORTANT: Return ONLY valid JSON. No explanations, no markdown, no code blocks. Just the JSON.

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

Example output:
{
  "entities": [
    {
      "entity_name": "Albert Einstein",
      "entity_type": "Person",
      "entity_description": "Albert Einstein was a theoretical physicist who developed the theory of relativity and made significant contributions to physics."
    }
  ],
  "relationships": [
    {
      "source_entity": "Albert Einstein",
      "target_entity": "Theory of Relativity",
      "relation": "developed",
      "relationship_description": "Albert Einstein is the developer of the theory of relativity."
    }
  ]
}

Text to analyze:
{text}

JSON output:"""

def parse_fn(response_str: str) -> Any:
    """Parse the LLM response to extract entities and relationships with robust error handling."""
    entities = []
    relationships = []
    
    # Try to find JSON in the response - look for opening brace
    json_start = response_str.find('{')
    if json_start == -1:
        # Try to extract data using regex patterns even without JSON structure
        try:
            partial_entities, partial_relationships = extract_partial_json(response_str)
            return partial_entities, partial_relationships
        except:
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
    
    # Try multiple parsing strategies
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
            if entity.get("entity_name") and entity.get("entity_name").strip()
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
            if relation.get("source_entity") and relation.get("target_entity") and relation.get("relation")
        ]
        
        # Suppress parsing success messages to keep progress bar clean
        return entities, relationships
        
    except json.JSONDecodeError as e:
        # Try to extract partial data from malformed JSON
        try:
            partial_entities, partial_relationships = extract_partial_json(response_str)
            if partial_entities or partial_relationships:
                return partial_entities, partial_relationships
        except:
            pass
        
        # If partial extraction also fails, return empty results (no retry to avoid infinite loops)
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
        # Use smaller batches for better performance and error handling
        batch_size = 20  # Increased batch size for efficiency
        processed_nodes = []
        total_batches = (len(nodes) + batch_size - 1) // batch_size
        
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            try:
                batch_processed = self.kg_extractor(batch, show_progress=False)
                processed_nodes.extend(batch_processed)
                
            except Exception as batch_error:
                print(f"⚠️ Batch {batch_num} failed: {batch_error}")
                print(f"🔄 Using original nodes for batch {batch_num}...")
                processed_nodes.extend(batch)  # Use original nodes as fallback

        if self.index is None:
            # Ensure entities are properly processed before creating the index
            self._process_extracted_entities(processed_nodes)
            
            self.index = PropertyGraphIndex(
                nodes=processed_nodes,
                storage_context=self.storage_context,
                property_graph_store=self.graph_store,  # Explicitly pass our custom store
                show_progress=False,
            )
        else:
            self.index.insert_nodes(processed_nodes)
        return self.index

    def _process_extracted_entities(self, nodes: List[BaseNode]) -> None:
        """Process extracted entities and ensure they are properly stored in the graph."""
        try:
            total_entities = 0
            total_relations = 0
            
            for node in nodes:
                if hasattr(node, 'metadata'):
                    # Extract entities and relations from metadata
                    entities = node.metadata.get('nodes', [])
                    relations = node.metadata.get('relations', [])
                    
                    if entities:
                        total_entities += len(entities)
                        # Store entities in the graph store
                        try:
                            self.graph_store.upsert_nodes(entities)
                        except Exception as e:
                            print(f"⚠️ Failed to store entities: {e}")
                    
                    if relations:
                        total_relations += len(relations)
                        # Store relations in the graph store
                        try:
                            self.graph_store.upsert_relations(relations)
                        except Exception as e:
                            print(f"⚠️ Failed to store relations: {e}")
            
            if total_entities > 0:
                print(f"✅ Processed {total_entities} entities and {total_relations} relations")
            else:
                print(f"⚠️ No entities found in {len(nodes)} processed nodes")
                
        except Exception as e:
            print(f"⚠️ Error processing extracted entities: {e}")

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

    def _deduplicate_graph_results(self, results: List[str]) -> List[str]:
        """Deduplicate GraphRAG results based on content similarity."""
        if not results:
            return results
        
        # For query broadening, we want to keep diverse responses even if they're similar
        # Only remove exact duplicates or very short responses
        deduplicated = []
        seen_content = set()
        
        for result in results:
            if result and result.strip() and len(result.strip()) > 50:  # Keep substantial responses
                # Only remove exact duplicates (normalized)
                normalized_content = result.strip().lower()
                if normalized_content not in seen_content:
                    seen_content.add(normalized_content)
                    deduplicated.append(result)
        
        return deduplicated

    def _get_nodes_for_query(self, query_str: str):
        """Get nodes and their IDs for a query."""
        # Get all nodes quickly
        all_nodes = list(self.index.docstore.docs.values())
        
        # Filter nodes by query relevance using improved matching
        query_terms = query_str.lower().split()
        # Remove common stop words for better matching, but keep question words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can'}
        # Keep important question words and terms longer than 1 character
        query_terms = [term for term in query_terms if term not in stop_words and len(term) > 1]
        
        # If no terms remain after filtering, use the original query as a fallback
        if not query_terms:
            query_terms = [query_str.lower().strip()]
        
        relevant_nodes = []
        
        for node in all_nodes:
            if hasattr(node, 'text') and node.text:
                node_text = node.text.lower()
                
                # Multiple matching strategies
                match_score = 0
                
                # Strategy 1: Exact term matching
                for term in query_terms:
                    if term in node_text:
                        match_score += 1
                
                # Strategy 2: Partial word matching (for compound terms)
                for term in query_terms:
                    if any(term in word or word in term for word in node_text.split() if len(word) > 3):
                        match_score += 0.5
                
                # Strategy 3: Semantic similarity (simple keyword expansion)
                semantic_groups = {
                    'person': ['person', 'people', 'individual', 'man', 'woman', 'human'],
                    'place': ['place', 'location', 'city', 'country', 'state', 'region'],
                    'time': ['time', 'year', 'date', 'period', 'era', 'century'],
                    'event': ['event', 'incident', 'happening', 'occurrence', 'situation']
                }
                
                for term in query_terms:
                    for category, related_words in semantic_groups.items():
                        if term in related_words:
                            for related_word in related_words:
                                if related_word in node_text:
                                    match_score += 0.3
                                    break
                
                # Accept nodes with any match
                if match_score > 0:
                    relevant_nodes.append((node, getattr(node, 'id_', getattr(node, 'node_id', None))))
        
        if len(relevant_nodes) == 0:
            # Fallback: use first few nodes
            relevant_nodes = [(node, getattr(node, 'id_', getattr(node, 'node_id', None))) for node in all_nodes[:5]]
            print(f"⚠️  No relevant nodes found for query '{query_str}' with terms {query_terms}, using fallback")
            print(f"   Total nodes available: {len(all_nodes)}")
            if all_nodes:
                sample_node_text = all_nodes[0].text[:200] if hasattr(all_nodes[0], 'text') and all_nodes[0].text else "No text"
                print(f"   Sample node text: {sample_node_text}...")
        
        return relevant_nodes

    def _generate_response_from_nodes(self, nodes, query_str):
        """Generate response from a list of nodes."""
        # Extract and clean text
        relevant_texts = []
        for node in nodes:
            if hasattr(node, 'text') and node.text:
                clean_text = node.text.strip()
                if clean_text and len(clean_text) > 50:  # Filter out very short text
                    relevant_texts.append(clean_text)
        
        if not relevant_texts:
            return ""
        
        # Join texts and return (similar to query engine logic)
        combined_text = " ".join(relevant_texts[:5])  # Limit to top 5 texts
        return combined_text[:2000]  # Limit length

    def query(self, query: List[str]) -> str:
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

        result = []
        
        for q in query:
            # Get nodes and their IDs from the query engine
            nodes_with_ids = self._get_nodes_for_query(q)
            
            if nodes_with_ids:
                # Generate response from all nodes for this query (no cross-query deduplication)
                nodes = [node for node, node_id in nodes_with_ids]
                response = self._generate_response_from_nodes(nodes, q)
                if response and response.strip():
                    result.append(response)
        
        # For query broadening, keep all responses to provide diverse perspectives
        # Only deduplicate if we have many responses (>10) to avoid overwhelming the user
        if len(result) > 10:
            deduplicated_result = self._deduplicate_graph_results(result)
            return deduplicated_result
        else:
            return result

    # def fast_query(self, query: str, max_communities: int = 5) -> Dict[str, Any]:
    #     """
    #     🚀 Ultra-fast query that returns raw data in 1-2 seconds.
    #     Skips expensive LLM processing during retrieval - perfect for downstream LLM processing.
    #     """
    #     import time
    #     from typing import Dict, Any, List
        
    #     start_time = time.time()
        
    #     if self.index is None:
    #         try:
    #             self.load_index()
    #         except ValueError as exc:
    #             raise ValueError("Index is not built and no persisted index found.") from exc
        
    #     try:
    #         # 🚀 Step 1: Fast entity extraction (simplified)
    #         entities = self._fast_extract_entities(query)
    #         step1_time = time.time() - start_time
            
    #         # 🚀 Step 2: Lightning-fast community lookup
    #         community_data = self._fast_get_communities(entities, max_communities)
    #         step2_time = time.time() - start_time
            
    #         # 🚀 Step 3: Return raw data (no LLM calls!)
    #         result = {
    #             "query": query,
    #             "entities": entities[:10],
    #             "communities": community_data,
    #             "total_retrieval_time": time.time() - start_time,
    #             "performance_breakdown": {
    #                 "entity_extraction": step1_time,
    #                 "community_lookup": step2_time - step1_time,
    #                 "data_formatting": time.time() - start_time - step2_time
    #             },
    #             "optimization_note": "Raw data returned - no LLM processing during retrieval"
    #         }
            
            
    #         return result
            
    #     except Exception as e:
    #         print(f"⚠️  Fast query failed: {e}")
    #         return {
    #             "query": query,
    #             "entities": [],
    #             "communities": [],
    #             "total_retrieval_time": time.time() - start_time,
    #             "error": str(e)
    #         }

    # def _fast_extract_entities(self, query: str) -> List[str]:
    #     """Fast entity extraction optimized for speed."""
    #     import re
        
    #     try:
    #         # Use minimal similarity search for speed
    #         nodes_retrieved = self.index.as_retriever(similarity_top_k=5).retrieve(query)
            
    #         entities = set()
            
    #         # Extract entities from top nodes only
    #         for node in nodes_retrieved[:3]:  # Limit for speed
    #             # Structured extraction (original method)
    #             matches = re.findall(
    #                 r"^(\w+(?:\s+\w+)*)\s*->\s*([a-zA-Z\s]+?)\s*->\s*(\w+(?:\s+\w+)*)$",
    #                 node.text, re.MULTILINE | re.IGNORECASE
    #             )
                
    #             for match in matches:
    #                 entities.add(match[0].strip())
    #                 entities.add(match[2].strip())
            
    #         # Fallback: extract from query itself
    #         query_entities = re.findall(r'\b([A-Z][a-z]*(?:\s+[A-Z][a-z]*)*)\b', query)
    #         entities.update(query_entities)
            
    #         # Filter and limit
    #         entity_list = [e for e in entities if len(e) > 2][:15]
    #         return entity_list
            
    #     except Exception as e:
    #         # Emergency fallback
    #         return re.findall(r'\b([A-Z][a-z]+)\b', query)[:5]

    # def _fast_get_communities(self, entities: List[str], max_communities: int = 5) -> List[Dict[str, Any]]:
    #     """Lightning-fast community retrieval."""
    #     try:
    #         # Get cached community summaries
    #         all_communities = self.graph_store.get_community_summaries()
    #         entity_info = self.graph_store.entity_info
            
    #         community_scores = {}
            
    #         # Score communities based on entity matches
    #         for entity in entities:
    #             if entity in entity_info:
    #                 for comm_id in entity_info[entity]:
    #                     if comm_id in all_communities:
    #                         if comm_id not in community_scores:
    #                             community_scores[comm_id] = {
    #                                 "score": 0,
    #                                 "matched_entities": []
    #                             }
    #                         community_scores[comm_id]["score"] += 1
    #                         community_scores[comm_id]["matched_entities"].append(entity)
            
    #         # Create result list with summaries
    #         results = []
    #         for comm_id, data in community_scores.items():
    #             summary = all_communities[comm_id]
    #             results.append({
    #                 "community_id": comm_id,
    #                 "summary": summary,
    #                 "score": data["score"],
    #                 "matched_entities": data["matched_entities"],
    #                 "summary_preview": summary[:200] + "..." if len(summary) > 200 else summary
    #             })
            
    #         # Sort by score and limit
    #         results.sort(key=lambda x: x["score"], reverse=True)
    #         return results[:max_communities]
            
    #     except Exception as e:
    #         print(f"⚠️  Community retrieval failed: {e}")
    #         return []
    
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
