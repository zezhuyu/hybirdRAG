import re
from collections import defaultdict
from typing import Dict, List, Optional

import networkx as nx
try:
    from graspologic.partition import hierarchical_leiden
except ImportError:
    # graspologic not available, define dummy function
    def hierarchical_leiden(*args, **kwargs):
        raise ImportError("graspologic is required for hierarchical_leiden function. Please install it with: pip install graspologic")

from llama_index.core import Settings
from llama_index.core.llms import ChatMessage, LLM
try:
    from .custom_neo4j_store import CustomNeo4jPropertyGraphStore
    NEO4J_AVAILABLE = True
except ImportError as e:
    from llama_index.core.graph_stores.simple import SimpleGraphStore as CustomNeo4jPropertyGraphStore
    NEO4J_AVAILABLE = False
    print(f"⚠️  Neo4j not available, using SimpleGraphStore: {e}")


class GraphRAGStore(CustomNeo4jPropertyGraphStore):
    """Neo4j-backed property graph store with community summarisation support."""

    def __init__(
        self,
        *args,
        summarizer_llm: Optional[LLM] = None,
        max_cluster_size: int = 5,
        **kwargs,
    ) -> None:
        self.max_cluster_size = max_cluster_size
        self._summarizer_llm = summarizer_llm
        super().__init__(*args, **kwargs)
        self.community_summary: Dict[int, str] = {}
        self.entity_info: Dict[str, List[int]] = {}

    def set_summarizer_llm(self, llm: LLM) -> None:
        """Attach an LLM instance for community summarisation."""
        self._summarizer_llm = llm

    def generate_community_summary(self, text: str) -> str:
        """Generate summary for a given text using an LLM."""
        llm = self._summarizer_llm or Settings.llm
        if llm is None:
            raise ValueError(
                "No LLM available for community summarisation. Provide an LLM via "
                "GraphRAGStore(summarizer_llm=...) or llama_index Settings.llm."
            )

        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are provided with a set of relationships from a knowledge graph, each represented as "
                    "entity1->entity2->relation->relationship_description. Your task is to create a summary of these "
                    "relationships. The summary should include the names of the entities involved and a concise synthesis "
                    "of the relationship descriptions. The goal is to capture the most critical and relevant details that "
                    "highlight the nature and significance of each relationship. Ensure that the summary is coherent and "
                    "integrates the information in a way that emphasizes the key aspects of the relationships."
                ),
            ),
            ChatMessage(role="user", content=text),
        ]
        response = llm.chat(messages)
        clean_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return clean_response

    def build_communities(self, max_workers=10):
        """Builds communities from the graph and summarizes them."""
        nx_graph = self._create_nx_graph()
        if nx_graph.number_of_nodes() == 0:
            self.community_summary = {}
            self.entity_info = {}
            return
        community_hierarchical_clusters = hierarchical_leiden(
            nx_graph, max_cluster_size=self.max_cluster_size
        )
        self.entity_info, community_info = self._collect_community_info(
            nx_graph, community_hierarchical_clusters
        )
        self._summarize_communities(community_info, max_workers=max_workers)

    def _create_nx_graph(self):
        """Converts internal graph representation to NetworkX graph."""
        nx_graph = nx.Graph()
        try:
            # Try different method names for getting triplets
            if hasattr(self, 'get_triplets'):
                triplets = self.get_triplets()
            elif hasattr(self, 'get_all_triplets'):
                triplets = self.get_all_triplets()
            elif hasattr(self, 'get_relations'):
                triplets = self.get_relations()
            else:
                # Fallback: create empty graph
                print("⚠️  No triplets method found, creating empty graph")
                return nx_graph
            
            if len(triplets) == 0:
                print("⚠️  No triplets found - graph will be empty, no communities can be formed")
                return nx_graph
            
            for entity1, relation, entity2 in triplets:
                nx_graph.add_node(entity1.name)
                nx_graph.add_node(entity2.name)
                nx_graph.add_edge(
                    relation.source_id,
                    relation.target_id,
                    relationship=relation.label,
                    description=relation.properties.get("relationship_description", ""),
                )
        except Exception as e:
            print(f"⚠️  Error creating graph: {e}")
            return nx_graph
        
        return nx_graph

    def _collect_community_info(self, nx_graph, clusters):
        """
        Collect information for each node based on their community,
        allowing entities to belong to multiple clusters.
        """
        entity_info = defaultdict(set)
        community_info = defaultdict(list)

        for item in clusters:
            node = item.node
            cluster_id = item.cluster

            # Update entity_info
            entity_info[node].add(cluster_id)

            for neighbor in nx_graph.neighbors(node):
                edge_data = nx_graph.get_edge_data(node, neighbor)
                if edge_data:
                    detail = f"{node} -> {neighbor} -> {edge_data['relationship']} -> {edge_data['description']}"
                    community_info[cluster_id].append(detail)

        # Convert sets to lists for easier serialization if needed
        entity_info = {k: list(v) for k, v in entity_info.items()}

        return dict(entity_info), dict(community_info)

    def _summarize_communities(self, community_info, max_workers=5):
        """Generate and store summaries for each community using parallel processing."""
        import concurrent.futures
        import threading
        from typing import Dict, Tuple
        
        # Limit the number of communities to summarize to improve performance
        max_communities = 30  # Reduced for better performance
        sorted_communities = sorted(
            community_info.items(), 
            key=lambda x: len(x[1]), 
            reverse=True
        )[:max_communities]
        
        def process_community(community_data: Tuple[str, list]) -> Tuple[str, str]:
            """Process a single community and return (community_id, summary)."""
            community_id, details = community_data
            details_text = "\n".join(details) + "."  # Ensure it ends with a period
            try:
                summary = self.generate_community_summary(details_text)
                return community_id, summary
            except Exception as e:
                print(f"⚠️  Failed to summarize community {community_id}: {e}")
                return community_id, f"Error generating summary: {str(e)}"
        
        # Use ThreadPoolExecutor for parallel processing
        # max_workers is now a parameter with default value of 5
        completed = 0
        import time
        start_time = time.time()
        
        # Process communities silently to avoid interfering with progress bar
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_community = {
                executor.submit(process_community, community_data): community_data[0]
                for community_data in sorted_communities
            }
            
            # Tasks started silently
            
            # Process completed tasks as they finish
            for future in concurrent.futures.as_completed(future_to_community):
                community_id = future_to_community[future]
                try:
                    result_id, summary = future.result()
                    self.community_summary[result_id] = summary
                    completed += 1
                    # Show progress more frequently to demonstrate parallel processing
                    # Suppress progress messages to keep progress bar clean
                except Exception as e:
                    print(f"⚠️  Error processing community {community_id}: {e}")
                    completed += 1
        
        # Suppress community summary completion messages to keep progress bar clean

    def get_community_summaries(self, max_workers=10, fast_mode=True):
        """Returns the community summaries, with smart loading optimized for large datasets."""
        if not self.community_summary:
            # 🚀 Fast mode: Load intelligently based on dataset size
            if fast_mode:
                print("⚡ Fast mode enabled - using intelligent community loading")
                if self._load_communities_from_db():
                    print("✅ Loaded existing communities from database")
                    return self.community_summary
            else:
                # Traditional mode: Try to load all communities
                if self._load_all_communities_from_db():
                    print("✅ Loaded existing communities from database")
                    return self.community_summary
            
            # Only build if not found in database
            print("🔄 Building communities (this may take a while on first run)...")
            self.build_communities(max_workers=max_workers)
            
            # 🚀 Save to database after building
            self._save_communities_to_db()
            
        return self.community_summary

    def _load_all_communities_from_db(self):
        """Load ALL communities - use only when specifically needed."""
        try:
            import time
            start_time = time.time()
            print("⚠️  Loading ALL communities - this may take a while for large datasets...")
            
            # Load all communities (original behavior)
            query = "MATCH (c:Community) RETURN c.id as id, c.summary as summary"
            result = self._session.run(query)
            
            communities_found = 0
            for record in result:
                community_id = record["id"]
                summary = record["summary"]
                if community_id is not None and summary:
                    self.community_summary[community_id] = summary
                    communities_found += 1
                    
                    # Progress indicator for large datasets
                    if communities_found % 1000 == 0:
                        elapsed = time.time() - start_time
                        rate = communities_found / elapsed
                        print(f"   📊 Loaded {communities_found} communities ({rate:.0f} communities/sec)...")
            
            # Load all entities
            entity_query = "MATCH (e:Entity) RETURN e.name as name, e.communities as communities"
            entity_result = self._session.run(entity_query)
            
            entities_found = 0
            for record in entity_result:
                entity_name = record["name"]
                community_ids = record["communities"]
                if entity_name and community_ids:
                    self.entity_info[entity_name] = community_ids
                    entities_found += 1
            
            load_time = time.time() - start_time
            
            print(f"🚀 Performance: Loaded {communities_found} existing communities from database in {load_time:.3f}s (complete dataset)")
            print(f"   └── Also loaded {entities_found} entity mappings")
            return communities_found > 0
            
        except Exception as e:
            print(f"⚠️  Failed to load all communities from database: {e}")
            return False

    def _save_communities_to_db(self):
        """Save community summaries and entity info to Neo4j database."""
        try:
            import json
            
            # Save community summaries
            for community_id, summary in self.community_summary.items():
                query = """
                MERGE (c:Community {id: $community_id})
                SET c.summary = $summary,
                    c.updated_at = datetime()
                """
                self._session.run(query, community_id=community_id, summary=summary)
            
            # Save entity-community mappings
            for entity, community_ids in self.entity_info.items():
                query = """
                MERGE (e:Entity {name: $entity})
                SET e.communities = $community_ids,
                    e.updated_at = datetime()
                """
                self._session.run(query, entity=entity, community_ids=community_ids)
            
            print(f"💾 Saved {len(self.community_summary)} communities and {len(self.entity_info)} entity mappings to database")
            
        except Exception as e:
            print(f"⚠️  Failed to save communities to database: {e}")

    def _load_communities_from_db(self):
        """Load community summaries and entity info from Neo4j database with smart optimization for large datasets."""
        try:
            import time
            start_time = time.time()
            
            # 🚀 STEP 1: Check dataset size first
            count_query = "MATCH (c:Community) RETURN count(c) as community_count"
            count_result = self._session.run(count_query)
            community_count = count_result.single()["community_count"]
            
            print(f"📊 Found {community_count} communities in database")
            
            # 🚀 STEP 2: Smart loading strategy based on dataset size
            if community_count > 1000:
                print(f"⚡ Large dataset detected - using optimized loading strategy")
                return self._load_communities_optimized_for_large_dataset(community_count)
            else:
                print(f"⚡ Small dataset - using standard loading")
                return self._load_communities_standard()
                
        except Exception as e:
            print(f"⚠️  Optimized community loading failed: {e}")
            return self._load_communities_from_db_fallback()

    def _load_communities_optimized_for_large_dataset(self, total_count):
        """Optimized loading for large community datasets (>1000 communities)."""
        try:
            import time
            start_time = time.time()
            
            # 🚀 Strategy 1: Load only top communities by size/importance
            # Most relevant communities are typically larger (more entities)
            top_communities_query = """
            MATCH (c:Community)
            RETURN c.id as community_id, c.summary as community_summary
            ORDER BY length(c.summary) DESC
            LIMIT 500
            """
            
            print(f"⚡ Loading top 500 most important communities (out of {total_count})...")
            result = self._session.run(top_communities_query)
            
            communities_loaded = 0
            for record in result:
                community_id = record["community_id"]
                community_summary = record["community_summary"]
                if community_id is not None and community_summary:
                    self.community_summary[community_id] = community_summary
                    communities_loaded += 1
            
            # 🚀 Strategy 2: Load entity mappings for loaded communities only
            if communities_loaded > 0:
                loaded_ids = list(self.community_summary.keys())
                # Convert to string format for Neo4j query
                ids_str = ', '.join(map(str, loaded_ids))
                
                entity_query = f"""
                MATCH (e:Entity)
                WHERE any(community_id IN e.communities WHERE community_id IN [{ids_str}])
                RETURN e.name as entity_name, e.communities as entity_communities
                """
                
                entity_result = self._session.run(entity_query)
                entities_loaded = 0
                
                for record in entity_result:
                    entity_name = record["entity_name"]
                    entity_communities = record["entity_communities"]
                    if entity_name and entity_communities:
                        # Filter to only include loaded communities
                        filtered_communities = [c for c in entity_communities if c in loaded_ids]
                        if filtered_communities:
                            self.entity_info[entity_name] = filtered_communities
                            entities_loaded += 1
            
            load_time = time.time() - start_time
            
            if communities_loaded > 0:
                print(f"🚀 Performance: Loaded {communities_loaded} top communities from database in {load_time:.3f}s")
                print(f"   └── Optimization: Loaded {communities_loaded}/{total_count} communities ({communities_loaded/total_count*100:.1f}%)")
                print(f"   └── Also loaded {entities_loaded} relevant entity mappings")
                print(f"   └── Speed improvement: {total_count/communities_loaded:.1f}x faster than loading all communities")
                return True
            
            return False
            
        except Exception as e:
            print(f"⚠️  Optimized large dataset loading failed: {e}")
            return self._load_communities_standard()

    def _load_communities_standard(self):
        """Standard loading for smaller datasets or as fallback."""
        try:
            import time
            start_time = time.time()
            
            # Load communities with LIMIT for safety
            query = "MATCH (c:Community) RETURN c.id as id, c.summary as summary LIMIT 1000"
            result = self._session.run(query)
            
            communities_found = 0
            for record in result:
                community_id = record["id"]
                summary = record["summary"]
                if community_id is not None and summary:
                    self.community_summary[community_id] = summary
                    communities_found += 1
            
            # Load entities with LIMIT for safety
            entity_query = "MATCH (e:Entity) RETURN e.name as name, e.communities as communities LIMIT 5000"
            entity_result = self._session.run(entity_query)
            
            entities_found = 0
            for record in entity_result:
                entity_name = record["name"]
                community_ids = record["communities"]
                if entity_name and community_ids:
                    self.entity_info[entity_name] = community_ids
                    entities_found += 1
            
            load_time = time.time() - start_time
            
            if communities_found > 0:
                print(f"🚀 Performance: Loaded {communities_found} existing communities from database in {load_time:.3f}s (standard method)")
                if entities_found > 0:
                    print(f"   └── Also loaded {entities_found} entity mappings")
                return True
            
            return False
            
        except Exception as e:
            print(f"⚠️  Standard community loading failed: {e}")
            return False

    def _load_communities_from_db_fallback(self):
        """Fallback method using separate queries if the optimized single query fails."""
        try:
            print("🔄 Using fallback method with separate queries...")
            
            # Load community summaries
            query = "MATCH (c:Community) RETURN c.id as id, c.summary as summary"
            result = self._session.run(query)
            
            communities_found = 0
            for record in result:
                community_id = record["id"]
                summary = record["summary"]
                if community_id is not None and summary:
                    self.community_summary[community_id] = summary
                    communities_found += 1
            
            print(f"   └── Loaded {communities_found} communities")
            
            # Load entity-community mappings
            query = "MATCH (e:Entity) RETURN e.name as name, e.communities as communities"
            result = self._session.run(query)
            
            entities_found = 0
            for record in result:
                entity_name = record["name"]
                community_ids = record["communities"]
                if entity_name and community_ids:
                    self.entity_info[entity_name] = community_ids
                    entities_found += 1
            
            print(f"   └── Loaded {entities_found} entity mappings")
            
            if communities_found > 0:
                print(f"🚀 Performance: Loaded {communities_found} existing communities from database (fallback method)")
                return True
            
            return False
            
        except Exception as e:
            print(f"⚠️  Failed to load communities from database: {e}")
            return False

    def invalidate_communities(self):
        """Clear communities and force regeneration on next access."""
        self.community_summary.clear()
        self.entity_info.clear()
        
        # Only clear community data from database, NOT the entire graph
        try:
            # Only delete community nodes, leave entities and relationships intact
            self._session.run("MATCH (c:Community) DELETE c")
            # Communities invalidated silently
        except Exception as e:
            print(f"⚠️  Failed to clear communities from database: {e}")

    def _should_rebuild_communities(self):
        """Check if communities should be rebuilt based on graph changes."""
        try:
            # Simple heuristic: check if there are new entities/relations since last build
            # You could implement more sophisticated change detection here
            
            # Get current graph size
            current_triplets = len(list(self.get_triplets() if hasattr(self, 'get_triplets') else []))
            
            # Check if we have a record of previous size (suppress warnings)
            check_query = "OPTIONAL MATCH (m:Metadata {key: 'graph_size'}) RETURN m.value as size"
            result = self._session.run(check_query)
            record = result.single()
            
            if record and record["size"] is not None:
                previous_size = record["size"]
                change_threshold = 0.1  # Rebuild if graph grew by 10%
                
                if current_triplets > previous_size * (1 + change_threshold):
                    # Only log significant changes in debug mode
                    pass
                    return True
            
            # Update graph size record
            query = """
            MERGE (m:Metadata {key: 'graph_size'})
            SET m.value = $size, m.updated_at = datetime()
            """
            self._session.run(query, size=current_triplets)
            
            return False
            
        except Exception as e:
            print(f"⚠️  Error checking if communities should rebuild: {e}")
            return False
