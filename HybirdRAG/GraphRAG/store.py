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
    print("✅ Custom Neo4j graph store available")
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

    def build_communities(self):
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
        self._summarize_communities(community_info)

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

    def _summarize_communities(self, community_info):
        """Generate and store summaries for each community."""
        for community_id, details in community_info.items():
            details_text = (
                "\n".join(details) + "."
            )  # Ensure it ends with a period
            self.community_summary[
                community_id
            ] = self.generate_community_summary(details_text)

    def get_community_summaries(self):
        """Returns the community summaries, building them if not already done."""
        if not self.community_summary:
            self.build_communities()
        return self.community_summary
