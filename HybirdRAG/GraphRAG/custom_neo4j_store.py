"""
Custom Neo4j Graph Store for ReqQuest

This module provides a custom Neo4j graph store implementation that integrates
with llama-index's PropertyGraphStore interface.
"""

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from neo4j import GraphDatabase
from llama_index.core.graph_stores.types import (
    EntityNode,
    Relation,
    PropertyGraphStore,
)
from llama_index.core.schema import BaseNode


class CustomNeo4jPropertyGraphStore(PropertyGraphStore):
    """Custom Neo4j Property Graph Store implementation."""
    
    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
        **kwargs: Any,
    ) -> None:
        """Initialize the Neo4j graph store.
        
        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            username: Neo4j username
            password: Neo4j password
            database: Neo4j database name (default: "neo4j")
        """
        super().__init__()
        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        self._database = database
        self._session = self._driver.session(database=database)
        
    def close(self) -> None:
        """Close the Neo4j connection."""
        if self._session:
            self._session.close()
        if self._driver:
            self._driver.close()
    
    def get(self, subj: str = None, ids: List[str] = None) -> List[str]:
        """Get all objects that are connected to the subject or by IDs."""
        if ids is not None:
            # Handle the case where PropertyGraphIndex calls get(ids=list_of_ids)
            if not ids:
                return []
            # For now, return empty list as we don't have a direct ID lookup
            # This might need to be implemented based on llama-index requirements
            return []
        
        if subj is None:
            return []
            
        query = "MATCH (s {name: $subj})-[:RELATED_TO]->(o) RETURN o.name as obj_name"
        result = self._session.run(query, subj=subj)
        return [record["obj_name"] for record in result]
    
    def get_rel_map(
        self, subj_entities: List[str], depth: int = 2, limit: int = 30
    ) -> Dict[str, List[List[str]]]:
        """Get relationship map for given subjects."""
        rel_map = {}
        for subj in subj_entities:
            query = """
            MATCH path = (s {name: $subj})-[:RELATED_TO*1..2]-(o)
            WHERE s.name = $subj
            RETURN [node in nodes(path) | node.name] as path_nodes
            LIMIT $limit
            """
            result = self._session.run(query, subj=subj, limit=limit)
            paths = [record["path_nodes"] for record in result]
            rel_map[subj] = paths
        return rel_map
    
    def upsert_nodes(self, nodes: List[BaseNode]) -> None:
        """Upsert nodes into the Neo4j database."""
        print(f"🔍 upsert_nodes called with {len(nodes)} nodes")
        for node in nodes:
            if isinstance(node, EntityNode):
                print(f"  Upserting EntityNode: {node.name} ({node.label})")
                self._upsert_entity_node(node)
            else:
                print(f"  Skipping non-EntityNode: {type(node)}")
    
    def _upsert_entity_node(self, node: EntityNode) -> None:
        """Upsert an entity node into Neo4j."""
        # Sanitize the label for Neo4j (replace spaces and special chars with underscores)
        sanitized_label = self._sanitize_label(node.label)
        
        # Create or update the node with the proper label
        query = f"""
        MERGE (n:{sanitized_label} {{name: $name}})
        SET n.description = $description,
            n.properties = $properties,
            n.original_label = $original_label
        """
        properties = node.properties or {}
        self._session.run(
            query,
            name=node.name,
            description=properties.get("entity_description", ""),
            properties=json.dumps(properties),
            original_label=node.label  # Store original label as property
        )
    
    def _sanitize_label(self, label: str) -> str:
        """Sanitize label for Neo4j compatibility."""
        if not label:
            return "Entity"
        # Replace spaces and special characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', label)
        # Remove consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        # Ensure it starts with a letter
        if sanitized and not sanitized[0].isalpha():
            sanitized = 'Entity_' + sanitized
        # Fallback if empty
        if not sanitized:
            sanitized = "Entity"
        return sanitized
    
    def upsert_relations(self, relations: List[Relation]) -> None:
        """Upsert relations into the Neo4j database."""
        print(f"🔍 upsert_relations called with {len(relations)} relations")
        for relation in relations:
            print(f"  Upserting Relation: {relation.source_id} → {relation.label} → {relation.target_id}")
            self._upsert_relation(relation)
    
    def _upsert_relation(self, relation: Relation) -> None:
        """Upsert a relation into Neo4j."""
        query = """
        MATCH (s {name: $source_id})
        MATCH (t {name: $target_id})
        MERGE (s)-[r:RELATED_TO]->(t)
        SET r.relationship = $label,
            r.description = $description,
            r.properties = $properties
        """
        properties = relation.properties or {}
        self._session.run(
            query,
            source_id=relation.source_id,
            target_id=relation.target_id,
            label=relation.label,
            description=properties.get("relationship_description", ""),
            properties=json.dumps(properties)
        )
    
    def delete(self, subj: str, rel: str, obj: str) -> None:
        """Delete a relation from the Neo4j database."""
        query = """
        MATCH (s {name: $subj})-[r:RELATED_TO]->(o {name: $obj})
        WHERE r.relationship = $rel
        DELETE r
        """
        self._session.run(query, subj=subj, rel=rel, obj=obj)
    
    def get_triplets(self) -> List[Tuple[EntityNode, Relation, EntityNode]]:
        """Get all triplets from the Neo4j database."""
        query = """
        MATCH (s)-[r]->(o)
        RETURN s.name as source_name, labels(s) as source_labels, properties(s) as source_properties,
               type(r) as relation_type, properties(r) as relation_properties,
               o.name as target_name, labels(o) as target_labels, properties(o) as target_properties
        """
        try:
            result = self._session.run(query)
            triplets = []
            for record in result:
                source_node = EntityNode(
                    name=record["source_name"],
                    label=record["source_labels"][0] if record["source_labels"] else "Node",
                    properties=record["source_properties"]
                )
                target_node = EntityNode(
                    name=record["target_name"],
                    label=record["target_labels"][0] if record["target_labels"] else "Node",
                    properties=record["target_properties"]
                )
                relation = Relation(
                    label=record["relation_type"],
                    source_id=record["source_name"],
                    target_id=record["target_name"],
                    properties=record["relation_properties"]
                )
                triplets.append((source_node, relation, target_node))
            return triplets
        except Exception as e:
            print(f"❌ Error getting triplets from Neo4j: {e}")
            return []
    
    def get_all_triplets(self) -> List[Tuple[EntityNode, Relation, EntityNode]]:
        """Alias for get_triplets."""
        return self.get_triplets()
    
    def get_relations(self) -> List[Relation]:
        """Get all relations from the Neo4j database."""
        query = """
        MATCH (s)-[r:RELATED_TO]->(o)
        RETURN s.name as source_id, r.relationship as label, o.name as target_id, r.properties as properties
        """
        result = self._session.run(query)
        relations = []
        for record in result:
            properties = json.loads(record["properties"]) if record["properties"] else {}
            relation = Relation(
                source_id=record["source_id"],
                target_id=record["target_id"],
                label=record["label"],
                properties=properties
            )
            relations.append(relation)
        return relations
    
    def supports_vector_queries(self) -> bool:
        """Return True if the graph store supports vector queries."""
        return False
    
    def get_schema(self, refresh: bool = False) -> Dict[str, Any]:
        """Get the schema of the Neo4j database."""
        # Get all node labels
        labels_query = "CALL db.labels()"
        labels_result = self._session.run(labels_query)
        labels = [record["label"] for record in labels_result]
        
        # Get all relationship types
        rel_types_query = "CALL db.relationshipTypes()"
        rel_types_result = self._session.run(rel_types_query)
        relationship_types = [record["relationshipType"] for record in rel_types_result]
        
        return {
            "node_labels": labels,
            "relationship_types": relationship_types,
        }
    
    def structured_query(self, query: str, param_map: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a structured query against the Neo4j database."""
        if param_map is None:
            param_map = {}
        result = self._session.run(query, param_map)
        return [dict(record) for record in result]
    
    def vector_query(
        self, 
        vector: List[float], 
        limit: int = 10, 
        where: Optional[str] = None,
        offset: int = 0,
        **kwargs: Any,
    ) -> List[str]:
        """Execute a vector query (not supported in Neo4j)."""
        # Neo4j doesn't support vector queries by default
        # This would require additional vector extensions
        return []
