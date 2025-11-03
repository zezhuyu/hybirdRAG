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
        collection_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Neo4j graph store.
        
        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            username: Neo4j username
            password: Neo4j password
            database: Neo4j database name (default: "neo4j")
            collection_name: Optional collection name to tag all nodes with
        """
        super().__init__()
        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        self._database = database
        self._session = self._driver.session(database=database)
        self._collection_name = collection_name  # Store collection name for tagging nodes
        
    def close(self) -> None:
        """Close the Neo4j connection."""
        if self._session:
            self._session.close()
        if self._driver:
            self._driver.close()
    
    def get(self, subj: str = None, ids: List[str] = None) -> List[EntityNode]:
        """Get all objects that are connected to the subject or by IDs."""
        if ids is not None:
            # Handle the case where PropertyGraphIndex calls get(ids=list_of_ids)
            if not ids:
                return []
            # For now, return empty list as we don't have a direct ID lookup
            # This might need to be implemented based on llama-index requirements
            return []
        
        if subj is None:
            # Return all entity nodes when no specific subject is provided
            query = "MATCH (n) WHERE n.name IS NOT NULL RETURN n.name as name, labels(n) as labels, n.description as description, n.properties as properties"
            result = self._session.run(query)
            entities = []
            for record in result:
                name = record["name"]
                labels = record["labels"] if record["labels"] else []
                description = record["description"] or ""
                properties_str = record["properties"] or "{}"
                
                # Parse properties if it's a string
                try:
                    if isinstance(properties_str, str):
                        properties = json.loads(properties_str)
                    else:
                        properties = properties_str or {}
                except:
                    properties = {}
                
                # Add basic properties
                properties["entity_description"] = description
                properties["entity_type"] = labels[0] if labels else "Unknown"
                
                entity = EntityNode(
                    name=name,
                    label=labels[0] if labels else "Unknown",
                    properties=properties
                )
                entities.append(entity)
            return entities
            
        query = "MATCH (s {name: $subj})-[:RELATED_TO]->(o) RETURN o.name as obj_name"
        result = self._session.run(query, subj=subj)
        return [record["obj_name"] for record in result]
    
    def get_rel_map(
        self, graph_nodes: List[Any], depth: int = 2, limit: int = 30, ignore_rels: Optional[List[str]] = None
    ) -> List[Tuple[Any, Any, Any]]:
        """Get relationship map for given graph nodes."""
        if ignore_rels is None:
            ignore_rels = []
        
        # Extract node names from graph_nodes (they might be LabelledNode objects)
        node_names = []
        for node in graph_nodes:
            if hasattr(node, 'name'):
                node_names.append(node.name)
            elif isinstance(node, str):
                node_names.append(node)
            else:
                # Handle other node types
                node_names.append(str(node))
        
        # Get all triplets that involve the specified nodes
        all_triplets = self.get_triplets()
        
        # Filter triplets that involve our nodes and don't match ignored relationships
        relevant_triplets = []
        for entity1, relation, entity2 in all_triplets:
            if (entity1.name in node_names or entity2.name in node_names) and relation.label not in ignore_rels:
                relevant_triplets.append((entity1, relation, entity2))
        
        return relevant_triplets[:limit]
    
    def upsert_nodes(self, nodes: List[BaseNode]) -> None:
        """Upsert nodes into the Neo4j database."""
        entity_count = sum(1 for node in nodes if isinstance(node, EntityNode))
        for node in nodes:
            if isinstance(node, EntityNode):
                self._upsert_entity_node(node)
        # Skip non-EntityNode silently
    
    def _upsert_entity_node(self, node: EntityNode) -> None:
        """Upsert an entity node into Neo4j."""
        # Sanitize the label for Neo4j (replace spaces and special chars with underscores)
        sanitized_label = self._sanitize_label(node.label)
        
        # Create or update the node with the proper label and collection tag
        if self._collection_name:
            query = f"""
            MERGE (n:{sanitized_label} {{name: $name}})
            SET n.description = $description,
                n.properties = $properties,
                n.original_label = $original_label,
                n.collection = $collection
            """
        else:
            query = f"""
            MERGE (n:{sanitized_label} {{name: $name}})
            SET n.description = $description,
                n.properties = $properties,
                n.original_label = $original_label
            """
        
        properties = node.properties or {}
        params = {
            "name": node.name,
            "description": properties.get("entity_description", ""),
            "properties": json.dumps(properties),
            "original_label": node.label  # Store original label as property
        }
        
        if self._collection_name:
            params["collection"] = self._collection_name
        
        self._session.run(query, **params)
    
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
        successful = 0
        for relation in relations:
            if self._upsert_relation(relation):
                successful += 1
    
    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity names for consistent matching."""
        # Remove common prefixes/suffixes and normalize
        normalized = name.strip()
        
        # Handle common patterns
        if "(" in normalized and ")" in normalized:
            # Extract the part in parentheses as it's often the canonical name
            parts = normalized.split("(")
            if len(parts) > 1:
                canonical = parts[1].split(")")[0].strip()
                if len(canonical) > 2:  # Use canonical name if it's substantial
                    normalized = canonical
        
        # Remove common descriptive words and articles (but be more careful)
        descriptive_words = ["works", "domain", "and licensed", "the ", "a ", "an ", "of ", "in ", "on ", "at "]
        for word in descriptive_words:
            normalized = normalized.replace(word, "").strip()
        
        # Handle specific problematic cases
        if "US Internal Revenue Service" in normalized:
            normalized = "IRS"
        elif "former mistress" in normalized:
            normalized = "mistress"
        elif normalized == "a frame":
            normalized = "frame"
        elif normalized == "Parents":
            normalized = "parent"
        
        # Also try just the first few words for partial matching
        words = normalized.split()
        if len(words) > 3:
            # Try first 2-3 words for partial matching
            normalized = " ".join(words[:3])
        
        # Safety check: if normalization resulted in empty string, use original
        if not normalized.strip():
            normalized = name.strip()
        
        return normalized

    def _upsert_relation(self, relation: Relation) -> None:
        """Upsert a relation into Neo4j."""
        # Try exact match first, then normalized match
        check_query = "MATCH (n {name: $name}) RETURN count(n) as count"
        
        source_count = self._session.run(check_query, name=relation.source_id).single()["count"]
        target_count = self._session.run(check_query, name=relation.target_id).single()["count"]
        
        # If exact match fails, try multiple fuzzy matching strategies
        if source_count == 0 or target_count == 0:
            # Try multiple fuzzy matching approaches
            fuzzy_queries = [
                "MATCH (n) WHERE n.name CONTAINS $partial_name RETURN n.name LIMIT 1",
                "MATCH (n) WHERE n.name =~ $regex_name RETURN n.name LIMIT 1", 
                "MATCH (n) WHERE n.name STARTS WITH $start_name RETURN n.name LIMIT 1",
                "MATCH (n) WHERE n.name ENDS WITH $end_name RETURN n.name LIMIT 1",
                "MATCH (n) WHERE n.name =~ $case_insensitive_name RETURN n.name LIMIT 1"
            ]
            
            if source_count == 0:
                source_normalized = self._normalize_entity_name(relation.source_id)
                # Try different fuzzy matching strategies
                for query in fuzzy_queries:
                    if query == "MATCH (n) WHERE n.name =~ $regex_name RETURN n.name LIMIT 1":
                        # Create regex pattern for partial matching
                        regex_pattern = f".*{source_normalized.replace(' ', '.*')}.*"
                        fuzzy_result = self._session.run(query, regex_name=regex_pattern)
                    elif query == "MATCH (n) WHERE n.name STARTS WITH $start_name RETURN n.name LIMIT 1":
                        fuzzy_result = self._session.run(query, start_name=source_normalized.split()[0])
                    elif query == "MATCH (n) WHERE n.name ENDS WITH $end_name RETURN n.name LIMIT 1":
                        fuzzy_result = self._session.run(query, end_name=source_normalized.split()[-1])
                    elif query == "MATCH (n) WHERE n.name =~ $case_insensitive_name RETURN n.name LIMIT 1":
                        # Case insensitive matching
                        case_pattern = f"(?i).*{source_normalized.replace(' ', '.*')}.*"
                        fuzzy_result = self._session.run(query, case_insensitive_name=case_pattern)
                    else:
                        fuzzy_result = self._session.run(query, partial_name=source_normalized)
                    
                    fuzzy_record = fuzzy_result.single()
                    if fuzzy_record:
                        relation.source_id = fuzzy_record["n.name"]
                        source_count = 1
                        break
                    
            if target_count == 0:
                target_normalized = self._normalize_entity_name(relation.target_id)
                # Try different fuzzy matching strategies
                for query in fuzzy_queries:
                    if query == "MATCH (n) WHERE n.name =~ $regex_name RETURN n.name LIMIT 1":
                        # Create regex pattern for partial matching
                        regex_pattern = f".*{target_normalized.replace(' ', '.*')}.*"
                        fuzzy_result = self._session.run(query, regex_name=regex_pattern)
                    elif query == "MATCH (n) WHERE n.name STARTS WITH $start_name RETURN n.name LIMIT 1":
                        fuzzy_result = self._session.run(query, start_name=target_normalized.split()[0])
                    elif query == "MATCH (n) WHERE n.name ENDS WITH $end_name RETURN n.name LIMIT 1":
                        fuzzy_result = self._session.run(query, end_name=target_normalized.split()[-1])
                    elif query == "MATCH (n) WHERE n.name =~ $case_insensitive_name RETURN n.name LIMIT 1":
                        # Case insensitive matching
                        case_pattern = f"(?i).*{target_normalized.replace(' ', '.*')}.*"
                        fuzzy_result = self._session.run(query, case_insensitive_name=case_pattern)
                    else:
                        fuzzy_result = self._session.run(query, partial_name=target_normalized)
                    
                    fuzzy_record = fuzzy_result.single()
                    if fuzzy_record:
                        relation.target_id = fuzzy_record["n.name"]
                        target_count = 1
                        break
        
        if source_count == 0 or target_count == 0:
            # Still no match - let's see what entities actually exist
            if source_count == 0:
                # Show some sample entities to help debug
                sample_entities = self._session.run("MATCH (n) RETURN n.name LIMIT 5")
                sample_names = [record["n.name"] for record in sample_entities]
            if target_count == 0:
                # Show some sample entities to help debug  
                sample_entities = self._session.run("MATCH (n) RETURN n.name LIMIT 5")
                sample_names = [record["n.name"] for record in sample_entities]
            return False
            
        # Create the relationship
        query = """
        MATCH (s {name: $source_id})
        MATCH (t {name: $target_id})
        MERGE (s)-[r:RELATED_TO]->(t)
        SET r.relationship = $label,
            r.description = $description,
            r.properties = $properties
        """
        properties = relation.properties or {}
        result = self._session.run(
            query,
            source_id=relation.source_id,
            target_id=relation.target_id,
            label=relation.label,
            description=properties.get("relationship_description", ""),
            properties=json.dumps(properties)
        )
        return True
    
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
               r.relationship as relation_label, r.description as relation_description, properties(r) as relation_properties,
               o.name as target_name, labels(o) as target_labels, properties(o) as target_properties
        """
        try:
            
            # First, let's check what's actually in the database
            check_entities = self._session.run("MATCH (n) RETURN count(n) as entity_count")
            entity_count = check_entities.single()["entity_count"]
            
            check_rels = self._session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
            rel_count = check_rels.single()["rel_count"]
            
            if rel_count > 0:
                # Show sample relationships
                sample_rels = self._session.run("MATCH (s)-[r]->(t) RETURN type(r), s.name, t.name LIMIT 3")
            
            result = self._session.run(query)
            triplets = []
            record_count = 0
            
            # Check if we get any results at all
            records = list(result)
            
            for record in records:
                record_count += 1
                try:
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
                    
                    # Use the stored relationship label and description
                    relation_label = record["relation_label"] or "RELATED_TO"
                    relation_description = record["relation_description"] or ""
                    
                    # Create properties dict with the description
                    relation_properties = record["relation_properties"] or {}
                    relation_properties["relationship_description"] = relation_description
                    
                    relation = Relation(
                        label=relation_label,
                        source_id=record["source_name"],
                        target_id=record["target_name"],
                        properties=relation_properties
                    )
                    triplets.append((source_node, relation, target_node))
                except Exception as record_error:
                    print(f"⚠️  Error processing record {record_count}: {record_error}")
                    continue
            
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
