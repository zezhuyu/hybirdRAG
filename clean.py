#!/usr/bin/env python3
"""
Database utility functions for ReqQuest hybrid RAG system.
These functions can be imported and used in your own Python scripts.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

def load_env_file(env_path: Path) -> None:
    """Populate os.environ from a .env file."""
    if not env_path.exists():
        return
    
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ[key.strip()] = value.strip()

def _require_env(key: str) -> str:
    """Get required environment variable or raise error."""
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Environment variable '{key}' must be set.")
    return value

def _parse_host_port(value: str, default_port: Optional[str] = None) -> Dict[str, str]:
    """Parse host:port string into dict."""
    if ":" in value:
        host, port = value.split(":", 1)
        return {"host": host, "port": port}
    if default_port:
        return {"host": value, "port": default_port}
    return {"host": value}

def clear_milvus_collection(collection_name: str = "vector_rag") -> Dict[str, Any]:
    """
    Clear all data from a Milvus collection.
    
    Args:
        collection_name: Name of the collection to clear
        
    Returns:
        dict: Result information with success status and details
    """
    result = {
        "success": False,
        "collection_name": collection_name,
        "deleted_count": 0,
        "error": None
    }
    
    try:
        from pymilvus import MilvusClient
        
        # Get Milvus configuration
        milvus_host = _require_env("MILVUS_HOST")
        milvus_parts = _parse_host_port(milvus_host)
        
        if "port" in milvus_parts:
            milvus_uri = f"http://{milvus_parts['host']}:{milvus_parts['port']}"
        else:
            milvus_uri = f"http://{milvus_parts['host']}:19530"
        
        # Create Milvus client
        milvus_kwargs = {"uri": milvus_uri}
        milvus_user = os.getenv("MILVUS_USERNAME")
        milvus_password = os.getenv("MILVUS_PASSWORD")
        
        if milvus_user:
            milvus_kwargs["user"] = milvus_user
        if milvus_password:
            milvus_kwargs["password"] = milvus_password
            
        milvus = MilvusClient(**milvus_kwargs)
        
        # Check if collection exists
        if not milvus.has_collection(collection_name=collection_name):
            result["error"] = f"Collection '{collection_name}' does not exist"
            return result
        
        # Clear the collection by deleting all entities
        # Milvus requires a non-empty filter, so we use "id >= 0" to match all entities
        delete_result = milvus.delete(
            collection_name=collection_name,
            filter="id >= 0",  # This will match all entities since IDs are positive
            output_fields=["id"]  # Only need ID field for deletion
        )
        
        result["success"] = True
        result["deleted_count"] = delete_result.get('delete_count', 0)
        
        return result
        
    except Exception as e:
        result["error"] = str(e)
        return result

def clear_neo4j_database(collection_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Clear data from the Neo4j database.
    
    Args:
        collection_name: Optional collection name to filter deletion. 
                        If provided, only deletes nodes with matching collection property.
                        If None, deletes ALL data in the database.
    
    Returns:
        dict: Result information with success status and details
    """
    result = {
        "success": False,
        "deleted_nodes": 0,
        "deleted_relationships": 0,
        "error": None,
        "collection": collection_name
    }
    
    try:
        # Get Neo4j configuration
        neo4j_host = _require_env("NEO4J_HOST")
        neo4j_username = _require_env("NEO4J_USERNAME")
        neo4j_password = _require_env("NEO4J_PASSWORD")
        neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
        
        # Parse Neo4j host
        neo4j_parts = _parse_host_port(neo4j_host, "7687")
        neo4j_uri = f"bolt://{neo4j_parts['host']}:{neo4j_parts['port']}"
        
        # Try to import Neo4j driver
        try:
            from neo4j import GraphDatabase
        except ImportError:
            result["error"] = "Neo4j driver not available. Install with: pip install neo4j"
            return result
        
        # Connect to Neo4j
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
        
        with driver.session(database=neo4j_database) as session:
            # Build query based on whether collection filter is provided
            if collection_name:
                # Delete only nodes with matching collection property
                count_query = "MATCH (n) WHERE n.collection = $collection RETURN count(n) as node_count"
                rel_count_query = """
                    MATCH (n)-[r]->(m) 
                    WHERE n.collection = $collection OR m.collection = $collection 
                    RETURN count(r) as rel_count
                """
                delete_query = "MATCH (n) WHERE n.collection = $collection DETACH DELETE n"
                verify_query = "MATCH (n) WHERE n.collection = $collection RETURN count(n) as remaining_nodes"
                
                # Get count of nodes and relationships before clearing
                result_query = session.run(count_query, collection=collection_name)
                node_count = result_query.single()["node_count"]
                
                result_query = session.run(rel_count_query, collection=collection_name)
                rel_count = result_query.single()["rel_count"]
                
                # Clear data for this collection
                session.run(delete_query, collection=collection_name)
                
                # Verify deletion
                result_query = session.run(verify_query, collection=collection_name)
                remaining = result_query.single()["remaining_nodes"]
            else:
                # Delete ALL data in the database
                count_query = "MATCH (n) RETURN count(n) as node_count"
                rel_count_query = "MATCH ()-[r]->() RETURN count(r) as rel_count"
                delete_query = "MATCH (n) DETACH DELETE n"
                verify_query = "MATCH (n) RETURN count(n) as remaining_nodes"
                
                # Get count of nodes and relationships before clearing
                result_query = session.run(count_query)
                node_count = result_query.single()["node_count"]
                
                result_query = session.run(rel_count_query)
                rel_count = result_query.single()["rel_count"]
                
                # Clear all data
                session.run(delete_query)
                
                # Verify deletion
                result_query = session.run(verify_query)
                remaining = result_query.single()["remaining_nodes"]
            
            if remaining == 0:
                result["success"] = True
                result["deleted_nodes"] = node_count
                result["deleted_relationships"] = rel_count
            else:
                result["error"] = f"{remaining} nodes still remain after deletion"
        
        driver.close()
        return result
        
    except Exception as e:
        result["error"] = str(e)
        return result

def clear_all_databases(collection_name: str = "vector_rag", skip_neo4j_on_auth_failure: bool = True) -> Dict[str, Any]:
    """
    Clear all data from both Milvus and Neo4j databases.
    
    Args:
        collection_name: Name of the Milvus collection to clear
        skip_neo4j_on_auth_failure: If True, skip Neo4j clearing if authentication fails
        
    Returns:
        dict: Combined result information from both operations
    """
    # Load environment
    env_path = Path("HybirdRAG/.env")
    if env_path.exists():
        load_env_file(env_path)
    
    # Clear Milvus (always try this)
    milvus_result = clear_milvus_collection(collection_name)
    
    # Clear Neo4j (with optional skip on auth failure)
    # Pass collection_name to only clear nodes belonging to this collection
    neo4j_result = clear_neo4j_database(collection_name=collection_name)
    
    # If Neo4j failed due to authentication and we should skip it, consider it successful
    if (skip_neo4j_on_auth_failure and 
        not neo4j_result["success"] and 
        neo4j_result["error"] and 
        "authentication" in neo4j_result["error"].lower()):
        
        neo4j_result["skipped"] = True
        neo4j_result["reason"] = "Authentication failure - skipped as requested"
    
    return {
        "success": milvus_result["success"] and (neo4j_result["success"] or neo4j_result.get("skipped", False)),
        "milvus": milvus_result,
        "neo4j": neo4j_result
    }

def get_database_status() -> Dict[str, Any]:
    """
    Get status information about both databases.
    
    Returns:
        dict: Status information for both Milvus and Neo4j
    """
    # Load environment
    env_path = Path("HybirdRAG/.env")
    if env_path.exists():
        load_env_file(env_path)
    
    status = {
        "milvus": {"connected": False, "collections": [], "error": None},
        "neo4j": {"connected": False, "nodes": 0, "relationships": 0, "error": None}
    }
    
    # Check Milvus status
    try:
        from pymilvus import MilvusClient
        
        milvus_host = _require_env("MILVUS_HOST")
        milvus_parts = _parse_host_port(milvus_host)
        
        if "port" in milvus_parts:
            milvus_uri = f"http://{milvus_parts['host']}:{milvus_parts['port']}"
        else:
            milvus_uri = f"http://{milvus_parts['host']}:19530"
        
        milvus_kwargs = {"uri": milvus_uri}
        milvus_user = os.getenv("MILVUS_USERNAME")
        milvus_password = os.getenv("MILVUS_PASSWORD")
        
        if milvus_user:
            milvus_kwargs["user"] = milvus_user
        if milvus_password:
            milvus_kwargs["password"] = milvus_password
            
        milvus = MilvusClient(**milvus_kwargs)
        
        # Get all collections
        collections = milvus.list_collections()
        status["milvus"]["connected"] = True
        status["milvus"]["collections"] = collections
        
    except Exception as e:
        status["milvus"]["error"] = str(e)
    
    # Check Neo4j status
    try:
        from neo4j import GraphDatabase
        
        neo4j_host = _require_env("NEO4J_HOST")
        neo4j_username = _require_env("NEO4J_USERNAME")
        neo4j_password = _require_env("NEO4J_PASSWORD")
        neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
        
        neo4j_parts = _parse_host_port(neo4j_host, "7687")
        neo4j_uri = f"bolt://{neo4j_parts['host']}:{neo4j_parts['port']}"
        
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
        
        with driver.session(database=neo4j_database) as session:
            result = session.run("MATCH (n) RETURN count(n) as node_count")
            node_count = result.single()["node_count"]
            
            result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
            rel_count = result.single()["rel_count"]
            
            status["neo4j"]["connected"] = True
            status["neo4j"]["nodes"] = node_count
            status["neo4j"]["relationships"] = rel_count
        
        driver.close()
        
    except Exception as e:
        status["neo4j"]["error"] = str(e)
    
    return status

# Example usage functions
def example_usage():
    """Example of how to use the database utility functions."""
    


    collection_name = "quickta"
    # Load environment first
    env_path = Path("HybirdRAG/.env")
    if env_path.exists():
        load_env_file(env_path)
    
    print("🔍 Checking database status...")
    status = get_database_status()
    print(f"Milvus connected: {status['milvus']['connected']}")
    print(f"Neo4j connected: {status['neo4j']['connected']}")
    
    if status['milvus']['connected']:
        print(f"Milvus collections: {status['milvus']['collections']}")
    
    if status['neo4j']['connected']:
        print(f"Neo4j nodes: {status['neo4j']['nodes']}")
        print(f"Neo4j relationships: {status['neo4j']['relationships']}")
    
    # Example: Clear only Milvus
    print("\n🗑️  Clearing Milvus collection...")
    milvus_result = clear_milvus_collection(collection_name)
    if milvus_result["success"]:
        print(f"✅ Cleared {milvus_result['deleted_count']} entities from Milvus")
    else:
        print(f"❌ Failed to clear Milvus: {milvus_result['error']}")
    
    # Example: Clear all databases
    print("\n🗑️  Clearing all databases...")
    all_result = clear_all_databases(collection_name)
    if all_result["success"]:
        print("✅ All databases cleared successfully!")
    else:
        print("❌ Some operations failed")
        print(f"Milvus: {all_result['milvus']}")
        print(f"Neo4j: {all_result['neo4j']}")

if __name__ == "__main__":
    example_usage()
