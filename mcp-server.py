import sys
import asyncio
from pathlib import Path
from typing import Any, List, Optional, Dict
import json

# Add HybirdRAG to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    # Allow import even if MCP library is not available (for HTTP API usage)
    Server = None
    stdio_server = None
    Tool = None
    TextContent = None

from HybirdRAG.pipeline import HybridRAGPipeline
from clean import (
    clear_all_databases,
    clear_milvus_collection,
    clear_neo4j_database,
    get_database_status as get_db_status
)

# Global pipeline instance (initialized lazily)
_pipeline: Optional[HybridRAGPipeline] = None


def get_pipeline() -> HybridRAGPipeline:
    """Get or create the HybridRAG pipeline instance."""
    global _pipeline
    if _pipeline is None:
        try:
            _pipeline = HybridRAGPipeline()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize pipeline: {str(e)}") from e
    return _pipeline


# Initialize MCP server (only if MCP library is available)
if Server is not None:
    server = Server("hybirdrag-mcp-server")
else:
    server = None


# Tool Definitions (exportable)
def get_mcp_tools_dict() -> List[Dict[str, Any]]:
    """Get MCP tool definitions as dictionaries (for HTTP API)."""
    return [
        {
            "name": "add_document",
            "description": "Add a single document to the HybirdRAG system. The document will be chunked and indexed in both vector store (Milvus) and knowledge graph (Neo4j) if enabled.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The text content of the document to add"
                    },
                    "id": {
                        "type": "string",
                        "description": "Optional unique identifier for the document"
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional title of the document"
                    },
                    "page": {
                        "type": "integer",
                        "description": "Optional page number"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional additional metadata as key-value pairs"
                    }
                },
                "required": ["content"]
            }
        },
        {
            "name": "add_documents",
            "description": "Add multiple documents to the HybirdRAG system in batch. More efficient than adding documents one by one.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "documents": {
                        "type": "array",
                        "description": "List of documents to add",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "id": {"type": "string"},
                                "title": {"type": "string"},
                                "page": {"type": "integer"},
                                "metadata": {"type": "object"}
                            },
                            "required": ["content"]
                        }
                    }
                },
                "required": ["documents"]
            }
        },
        {
            "name": "query",
            "description": "Query the HybirdRAG system to retrieve relevant documents. Supports query rewriting, broadening, reranking, and compression for optimal results.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "rewrite": {
                        "type": "boolean",
                        "description": "Enable query rewriting for better specificity (default: true)",
                        "default": True
                    },
                    "chat_history": {
                        "type": "array",
                        "description": "Optional chat history for context-aware query rewriting",
                        "items": {"type": "string"}
                    },
                    "broaden_query": {
                        "type": "boolean",
                        "description": "Enable query broadening for better coverage (default: true)",
                        "default": True
                    },
                    "broaden_retry_limit": {
                        "type": "integer",
                        "description": "Maximum retries for query broadening (default: 3)",
                        "default": 3
                    },
                    "context_chunk_size": {
                        "type": "integer",
                        "description": "Context chunk size for processing (default: 256)",
                        "default": 256
                    },
                    "rerank": {
                        "type": "boolean",
                        "description": "Enable reranking for better relevance (default: true)",
                        "default": True
                    },
                    "compress": {
                        "type": "boolean",
                        "description": "Enable result compression (default: false)",
                        "default": False
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return (default: 30)",
                        "default": 30
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "clean_database",
            "description": "Clean/clear all data from both Milvus and Neo4j databases. WARNING: This operation is irreversible!",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "collection_name": {
                        "type": "string",
                        "description": "Name of the collection to clean (default: 'vector_rag')",
                        "default": "vector_rag"
                    },
                    "skip_neo4j_on_auth_failure": {
                        "type": "boolean",
                        "description": "Skip Neo4j cleaning if authentication fails (default: true)",
                        "default": True
                    }
                }
            }
        },
        {
            "name": "clean_milvus",
            "description": "Clean only the Milvus vector database collection. WARNING: This operation is irreversible!",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "collection_name": {
                        "type": "string",
                        "description": "Name of the Milvus collection to clean (default: 'vector_rag')",
                        "default": "vector_rag"
                    }
                }
            }
        },
        {
            "name": "clean_neo4j",
            "description": "Clean only the Neo4j graph database. If collection_name is provided, only deletes nodes with that collection. WARNING: This operation is irreversible!",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "collection_name": {
                        "type": "string",
                        "description": "Optional collection name to filter deletion. If None, deletes ALL data in Neo4j."
                    }
                }
            }
        },
        {
            "name": "get_database_status",
            "description": "Get status information about both Milvus and Neo4j databases, including connection status, collections, node counts, and relationship counts.",
            "inputSchema": {
                "type": "object",
                "properties": {}
            }
        }
    ]


# Tool handler function (exportable for HTTP API)
async def handle_mcp_tool_call(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle MCP tool calls. Returns dict with 'content' and 'isError' keys."""
    try:
        if name == "add_document":
            content = arguments.get("content")
            if not content:
                return {
                    "content": [{"type": "text", "text": json.dumps({"error": "content is required"}, indent=2)}],
                    "isError": True
                }
            
            doc_dict = {"content": content}
            if "id" in arguments and arguments["id"]:
                doc_dict["id"] = arguments["id"]
            if "title" in arguments and arguments["title"]:
                doc_dict["title"] = arguments["title"]
            if "page" in arguments and arguments["page"] is not None:
                doc_dict["page"] = arguments["page"]
            if "metadata" in arguments and arguments["metadata"]:
                doc_dict.update(arguments["metadata"])
            
            pipeline = get_pipeline()
            pipeline.add_document(doc_dict)
            
            return {
                "content": [{"type": "text", "text": json.dumps({
                    "success": True,
                    "message": "Document added successfully",
                    "document_id": doc_dict.get("id", "auto-generated")
                }, indent=2)}],
                "isError": False
            }
        
        elif name == "add_documents":
            documents = arguments.get("documents", [])
            if not documents:
                return {
                    "content": [{"type": "text", "text": json.dumps({"error": "documents array is required"}, indent=2)}],
                    "isError": True
                }
            
            doc_list = []
            for doc in documents:
                doc_dict = {"content": doc.get("content", "")}
                if "id" in doc and doc["id"]:
                    doc_dict["id"] = doc["id"]
                if "title" in doc and doc["title"]:
                    doc_dict["title"] = doc["title"]
                if "page" in doc and doc["page"] is not None:
                    doc_dict["page"] = doc["page"]
                if "metadata" in doc and doc["metadata"]:
                    doc_dict.update(doc["metadata"])
                doc_list.append(doc_dict)
            
            pipeline = get_pipeline()
            pipeline.add_documents(doc_list)
            
            return {
                "content": [{"type": "text", "text": json.dumps({
                    "success": True,
                    "message": f"Successfully added {len(doc_list)} documents",
                    "count": len(doc_list)
                }, indent=2)}],
                "isError": False
            }
        
        elif name == "query":
            query_text = arguments.get("query")
            if not query_text:
                return {
                    "content": [{"type": "text", "text": json.dumps({"error": "query is required"}, indent=2)}],
                    "isError": True
                }
            
            pipeline = get_pipeline()
            results = pipeline.query(
                query=query_text,
                rewrite=arguments.get("rewrite", True),
                chat_history=arguments.get("chat_history"),
                broaden_query=arguments.get("broaden_query", True),
                broaden_retry_limit=arguments.get("broaden_retry_limit", 3),
                context_chunk_size=arguments.get("context_chunk_size", 256),
                rerank=arguments.get("rerank", True),
                compress=arguments.get("compress", False),
                limit=arguments.get("limit", 30)
            )
            
            if not isinstance(results, list):
                results = [results] if results else []
            
            return {
                "content": [{"type": "text", "text": json.dumps({
                    "success": True,
                    "results": results,
                    "count": len(results)
                }, indent=2)}],
                "isError": False
            }
        
        elif name == "clean_database":
            collection_name = arguments.get("collection_name", "vector_rag")
            skip_neo4j_on_auth_failure = arguments.get("skip_neo4j_on_auth_failure", True)
            
            result = clear_all_databases(
                collection_name=collection_name,
                skip_neo4j_on_auth_failure=skip_neo4j_on_auth_failure
            )
            
            return {
                "content": [{"type": "text", "text": json.dumps({
                    "success": result["success"],
                    "milvus": result["milvus"],
                    "neo4j": result["neo4j"]
                }, indent=2)}],
                "isError": False
            }
        
        elif name == "clean_milvus":
            collection_name = arguments.get("collection_name", "vector_rag")
            result = clear_milvus_collection(collection_name=collection_name)
            
            if result["success"]:
                response_data = {
                    "success": True,
                    "message": f"Cleared {result['deleted_count']} entities from Milvus",
                    "deleted_count": result["deleted_count"]
                }
            else:
                response_data = {
                    "success": False,
                    "error": result.get("error", "Unknown error")
                }
            
            return {
                "content": [{"type": "text", "text": json.dumps(response_data, indent=2)}],
                "isError": not result["success"]
            }
        
        elif name == "clean_neo4j":
            collection_name = arguments.get("collection_name")
            result = clear_neo4j_database(collection_name=collection_name)
            
            if result["success"]:
                response_data = {
                    "success": True,
                    "message": f"Cleared {result['deleted_nodes']} nodes and {result['deleted_relationships']} relationships from Neo4j",
                    "deleted_nodes": result["deleted_nodes"],
                    "deleted_relationships": result["deleted_relationships"]
                }
            else:
                response_data = {
                    "success": False,
                    "error": result.get("error", "Unknown error")
                }
            
            return {
                "content": [{"type": "text", "text": json.dumps(response_data, indent=2)}],
                "isError": not result["success"]
            }
        
        elif name == "get_database_status":
            db_status = get_db_status()
            return {
                "content": [{"type": "text", "text": json.dumps({
                    "success": True,
                    "milvus": db_status["milvus"],
                    "neo4j": db_status["neo4j"]
                }, indent=2)}],
                "isError": False
            }
        
        else:
            return {
                "content": [{"type": "text", "text": json.dumps({"error": f"Unknown tool: {name}"}, indent=2)}],
                "isError": True
            }
    
    except Exception as e:
        return {
            "content": [{"type": "text", "text": json.dumps({
                "error": str(e),
                "type": type(e).__name__
            }, indent=2)}],
            "isError": True
        }


# Tool Definitions for MCP Server (using Tool objects)
if server is not None:
    @server.list_tools()
    async def list_tools() -> List[Tool]:
        """List all available tools for MCP server."""
        tools_dict = get_mcp_tools_dict()
        return [
            Tool(
                name=tool["name"],
                description=tool["description"],
                inputSchema=tool["inputSchema"]
            ) for tool in tools_dict
        ]


    # Tool Handlers for MCP Server
    @server.call_tool()
    async def call_tool(name: str, arguments: Any) -> List[TextContent]:
        """Handle tool calls for MCP server."""
        result = await handle_mcp_tool_call(name, arguments)
        return [TextContent(**item) for item in result["content"]]


# Legacy function name for backward compatibility
async def handle_mcp_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy function name - use handle_mcp_tool_call instead."""
    return await handle_mcp_tool_call(name, arguments)


# ============================================================================
# MCP Server Main (only if MCP library is available)
# ============================================================================

async def main():
    """Run the MCP server."""
    if server is None or stdio_server is None:
        print("Error: MCP library not available. Cannot run MCP server.")
        return
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
