#!/usr/bin/env python3
"""
HybirdRAG Unified API Server

Combines REST API and MCP server functionality by importing from:
- rest-server.py: REST API endpoints
- mcp-server.py: MCP tool definitions and handlers
"""

import json
import importlib.util
from pathlib import Path
from typing import Dict, Any, List
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from dotenv import load_dotenv
import os

load_dotenv()


OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PROMPT_REWRITE_MODEL = os.getenv("PROMPT_REWRITE_MODEL")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME")
GRAPH_CREATE_MODEL = os.getenv("GRAPH_CREATE_MODEL")

# Load rest-server.py
rest_server_path = Path(__file__).parent / "rest-server.py"
rest_server_spec = importlib.util.spec_from_file_location("rest_server", rest_server_path)
rest_server = importlib.util.module_from_spec(rest_server_spec)
rest_server_spec.loader.exec_module(rest_server)

# Load mcp-server.py
mcp_server_path = Path(__file__).parent / "mcp-server.py"
mcp_server_spec = importlib.util.spec_from_file_location("mcp_server", mcp_server_path)
mcp_server = importlib.util.module_from_spec(mcp_server_spec)
mcp_server_spec.loader.exec_module(mcp_server)

# Import from loaded modules
app = rest_server.app

# Define MCP models
class MCPToolCall(BaseModel):
    """MCP tool call request."""
    name: str = Field(..., description="Tool name")
    arguments: Dict[str, Any] = Field(..., description="Tool arguments")


class MCPToolResponse(BaseModel):
    """MCP tool call response."""
    content: List[Dict[str, str]] = Field(..., description="Tool response content")
    isError: bool = Field(False, description="Whether the response is an error")

# Import MCP functions
get_mcp_tools_dict = mcp_server.get_mcp_tools_dict
handle_mcp_tool_call = mcp_server.handle_mcp_tool_call

# ============================================================================
# MCP Endpoints (MCP-over-HTTP)
# ============================================================================

@app.get("/mcp/tools", tags=["MCP"])
async def mcp_list_tools():
    """
    List all available MCP tools.
    
    Returns tool definitions in MCP format.
    """
    tools = get_mcp_tools_dict()
    return {"tools": tools}


@app.post("/mcp/tools/call", tags=["MCP"], response_model=MCPToolResponse)
async def mcp_call_tool(tool_call: MCPToolCall):
    """
    Call an MCP tool via HTTP.
    
    This endpoint allows MCP clients to call tools over HTTP instead of stdio.
    """
    result = await handle_mcp_tool_call(
        name=tool_call.name,
        arguments=tool_call.arguments or {}
    )
    return MCPToolResponse(**result)


# ============================================================================
# WebSocket Endpoint for MCP Protocol (Optional)
# ============================================================================

@app.websocket("/mcp/ws")
async def mcp_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for MCP protocol communication.
    
    This allows MCP clients to connect via WebSocket for real-time communication.
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle MCP protocol messages
            if message.get("method") == "tools/list":
                # List tools
                tools_response = await mcp_list_tools()
                await websocket.send_json({
                    "jsonrpc": "2.0",
                    "id": message.get("id"),
                    "result": tools_response
                })
            
            elif message.get("method") == "tools/call":
                # Call tool
                params = message.get("params", {})
                tool_call = MCPToolCall(
                    name=params.get("name", ""),
                    arguments=params.get("arguments", {})
                )
                result = await mcp_call_tool(tool_call)
                await websocket.send_json({
                    "jsonrpc": "2.0",
                    "id": message.get("id"),
                    "result": result.dict()
                })
            
            else:
                await websocket.send_json({
                    "jsonrpc": "2.0",
                    "id": message.get("id"),
                    "error": {"code": -32601, "message": "Method not found"}
                })
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32000, "message": str(e)}
            })
        except Exception:
            pass


# ============================================================================
# Update Root Endpoint
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "HybirdRAG Unified API",
        "version": "1.0.0",
        "protocols": ["REST", "MCP-over-HTTP", "MCP-over-WebSocket"],
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "rest_api": {
                "add_document": "/api/v1/documents",
                "add_documents_batch": "/api/v1/documents/batch",
                "query": "/api/v1/query",
                "clean_database": "/api/v1/database/clean",
                "database_status": "/api/v1/database/status"
            },
            "mcp": {
                "list_tools": "/mcp/tools",
                "call_tool": "/mcp/tools/call",
                "websocket": "/mcp/ws"
            }
        }
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser(description="HybirdRAG Unified Server (REST + MCP)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
