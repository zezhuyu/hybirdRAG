#!/usr/bin/env python3
"""
Check if HybirdRAG is using GraphRAG.
"""
import sys
from pathlib import Path

# Add HybirdRAG to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
import os

# Load .env file
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)
    print(f"✅ Loaded .env file from: {env_path}")
else:
    load_dotenv()
    print("⚠️  No .env file found, using environment variables")

# Check configuration
print("\n📊 Configuration Check:")
print("=" * 50)
graphrag_enabled = os.getenv("GRAPHRAG_ENABLED", "true").lower() == "true"
print(f"GRAPHRAG_ENABLED: {os.getenv('GRAPHRAG_ENABLED', 'NOT SET')} → {graphrag_enabled}")
print(f"NEO4J_HOST: {os.getenv('NEO4J_HOST', 'NOT SET')}")
print(f"NEO4J_USERNAME: {os.getenv('NEO4J_USERNAME', 'NOT SET')}")
print(f"NEO4J_PASSWORD: {'SET' if os.getenv('NEO4J_PASSWORD') else 'NOT SET'}")
print(f"NEO4J_DATABASE: {os.getenv('NEO4J_DATABASE', 'NOT SET')}")

# Try to initialize pipeline
print("\n🔍 Pipeline Initialization Check:")
print("=" * 50)
try:
    from HybirdRAG.pipeline import HybridRAGPipeline
    
    pipeline = HybridRAGPipeline()
    
    print(f"✅ Pipeline initialized successfully")
    print(f"   graphrag_enabled: {pipeline.graphrag_enabled}")
    print(f"   graph_rag object: {'✅ Present' if hasattr(pipeline, 'graph_rag') and pipeline.graph_rag else '❌ Missing'}")
    print(f"   graph_store object: {'✅ Present' if hasattr(pipeline, 'graph_store') and pipeline.graph_store else '❌ Missing'}")
    
    if hasattr(pipeline, 'graph_rag') and pipeline.graph_rag:
        print(f"   GraphRAG index: {'✅ Present' if pipeline.graph_rag.index else '❌ Not built'}")
        print(f"   GraphRAG query_engine: {'✅ Present' if pipeline.graph_rag.query_engine else '❌ Not initialized'}")
    
    if pipeline.graphrag_enabled:
        print("\n✅ GraphRAG is ENABLED and will be used in:")
        print("   - Query operations (if graph answer is available)")
        print("   - Document indexing (builds knowledge graph)")
        print("   - Community building")
    else:
        print("\n⚠️  GraphRAG is DISABLED - only vector RAG will be used")
        
except Exception as e:
    print(f"❌ Failed to initialize pipeline: {e}")
    import traceback
    traceback.print_exc()

