"""
Fixed SimpleVectorStore that properly populates result.nodes
"""
from typing import Any, List, Optional
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryResult
from llama_index.core.storage.docstore import SimpleDocumentStore


class FixedSimpleVectorStore(SimpleVectorStore):
    """Fixed SimpleVectorStore that populates result.nodes from docstore."""
    
    def __init__(self, docstore: Optional[SimpleDocumentStore] = None, **kwargs):
        super().__init__(**kwargs)
        self._docstore = docstore
    
    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Get nodes for response with proper node population."""
        # Call the parent query method to get IDs and similarities
        result = super().query(query, **kwargs)
        
        # If we have IDs and a docstore, populate the nodes
        if result.ids and self._docstore:
            nodes = []
            for node_id in result.ids:
                try:
                    node = self._docstore.get_node(node_id)
                    nodes.append(node)
                except Exception as e:
                    print(f"⚠️  Failed to get node {node_id}: {e}")
                    continue
            
            # Create a new result with nodes populated
            result = VectorStoreQueryResult(
                nodes=nodes,
                similarities=result.similarities,
                ids=result.ids
            )
        
        return result
    
    def set_docstore(self, docstore: SimpleDocumentStore) -> None:
        """Set the docstore for node retrieval."""
        self._docstore = docstore
