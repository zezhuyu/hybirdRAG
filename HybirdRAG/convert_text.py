from datetime import datetime
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


class Formatter:
    def __init__(self):
        self.splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

    def to_llamaindex(self, texts: list):
        """Convert list[str] -> LlamaIndex Documents + Nodes."""
        documents = [Document(text=page) for page in texts]
        nodes = self.splitter.get_nodes_from_documents(documents)
        return documents, nodes

    def to_custom_rag(self, texts: list, with_chapter_section=False):
        """Convert list[str] -> Custom RAG format dicts."""
        now_ts = int(datetime.utcnow().timestamp())
        rag_docs = []

        for page_num, text in enumerate(texts, start=1):
            entry = {"content": text, "add_at": now_ts}
            if with_chapter_section:
                entry["chapter"] = page_num
                entry["section"] = 0
            rag_docs.append(entry)
        return rag_docs
