import spacy
from typing import List, Dict

class ContextualChunker:
    def __init__(self, 
                 mode: str = "sentence", 
                 max_chunk_size: int = 512,   # tokens
                 stride: int = 1, 
                 padding_context: int = 0):
        self.mode = mode
        self.max_chunk_size = max_chunk_size
        self.stride = stride
        self.padding_context = padding_context
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # lightweight
    
    def chunk_text(self, text: str, doc_id: str = None) -> List[Dict]:
        """Split one retrieved doc into contextual chunks"""
        
        # Step A: Sentence segmentation
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        # Step B: Sliding window merge
        chunks = []
        buffer = []
        token_count = 0
        chunk_id = 0
        
        for i, sentence in enumerate(sentences):
            tokens = sentence.split()
            if token_count + len(tokens) > self.max_chunk_size:
                # flush buffer as a chunk
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_{chunk_id}",
                    "text": " ".join(buffer)
                })
                buffer = []
                token_count = 0
                chunk_id += 1
            
            buffer.append(sentence)
            token_count += len(tokens)
        
        if buffer:
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_{chunk_id}",
                "text": " ".join(buffer)
            })
        
        return chunks
    
    def process_results(self, retrieved_texts: List[str]) -> List[Dict]:
        """Apply contextual chunking over retrieval results"""
        all_chunks = []
        for idx, text in enumerate(retrieved_texts):
            doc_id = f"retrieved_{idx}"
            chunks = self.chunk_text(text, doc_id)
            all_chunks.extend(chunks)
        return all_chunks

class LateChunker:
    def __init__(self, max_chunk_tokens: int = 500, stride: int = 0):
        """
        Args:
            max_chunk_tokens: max tokens per late chunk (approx by whitespace split).
            stride: number of sentences to overlap between chunks.
        """
        self.max_chunk_tokens = max_chunk_tokens
        self.stride = stride
        # lightweight pipeline, no NER/parser overhead
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  

    def chunk_document(self, text: str) -> List[str]:
        """Split doc into sentences, then group into late chunks."""
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        chunks = []
        buffer, token_count = [], 0
        start_idx = 0

        for i, sentence in enumerate(sentences):
            tokens = sentence.split()
            if token_count + len(tokens) > self.max_chunk_tokens:
                # flush buffer as a late chunk
                chunks.append(" ".join(buffer))
                
                # apply stride (keep last n sentences in buffer)
                if self.stride > 0:
                    buffer = sentences[max(0, i - self.stride):i]
                    token_count = sum(len(s.split()) for s in buffer)
                else:
                    buffer, token_count = [], 0
            
            buffer.append(sentence)
            token_count += len(tokens)
        
        if buffer:
            chunks.append(" ".join(buffer))
        
        return chunks