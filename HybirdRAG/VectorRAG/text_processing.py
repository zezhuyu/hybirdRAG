import spacy
from typing import List, Dict, Optional
from datetime import datetime
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import nltk
nltk.download("punkt", quiet=True)


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
        # Add sentencizer to handle sentence boundaries
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")
    
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
        # Add sentencizer to handle sentence boundaries
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")  

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

class HybridChunker:
    """
    Hybrid chunker that:
    1. Uses contextual retrieval to segment large paragraphs into smaller topical sections
    2. Applies late chunking (token-level pooling)
    3. Applies Max–Min semantic chunking (sentence-level adaptive grouping)
    
    Uses an embedding client instead of loading a separate embedding model.
    """

    def __init__(self,
                 embedding_client,  # OpenAIEmbeddingClient or similar with embed_sentences() method
                 max_tokens_per_chunk: int = 512,
                 sim_drop: float = 0.55,
                 fixed_threshold: float = 0.6,
                 c: float = 0.9,
                 init_constant: float = 1.5):
        """
        Args:
            embedding_client: An embedding client with embed_sentences(sentences: List[str]) -> List[List[float]] method
            max_tokens_per_chunk: Maximum tokens per chunk in late chunking stage
            sim_drop: Similarity threshold for contextual split (topic shift detection)
            fixed_threshold: Fixed similarity threshold for max-min semantic chunking
            c: Coefficient for adaptive threshold calculation
            init_constant: Initial constant for similarity calculation when cluster has 1 sentence
        """
        if embedding_client is None:
            raise ValueError("embedding_client is required")
        
        self.embedding_client = embedding_client
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.sim_drop = sim_drop
        self.fixed_threshold = fixed_threshold
        self.c = c
        self.init_constant = init_constant

        # Load spacy for sentence tokenization
        self.spacy_nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        # Add sentencizer if not present
        if "sentencizer" not in self.spacy_nlp.pipe_names:
            self.spacy_nlp.add_pipe("sentencizer")

    # ---------- Step 1: Contextual Retrieval Segmentation ----------
    def contextual_split(self, text: str):
        """Split long paragraphs into smaller coherent sections based on embedding similarity."""
        if not text or not text.strip():
            return []
        
        # Use spacy for consistent sentence tokenization
        doc = self.spacy_nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        if len(sentences) <= 2:
            return [text]

        # Get embeddings using embedding client
        try:
            embeddings = self.embedding_client.embed_sentences(sentences)
            if embeddings is None or len(embeddings) != len(sentences):
                # Fallback to individual embeddings if batch fails
                embeddings = [self.embedding_client.embed_sentence(s) for s in sentences]
            
            # Convert to numpy array
            sent_embs = np.array(embeddings)
        except Exception as e:
            # If embedding fails, return text as single section
            print(f"⚠️  Embedding failed in contextual_split: {e}")
            return [text]

        splits, current = [], [sentences[0]]
        for i in range(1, len(sentences)):
            # Calculate cosine similarity between consecutive sentences
            sim = cosine_similarity(sent_embs[i-1:i], sent_embs[i:i+1])[0][0]
            if sim < self.sim_drop:  # detect topic shift
                splits.append(" ".join(current))
                current = []
            current.append(sentences[i])
        if current:
            splits.append(" ".join(current))
        return splits

    # ---------- Step 2: Late Chunking ----------
    def late_chunk(self, text: str):
        """Perform token-level contextual chunking using embeddings window pooling."""
        doc = self.spacy_nlp(text)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
        chunks = []
        buffer, token_count = [], 0

        for sentence in sentences:
            tokens = sentence.split()
            if token_count + len(tokens) > self.max_tokens_per_chunk:
                chunks.append(" ".join(buffer))
                buffer, token_count = [], 0
            buffer.append(sentence)
            token_count += len(tokens)

        if buffer:
            chunks.append(" ".join(buffer))
        return chunks

    # ---------- Step 3: Max–Min Semantic Chunking ----------
    def process_sentences(self, sentences, embeddings):
        """Implements Max–Min Semantic Chunking with adaptive threshold."""
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        if len(sentences) <= 1:
            return [" ".join(sentences)] if sentences else []

        paragraphs = []
        current_paragraph = [sentences[0]]
        cluster_start, cluster_end = 0, 1
        pairwise_min = None  # Initialize as None, not -inf

        for i in range(1, len(sentences)):
            cluster_embeddings = embeddings[cluster_start:cluster_end]

            if cluster_end - cluster_start > 1:
                # Cluster has multiple sentences - use adaptive threshold
                new_sim = cosine_similarity(embeddings[i].reshape(1, -1), cluster_embeddings)[0]
                new_sentence_sim = float(np.max(new_sim))  # Extract scalar
                
                # Update pairwise_min if it was initialized
                if pairwise_min is not None:
                    pairwise_min = min(float(np.min(new_sim)), pairwise_min)
                else:
                    pairwise_min = float(np.min(new_sim))
                
                # Calculate adjusted threshold
                adjusted_thresh = pairwise_min * self.c * sigmoid((cluster_end - cluster_start) - 1)
            else:
                # Cluster has only 1 sentence - compute initial similarity
                adjusted_thresh = 0
                sim_result = cosine_similarity(embeddings[i].reshape(1, -1), cluster_embeddings)
                # Extract scalar from array result
                pairwise_min = float(sim_result[0][0])
                new_sentence_sim = self.init_constant * pairwise_min

            # Decide whether to add to current paragraph or start new one
            if new_sentence_sim > max(adjusted_thresh, self.fixed_threshold):
                current_paragraph.append(sentences[i])
                cluster_end += 1
            else:
                paragraphs.append(current_paragraph)
                current_paragraph = [sentences[i]]
                cluster_start, cluster_end = i, i + 1
                pairwise_min = None  # Reset for new cluster

        paragraphs.append(current_paragraph)
        return [" ".join(p) for p in paragraphs]

    # ---------- Full Pipeline ----------
    def chunk_document(self, text: str):
        """
        Perform contextual segmentation → late chunking → max–min semantic chunking.
        Returns a list of final coherent chunks ready for DB storage.
        """
        if not text or not text.strip():
            return []
        
        all_chunks = []
        
        # Step 1: Contextual segmentation
        small_sections = self.contextual_split(text)

        for section in small_sections:
            # Step 2: Late chunking
            late_chunks = self.late_chunk(section)

            for chunk in late_chunks:
                # Step 3: Max–Min semantic chunking
                # Use spacy for consistent sentence tokenization
                doc = self.spacy_nlp(chunk)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                
                if len(sentences) <= 1:
                    all_chunks.append(chunk)
                    continue

                # Get embeddings using embedding client
                try:
                    embeddings = self.embedding_client.embed_sentences(sentences)
                    if embeddings is None or len(embeddings) != len(sentences):
                        # Fallback to individual embeddings if batch fails
                        embeddings = [self.embedding_client.embed_sentence(s) for s in sentences]
                    
                    # Convert to numpy array
                    sent_embs = np.array(embeddings)
                except Exception as e:
                    # If embedding fails, use the chunk as-is
                    print(f"⚠️  Embedding failed in chunk_document: {e}")
                    all_chunks.append(chunk)
                    continue

                refined_chunks = self.process_sentences(sentences, sent_embs)
                all_chunks.extend(refined_chunks)

        return all_chunks