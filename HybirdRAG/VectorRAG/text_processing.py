import spacy
import re
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


class HardTokenChunker:
    """
    Chunker that splits text by hard token limit.
    Splits at word boundaries when token limit is reached, without overlap.
    """
    
    def __init__(self, max_tokens: int = 512):
        """
        Args:
            max_tokens: Maximum number of tokens per chunk (approximated by whitespace split).
        """
        self.max_tokens = max_tokens
    
    def chunk_document(self, text: str) -> List[str]:
        """
        Split document into chunks by hard token limit.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks, each containing at most max_tokens tokens
        """
        if not text or not text.strip():
            return []
        
        # Split text into words (tokens approximated by whitespace)
        words = text.split()
        
        if len(words) <= self.max_tokens:
            return [text]
        
        chunks = []
        current_chunk = []
        current_token_count = 0
        
        for word in words:
            # Check if adding this word would exceed token limit
            if current_token_count + 1 > self.max_tokens:
                # Flush current chunk
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_token_count = 0
            
            current_chunk.append(word)
            current_token_count += 1
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks


class PunctuationChunker:
    """
    Chunker that splits text by punctuation marks (sentence boundaries).
    Splits at sentence-ending punctuation (., !, ?) and groups sentences into chunks
    based on token count (similar to HardTokenChunker but respects sentence boundaries).
    """
    
    def __init__(self, max_tokens: int = 512, min_tokens: int = 10, max_chunk_length: int = None, min_chunk_length: int = None):
        """
        Args:
            max_tokens: Maximum tokens (words) per chunk (primary limit)
            min_tokens: Minimum tokens (words) per chunk (to avoid very small chunks)
            max_chunk_length: DEPRECATED - Maximum characters per chunk (kept for backward compatibility)
            min_chunk_length: DEPRECATED - Minimum characters per chunk (kept for backward compatibility)
        
        Note: If max_chunk_length is provided, it will be used for backward compatibility,
        but max_tokens is preferred for consistency with HardTokenChunker.
        """
        # Support both token-based and character-based (for backward compatibility)
        if max_chunk_length is not None:
            # Backward compatibility: convert character limit to approximate token limit
            # Rough estimate: 5 characters per word on average
            self.max_tokens = max_tokens if max_tokens != 512 else max_chunk_length // 5
            self.min_tokens = min_tokens if min_tokens != 10 else (min_chunk_length or 50) // 5
        else:
            self.max_tokens = max_tokens
            self.min_tokens = min_tokens
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens by splitting on whitespace (same as HardTokenChunker)."""
        return len(text.split())
    
    def chunk_document(self, text: str) -> List[str]:
        """
        Split document into chunks by punctuation marks (sentence boundaries).
        Groups sentences into chunks based on token count.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks split at sentence boundaries, each containing at most max_tokens tokens
        """
        if not text or not text.strip():
            return []
        
        # Count tokens in full text
        total_tokens = self._count_tokens(text)
        
        # If text has fewer tokens than max_tokens, return as single chunk
        if total_tokens <= self.max_tokens:
            return [text.strip()]
        
        # Use nltk sentence tokenizer to split by sentence-ending punctuation
        sentences = nltk.sent_tokenize(text)
        
        # Group sentences into chunks respecting max_tokens
        chunks = []
        current_chunk = []
        current_token_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Count tokens in this sentence
            sentence_tokens = self._count_tokens(sentence)
            
            # Check if adding this sentence would exceed max_tokens
            if current_token_count + sentence_tokens > self.max_tokens:
                # Flush current chunk if it meets minimum token requirement
                if current_chunk and current_token_count >= self.min_tokens:
                    chunks.append(" ".join(current_chunk).strip())
                    current_chunk = [sentence]
                    current_token_count = sentence_tokens
                else:
                    # If current chunk is too small, append to it anyway
                    # (but only if it won't exceed max_tokens significantly)
                    if current_token_count + sentence_tokens <= self.max_tokens * 1.5:
                        current_chunk.append(sentence)
                        current_token_count += sentence_tokens
                    else:
                        # Force flush even if small to avoid extremely large chunks
                        if current_chunk:
                            chunks.append(" ".join(current_chunk).strip())
                        current_chunk = [sentence]
                        current_token_count = sentence_tokens
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_token_count += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk).strip())
        
        # Filter out chunks that are too small (unless it's the only chunk)
        if len(chunks) > 1:
            chunks = [chunk for chunk in chunks if self._count_tokens(chunk) >= self.min_tokens]
        
        # If no chunks meet minimum token requirement, return at least one chunk
        if not chunks:
            return [text.strip()]
        
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
                 min_tokens_per_chunk: int = 90,
                 sim_drop: float = 0.40,
                 fixed_threshold: float = 0.45,
                 c: float = 0.9,
                 init_constant: float = 1.5,
                 overlap_tokens: int = 40,  # Amount of token overlap between consecutive chunks
                 use_contextual_split: bool = True,  # Enable/disable Stage 1: Contextual segmentation
                 use_semantic_chunking: bool = True,  # Enable/disable Stage 3: Max-Min semantic chunking
                 ):
        """
        Args:
            embedding_client: An embedding client with embed_sentences(sentences: List[str]) -> List[List[float]] method
            max_tokens_per_chunk: Maximum tokens per chunk in late chunking stage
            min_tokens_per_chunk: Minimum tokens per chunk to prevent overly fragmented chunks
            sim_drop: Similarity threshold for contextual split (topic shift detection) - lower = less aggressive
            fixed_threshold: Fixed similarity threshold for max-min semantic chunking - lower = less aggressive
            c: Coefficient for adaptive threshold calculation
            init_constant: Initial constant for similarity calculation when cluster has 1 sentence
            overlap_tokens: Number of tokens to carry over as overlap when splitting late chunks
            use_contextual_split: If True, use Stage 1 (contextual segmentation). If False, skip to late chunking.
            use_semantic_chunking: If True, use Stage 3 (max-min semantic chunking). If False, stop after late chunking.
        """
        if embedding_client is None:
            raise ValueError("embedding_client is required")
        
        self.embedding_client = embedding_client
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.min_tokens_per_chunk = min_tokens_per_chunk
        self.sim_drop = sim_drop
        self.fixed_threshold = fixed_threshold
        self.c = c
        self.init_constant = init_constant
        self.overlap_tokens = max(0, overlap_tokens)
        self.use_contextual_split = use_contextual_split
        self.use_semantic_chunking = use_semantic_chunking

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
        
        # Skip splitting for short texts - they don't need fragmentation
        token_count = len(text.split())
        if len(sentences) <= 3 or token_count < self.max_tokens_per_chunk:
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
        """
        Perform token-level chunking with sentence awareness.
        More aggressive than before - splits at word boundaries when needed to match HardTokenChunker behavior.
        """
        doc = self.spacy_nlp(text)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
        chunks = []
        buffer_words = []  # Store words instead of sentences for more granular control
        token_count = 0

        for sentence in sentences:
            words = sentence.split()
            
            # Add words one by one, splitting when we would exceed max_tokens
            for word in words:
                # Check if adding this word would exceed max_tokens
                if token_count > 0 and token_count >= self.max_tokens_per_chunk:
                    # Flush current chunk before adding new word
                    if buffer_words:
                        chunk_text = " ".join(buffer_words)
                        chunks.append(chunk_text)
                        
                        # Handle overlap if enabled
                        if self.overlap_tokens > 0:
                            # Keep last N words for overlap
                            overlap_words = buffer_words[-self.overlap_tokens:] if len(buffer_words) >= self.overlap_tokens else buffer_words
                            buffer_words = overlap_words
                            token_count = len(buffer_words)
                        else:
                            buffer_words = []
                            token_count = 0
                
                buffer_words.append(word)
                token_count += 1

        # Handle remaining buffer
        if buffer_words:
            chunks.append(" ".join(buffer_words))
        
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
        
        # Filter and merge chunks that are too small, while respecting max_tokens_per_chunk
        filtered_paragraphs = []
        for p in paragraphs:
            chunk_text = " ".join(p)
            token_count = len(chunk_text.split())
            
            # If chunk exceeds max_tokens, split it further (shouldn't happen often with tuned thresholds)
            if token_count > self.max_tokens_per_chunk * 1.2:  # If significantly over, split by sentences
                # Split into multiple chunks at max_tokens boundary
                words = chunk_text.split()
                for i in range(0, len(words), self.max_tokens_per_chunk):
                    split_chunk = " ".join(words[i:i + self.max_tokens_per_chunk])
                    if len(split_chunk.split()) >= self.min_tokens_per_chunk:
                        filtered_paragraphs.append(split_chunk)
                continue
            
            # If chunk is too small, try to merge with previous chunk
            if token_count < self.min_tokens_per_chunk and filtered_paragraphs:
                # Check if merging won't exceed max_tokens too much
                last_chunk_tokens = len(filtered_paragraphs[-1].split())
                if last_chunk_tokens + token_count <= self.max_tokens_per_chunk * 1.2:  # Allow 20% overflow for merging
                    # Merge with previous chunk
                    filtered_paragraphs[-1] = filtered_paragraphs[-1] + " " + chunk_text
                else:
                    # Too large to merge, add as separate chunk anyway
                    filtered_paragraphs.append(chunk_text)
            else:
                # Add as new chunk (meets minimum size or can't be merged)
                filtered_paragraphs.append(chunk_text)
        
        return filtered_paragraphs

    # ---------- Full Pipeline ----------
    def chunk_document(self, text: str):
        """
        Perform chunking using configurable stages:
        - Stage 1 (optional): Contextual segmentation
        - Stage 2 (always): Late chunking (token-based)
        - Stage 3 (optional): Max–Min semantic chunking
        
        Returns a list of final coherent chunks ready for DB storage.
        """
        if not text or not text.strip():
            return []
        
        all_chunks = []
        
        # Stage 1: Contextual segmentation (optional)
        if self.use_contextual_split:
            sections = self.contextual_split(text)
        else:
            sections = [text]  # Skip contextual split, use entire text

        # Stage 2: Late chunking (always performed)
        for section in sections:
            late_chunks = self.late_chunk(section)

            # Stage 3: Max–Min semantic chunking (optional)
            if self.use_semantic_chunking:
                for chunk in late_chunks:
                    # Skip semantic chunking for chunks that are already appropriately sized
                    chunk_tokens = len(chunk.split())
                    # Only apply semantic chunking to chunks that are close to or exceed max_tokens
                    # This avoids over-processing already well-sized chunks
                    if chunk_tokens < self.max_tokens_per_chunk * 0.70:  # If chunk is < 70% of max, skip further splitting
                        all_chunks.append(chunk)
                        continue

                    # Use spacy for consistent sentence tokenization
                    doc = self.spacy_nlp(chunk)
                    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                    
                    if len(sentences) <= 2:  # Skip semantic chunking for very few sentences
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
            else:
                # Skip semantic chunking, use late chunks directly
                all_chunks.extend(late_chunks)

        return all_chunks


