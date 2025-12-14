import torch
import os
import gc
import threading
import psutil
from contextlib import contextmanager
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging

from FlagEmbedding import BGEM3FlagModel
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer, pipeline
from llmlingua import PromptCompressor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model loading and memory management"""
    load_rewriter: bool = os.getenv("LOAD_REWRITER", "false").lower() == "true"
    load_reranker: bool = os.getenv("LOAD_RERANKER", "false").lower() == "true"
    load_embedding: bool = os.getenv("LOAD_EMBEDDING", "false").lower() == "true"
    load_llmlingua: bool = os.getenv("LOAD_LLMLINGUA", "false").lower() == "true"
    rewriter_model_name: str = os.getenv("REWRITER_MODEL_NAME", "Qwen/Qwen3-0.6B")
    reranker_model_name: str = os.getenv("RERANKER_MODEL_NAME", "jinaai/jina-reranker-v2-base-multilingual")
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
    llmlingua_model_name: str = os.getenv("LLMLINGUA_MODEL_NAME", "microsoft/llmlingua-2-xlm-roberta-large-meetingbank")
    embedding_instruction: str = os.getenv("EMBEDDING_INSTRUCTION", "Represent the sentence for retrieval: ")
    max_memory_usage: float = float(os.getenv("MAX_MEMORY_GB", "8.0"))  # GB
    enable_model_offloading: bool = os.getenv("ENABLE_OFFLOADING", "false").lower() == "true"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class ModelManager:
    """Thread-safe singleton for managing ML models with memory optimization"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self.config = ModelConfig()
        self._models: Dict[str, Any] = {}
        self._model_locks: Dict[str, threading.Lock] = {
            'embedding': threading.Lock(),
            'reranker': threading.Lock(),
            'prompt_rewriter': threading.Lock(),
            'llm_lingua': threading.Lock()
        }
        self._memory_monitor = MemoryMonitor()
        self._initialized = True
        logger.info("ModelManager initialized with device: %s", self.config.device)
    
    @contextmanager
    def _memory_context(self, model_name: str):
        """Context manager for memory cleanup before and after model operations"""
        # Clear GPU memory before operation
        if self.config.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        self._memory_monitor.log_memory_usage(f"Before {model_name} operation")
        
        try:
            yield
        finally:
            # Clear GPU memory after operation
            if self.config.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            self._memory_monitor.log_memory_usage(f"After {model_name} operation")
    
    def _load_embedding_model(self):
        """Lazy load embedding model"""
        self._models['embedding'] = None
        if self.config.load_embedding and 'embedding' not in self._models:
            logger.info("Loading embedding model: %s", self.config.embedding_model_name)
            self._models['embedding'] = BGEM3FlagModel(
                self.config.embedding_model_name, 
                instruction=self.config.embedding_instruction,
                use_fp16=True,
                device=self.config.device
            )
        return self._models['embedding']
    
    def _load_reranker_model(self):
        """Lazy load reranker model"""
        self._models['reranker'] = None
        self._models['reranker_tokenizer'] = None
        if self.config.load_reranker and 'reranker' not in self._models:
            logger.info("Loading reranker model: %s", self.config.reranker_model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.reranker_model_name,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            ).to(self.config.device).eval()
            
            # Enable gradient checkpointing to save memory
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                
            self._models['reranker'] = model
            self._models['reranker_tokenizer'] = AutoTokenizer.from_pretrained(
                self.config.reranker_model_name
            )
        return self._models['reranker'], self._models['reranker_tokenizer']
    
    def _load_prompt_rewriter(self):
        """Lazy load prompt rewriter model"""
        self._models['prompt_rewriter'] = None
        if self.config.load_rewriter and 'prompt_rewriter' not in self._models:
            logger.info("Loading prompt rewriter: %s", self.config.rewriter_model_name)
            
            tokenizer = AutoTokenizer.from_pretrained(self.config.rewriter_model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.config.rewriter_model_name,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto" if self.config.device == "cuda" else None,
            )
            
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            
            pipeline_obj = pipeline(
            "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
            )
            
            self._models['prompt_rewriter'] = pipeline_obj
        return self._models['prompt_rewriter']
    
    def _load_llm_lingua(self):
        """Lazy load LLMLingua compressor"""
        self._models['llm_lingua'] = None
        if self.config.load_llmlingua and 'llm_lingua' not in self._models:
            logger.info("Loading LLMLingua: %s", self.config.llmlingua_model_name)
            self._models['llm_lingua'] = PromptCompressor(
                model_name=self.config.llmlingua_model_name,
                use_llmlingua2=True
            )
        return self._models['llm_lingua']
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information"""
        return self._memory_monitor.get_memory_info()
    
    def clear_gpu_memory(self):
        """Explicitly clear GPU memory and run garbage collection"""
        if self.config.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all CUDA operations are complete
        gc.collect()
        self._memory_monitor.log_memory_usage("After explicit GPU memory clear")
    
    def cleanup_unused_models(self):
        """Cleanup unused models to free memory"""
        logger.info("Cleaning up unused models")
        for model_name in list(self._models.keys()):
            if model_name in self._model_locks:
                with self._model_locks[model_name]:
                    if model_name in self._models:
                        del self._models[model_name]
        
        # Aggressive memory cleanup
        self.clear_gpu_memory()
        
        # Force Python garbage collection multiple times for thorough cleanup
        for _ in range(3):
            gc.collect()

class MemoryMonitor:
    """Monitor system memory usage"""
    
    def __init__(self):
        self.process = psutil.Process()
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage"""
        memory_info = {
            'ram_usage_gb': self.process.memory_info().rss / 1024**3,
            'ram_percent': self.process.memory_percent(),
        }
        
        if torch.cuda.is_available():
            memory_info.update({
                'gpu_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'gpu_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                'gpu_free_gb': (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1024**3
            })
        
        return memory_info
    
    def log_memory_usage(self, context: str = ""):
        """Log current memory usage"""
        info = self.get_memory_info()
        logger.info("Memory usage %s: RAM=%.2fGB (%.1f%%)", context, info['ram_usage_gb'], info['ram_percent'])
        if 'gpu_allocated_gb' in info:
            logger.info("GPU: Allocated=%.2fGB, Reserved=%.2fGB, Free=%.2fGB", info['gpu_allocated_gb'], info['gpu_reserved_gb'], info['gpu_free_gb'])

# Global model manager instance
model_manager = ModelManager()

def embed_sentence(sentence: str) -> Optional[List[float]]:
    """Embed a sentence using BGE-M3 model with memory management"""
    if not sentence or not sentence.strip():
        logger.warning("Empty sentence provided for embedding")
        return None
    
    with model_manager._model_locks['embedding']:
        try:
            with model_manager._memory_context('embedding'):
                embedding_model = model_manager._load_embedding_model()
                if embedding_model is None:
                    logger.error("Embedding model not loaded")
                    return None
                result = embedding_model.encode(sentence)['dense_vecs']
                return result.tolist() if hasattr(result, 'tolist') else result
        except Exception as e:
            logger.error("Error embedding sentence: %s", str(e))
            # Clear GPU memory on error
            if model_manager.config.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            return None

def rerank_documents(documents: List[str], query: str) -> Optional[List[float]]:
    """Rerank documents using the reranker model with memory management"""
    if not documents or not query:
        logger.warning("Empty documents or query provided for reranking")
        return None
    
    with model_manager._model_locks['reranker']:
        try:
            with model_manager._memory_context('reranker'):
                reranker, tokenizer = model_manager._load_reranker_model()

                if reranker is None or tokenizer is None:
                    logger.error("Reranker or tokenizer not loaded")
                    return None
                # Prepare sentence pairs
                sentence_pairs = [[query, doc] for doc in documents]
                
                # Tokenize with proper device handling
                with torch.no_grad():
                    inputs = tokenizer(
                        sentence_pairs,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=1024
                    ).to(model_manager.config.device)
                    
                    # Get scores
                    outputs = reranker(**inputs)
                    scores = outputs.logits.squeeze()
                    
                    if scores.dim() == 0:  # Single document case
                        scores = scores.unsqueeze(0)
                    
                    return scores.cpu().tolist()
                    
        except Exception as e:
            logger.error("Error reranking documents: %s", str(e))
            # Clear GPU memory on error
            if model_manager.config.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            return None

def rewrite_prompt(prompt: str, max_length: int = 16384, temperature: float = 0.7) -> Optional[str]:
    """Rewrite prompt using the language model with memory management"""
    if not prompt or not prompt.strip():
        logger.warning("Empty prompt provided for rewriting")
        return None
    
    with model_manager._model_locks['prompt_rewriter']:
        try:
            with model_manager._memory_context('prompt_rewriter'):
                pipeline_obj = model_manager._load_prompt_rewriter()

                if pipeline_obj is None:
                    logger.error("Prompt rewriter not loaded")
                    return None
                result = pipeline_obj(
                    prompt,
                    max_length=max_length,
                    do_sample=True,
                    temperature=temperature,
                    num_return_sequences=1,
                    pad_token_id=pipeline_obj.tokenizer.eos_token_id,
                )
                
                return result[0]['generated_text'] if result else None
                
        except Exception as e:
            logger.error("Error rewriting prompt: %s", str(e))
            # Clear GPU memory on error
            if model_manager.config.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            return None

def compress_prompt(prompt: str, documents: List[str], rate: float = 0.33, force_tokens: List[str] = None) -> Optional[Dict[str, Any]]:
    """Compress prompt using LLMLingua with memory management"""
    if not prompt or not prompt.strip():
        logger.warning("Empty prompt provided for compression")
        return None

    if not documents:
        logger.warning("Empty documents provided for compression")
        return None
    
    if force_tokens is None:
        force_tokens = ['\n', '?']
    
    with model_manager._model_locks['llm_lingua']:
        try:
            with model_manager._memory_context('llm_lingua'):
                compressor = model_manager._load_llm_lingua()
                if compressor is None:
                    logger.error("LLMLingua not loaded")
                    return None
                result = compressor.compress_prompt(
                    documents, 
                    question=prompt, 
                    rate=rate, 
                    force_tokens=force_tokens
                )
                return result
                
        except Exception as e:
            logger.error("Error compressing prompt: %s", str(e))
            # Clear GPU memory on error
            if model_manager.config.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            return None

# Utility functions for gRPC server management
def get_model_memory_info() -> Dict[str, float]:
    """Get current model memory usage"""
    return model_manager.get_memory_info()

def cleanup_models():
    """Cleanup all loaded models to free memory"""
    model_manager.cleanup_unused_models()

def clear_gpu_memory():
    """Clear GPU memory and run garbage collection"""
    model_manager.clear_gpu_memory()

def warmup_models(models: List[str] = None):
    """Pre-load specified models for faster inference"""
    if models is None:
        models = ['embedding', 'reranker', 'prompt_rewriter', 'llm_lingua']
    
    logger.info("Warming up models: %s", models)
    
    # Load models in separate threads to avoid blocking
    def load_model(model_name: str):
        try:
            if model_name == 'embedding':
                model_manager._load_embedding_model()
            elif model_name == 'reranker':
                model_manager._load_reranker_model()
            elif model_name == 'prompt_rewriter':
                model_manager._load_prompt_rewriter()
            elif model_name == 'llm_lingua':
                model_manager._load_llm_lingua()
            logger.info("Model %s warmed up successfully", model_name)
        except Exception as e:
            logger.error("Failed to warm up %s: %s", model_name, str(e))
    
    threads = []
    for model_name in models:
        thread = threading.Thread(target=load_model, args=(model_name,))
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
    
    # Clear GPU memory after warmup
    model_manager.clear_gpu_memory()
    model_manager._memory_monitor.log_memory_usage("After warmup")