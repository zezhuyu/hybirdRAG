#!/usr/bin/env python3
"""
Production gRPC Server for ML Models

This server implements the ml_models.proto service definition with comprehensive
memory management and error handling.

Usage:
    # First, generate protobuf files:
    pip install grpcio-tools
    python -m grpc_tools.protoc --python_out=. --grpc_python_out=. ml_models.proto
    
    # Then start server:
    python ml_grpc_server.py --port 50051 --max_workers 10
"""

import grpc
import time
import signal
import argparse
import threading
from concurrent import futures
from typing import Dict, Any, List as TypingList
import logging

# Import our memory-optimized model functions
from run import (
    embed_sentence,
    rerank_documents, 
    rewrite_prompt,
    compress_prompt,
    get_model_memory_info,
    cleanup_models,
    warmup_models,
    logger,
    model_manager
)

# Import generated protobuf classes (generate with protoc first)
try:
    import ml_models_pb2 as pb2
    import ml_models_pb2_grpc as pb2_grpc
    PROTO_AVAILABLE = True
except ImportError:
    logger.warning("Protobuf files not found. Run: python -m grpc_tools.protoc --python_out=. --grpc_python_out=. ml_models.proto")
    PROTO_AVAILABLE = False
    # Create dummy classes for development
    class pb2:
        pass
    class pb2_grpc:
        class MLModelServiceServicer:
            pass

class MLModelServicer(pb2_grpc.MLModelServiceServicer):
    """gRPC servicer implementing ml_models.proto"""
    
    def __init__(self, max_batch_size: int = 32):
        self.max_batch_size = max_batch_size
        self.request_count = 0
        self.total_processing_time = 0.0
        self.model_stats = {
            'embedding_requests': 0,
            'reranking_requests': 0,
            'prompt_rewrite_requests': 0,
            'prompt_compression_requests': 0,
            'error_count': 0
        }
        self._stats_lock = threading.Lock()
        self.server_start_time = time.time()
        
        logger.info("MLModelServicer initialized with max_batch_size=%d", max_batch_size)
    
    def EmbedSentence(self, request, context):
        """Embed a single sentence"""
        start_time = time.time()
        
        try:
            if not request.sentence:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details('Sentence cannot be empty')
                return pb2.EmbedSentenceResponse(
                    success=False, 
                    error_message='Sentence cannot be empty'
                )
            
            # Get embedding
            embedding = embed_sentence(request.sentence)
            processing_time = time.time() - start_time
            
            if embedding is not None:
                self._update_stats('embedding_requests', processing_time)
                return pb2.EmbedSentenceResponse(
                    embedding=embedding,
                    success=True
                )
            else:
                self._update_stats('error_count', processing_time)
                return pb2.EmbedSentenceResponse(
                    success=False,
                    error_message='Failed to generate embedding'
                )
                
        except Exception as e:
            logger.error("Error in EmbedSentence: %s", str(e))
            self._update_stats('error_count', time.time() - start_time)
            return pb2.EmbedSentenceResponse(
                success=False,
                error_message=f'Internal error: {str(e)}'
            )
    
    def EmbedBatch(self, request, context):
        """Embed multiple sentences"""
        start_time = time.time()
        
        try:
            if not request.sentences:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details('No sentences provided')
                return pb2.EmbedBatchResponse(
                    success=False,
                    error_message='No sentences provided'
                )
            
            # Limit batch size
            sentences = list(request.sentences)[:self.max_batch_size]
            if len(request.sentences) > self.max_batch_size:
                logger.warning("Batch size limited from %d to %d", 
                             len(request.sentences), self.max_batch_size)
            
            # Process sentences
            embeddings = []
            processed_count = 0
            failed_count = 0
            
            for sentence in sentences:
                embedding = embed_sentence(sentence)
                if embedding is not None:
                    embeddings.append(pb2.FloatArray(values=embedding))
                    processed_count += 1
                else:
                    embeddings.append(pb2.FloatArray(values=[]))  # Empty for failed
                    failed_count += 1
            
            processing_time = time.time() - start_time
            self._update_stats('embedding_requests', processing_time, count=len(sentences))
            
            return pb2.EmbedBatchResponse(
                embeddings=embeddings,
                success=True,
                processed_count=processed_count,
                failed_count=failed_count
            )
            
        except Exception as e:
            logger.error("Error in EmbedBatch: %s", str(e))
            self._update_stats('error_count', time.time() - start_time)
            return pb2.EmbedBatchResponse(
                success=False,
                error_message=f'Internal error: {str(e)}'
            )
    
    def RerankDocuments(self, request, context):
        """Rerank documents against a query"""
        start_time = time.time()
        
        try:
            if not request.query:
                return pb2.RerankDocumentsResponse(
                    success=False,
                    error_message='Query cannot be empty'
                )
            
            if not request.documents:
                return pb2.RerankDocumentsResponse(
                    success=False,
                    error_message='No documents provided'
                )
            
            # Apply batch size limit
            max_batch = request.max_batch_size if request.max_batch_size > 0 else self.max_batch_size
            documents = list(request.documents)[:max_batch]
            
            # Get reranking scores
            scores = rerank_documents(documents, request.query)
            processing_time = time.time() - start_time
            
            if scores is not None:
                self._update_stats('reranking_requests', processing_time)
                return pb2.RerankDocumentsResponse(
                    scores=scores,
                    success=True,
                    processed_count=len(documents)
                )
            else:
                self._update_stats('error_count', processing_time)
                return pb2.RerankDocumentsResponse(
                    success=False,
                    error_message='Failed to rerank documents'
                )
                
        except Exception as e:
            logger.error("Error in RerankDocuments: %s", str(e))
            self._update_stats('error_count', time.time() - start_time)
            return pb2.RerankDocumentsResponse(
                success=False,
                error_message=f'Internal error: {str(e)}'
            )
    
    def RewritePrompt(self, request, context):
        """Rewrite a prompt using language model"""
        start_time = time.time()
        
        try:
            if not request.prompt:
                return pb2.RewritePromptResponse(
                    success=False,
                    error_message='Prompt cannot be empty'
                )
            
            # Use provided parameters or defaults
            max_length = request.max_length if request.max_length > 0 else 16384
            temperature = request.temperature if request.temperature > 0 else 0.7
            
            # Rewrite prompt
            rewritten = rewrite_prompt(request.prompt, max_length, temperature)
            processing_time = time.time() - start_time
            
            if rewritten is not None:
                self._update_stats('prompt_rewrite_requests', processing_time)
                return pb2.RewritePromptResponse(
                    rewritten_prompt=rewritten,
                    success=True,
                    input_length=len(request.prompt),
                    output_length=len(rewritten)
                )
            else:
                self._update_stats('error_count', processing_time)
                return pb2.RewritePromptResponse(
                    success=False,
                    error_message='Failed to rewrite prompt'
                )
                
        except Exception as e:
            logger.error("Error in RewritePrompt: %s", str(e))
            self._update_stats('error_count', time.time() - start_time)
            return pb2.RewritePromptResponse(
                success=False,
                error_message=f'Internal error: {str(e)}'
            )
    
    def CompressPrompt(self, request, context):
        """Compress prompt using LLMLingua"""
        start_time = time.time()
        
        try:
            if not request.prompt:
                return pb2.CompressPromptResponse(
                    success=False,
                    error_message='Prompt cannot be empty'
                )
            
            if not request.documents:
                return pb2.CompressPromptResponse(
                    success=False,
                    error_message='No documents provided for compression'
                )
            # Use provided parameters or defaults
            compression_rate = request.compression_rate if request.compression_rate > 0 else 0.33
            force_tokens = list(request.force_tokens) if request.force_tokens else None
            
            # Compress prompt
            result = compress_prompt(
                request.prompt,
                list(request.documents),
                rate=compression_rate, 
                force_tokens=force_tokens
            )
            processing_time = time.time() - start_time
            
            if result is not None:
                self._update_stats('prompt_compression_requests', processing_time)
                
                # Extract compression results (adjust based on actual LLMLingua output)
                compressed_text = result.get('compressed_prompt', '')
                ratio = result.get('compression_ratio', 0.0)
                original_tokens = result.get('origin_tokens', 0)
                compressed_tokens = result.get('compressed_tokens', 0)
                
                # Convert any additional metadata to string format
                metadata = {}
                for k, v in result.items():
                    if k not in ['compressed_prompt', 'compression_ratio', 'origin_tokens', 'compressed_tokens']:
                        metadata[k] = str(v)
                
                return pb2.CompressPromptResponse(
                    compressed_prompt=compressed_text,
                    compression_ratio=ratio,
                    original_tokens=original_tokens,
                    compressed_tokens=compressed_tokens,
                    success=True,
                    metadata=metadata
                )
            else:
                self._update_stats('error_count', processing_time)
                return pb2.CompressPromptResponse(
                    success=False,
                    error_message='Failed to compress prompt'
                )
                
        except Exception as e:
            logger.error("Error in CompressPrompt: %s", str(e))
            self._update_stats('error_count', time.time() - start_time)
            return pb2.CompressPromptResponse(
                success=False,
                error_message=f'Internal error: {str(e)}'
            )
    
    def GetMemoryInfo(self, request, context):
        """Get current memory usage information"""
        try:
            memory_info = get_model_memory_info()
            
            # Build response
            memory_response = pb2.MemoryInfo(
                ram_usage_gb=memory_info.get('ram_usage_gb', 0.0),
                ram_percent=memory_info.get('ram_percent', 0.0),
                gpu_allocated_gb=memory_info.get('gpu_allocated_gb', 0.0),
                gpu_reserved_gb=memory_info.get('gpu_reserved_gb', 0.0),
                gpu_free_gb=memory_info.get('gpu_free_gb', 0.0),
                timestamp=int(time.time())
            )
            
            return pb2.GetMemoryInfoResponse(
                memory_info=memory_response,
                success=True
            )
            
        except Exception as e:
            logger.error("Error in GetMemoryInfo: %s", str(e))
            return pb2.GetMemoryInfoResponse(
                success=False,
                error_message=f'Internal error: {str(e)}'
            )
    
    def CleanupModels(self, request, context):
        """Cleanup loaded models to free memory"""
        try:
            # Get memory before cleanup
            before_memory = get_model_memory_info().get('ram_usage_gb', 0.0)
            
            # Perform cleanup
            cleanup_models()
            
            # Get memory after cleanup
            after_memory = get_model_memory_info().get('ram_usage_gb', 0.0)
            memory_freed = max(0.0, before_memory - after_memory)
            
            return pb2.CleanupModelsResponse(
                success=True,
                cleaned_models=['all'],  # Could be more specific
                memory_freed_gb=memory_freed
            )
            
        except Exception as e:
            logger.error("Error in CleanupModels: %s", str(e))
            return pb2.CleanupModelsResponse(
                success=False,
                error_message=f'Internal error: {str(e)}'
            )
    
    def WarmupModels(self, request, context):
        """Warm up specified models"""
        try:
            models_to_warmup = list(request.models) if request.models else None
            start_time = time.time()
            
            # Warm up models
            warmup_models(models_to_warmup)
            
            total_time = int(time.time() - start_time)
            
            # Build results (simplified - could track per-model timing)
            results = []
            model_names = models_to_warmup or ['embedding', 'reranker', 'prompt_rewriter', 'llm_lingua']
            
            for model_name in model_names:
                results.append(pb2.ModelWarmupResult(
                    model_name=model_name,
                    success=True,
                    load_time_seconds=total_time // len(model_names)
                ))
            
            return pb2.WarmupModelsResponse(
                success=True,
                results=results,
                total_time_seconds=total_time
            )
            
        except Exception as e:
            logger.error("Error in WarmupModels: %s", str(e))
            return pb2.WarmupModelsResponse(
                success=False,
                error_message=f'Internal error: {str(e)}'
            )
    
    def GetServerStatus(self, request, context):
        """Get comprehensive server status"""
        try:
            with self._stats_lock:
                uptime = int(time.time() - self.server_start_time)
                avg_response = (
                    self.total_processing_time / max(self.request_count, 1) * 1000
                )
                
                memory_info = get_model_memory_info()
                current_memory = pb2.MemoryInfo(
                    ram_usage_gb=memory_info.get('ram_usage_gb', 0.0),
                    ram_percent=memory_info.get('ram_percent', 0.0),
                    gpu_allocated_gb=memory_info.get('gpu_allocated_gb', 0.0),
                    gpu_reserved_gb=memory_info.get('gpu_reserved_gb', 0.0),
                    gpu_free_gb=memory_info.get('gpu_free_gb', 0.0),
                    timestamp=int(time.time())
                )
                
                performance = pb2.PerformanceStats(
                    embedding_requests=self.model_stats['embedding_requests'],
                    reranking_requests=self.model_stats['reranking_requests'],
                    prompt_rewrite_requests=self.model_stats['prompt_rewrite_requests'],
                    prompt_compression_requests=self.model_stats['prompt_compression_requests'],
                    error_count=self.model_stats['error_count']
                )
                
                status = pb2.ServerStatus(
                    server_version="1.0.0",
                    uptime_seconds=uptime,
                    total_requests=self.request_count,
                    avg_response_time_ms=avg_response,
                    current_memory=current_memory,
                    performance=performance
                )
                
                return pb2.GetServerStatusResponse(
                    status=status,
                    success=True
                )
            
        except Exception as e:
            logger.error("Error in GetServerStatus: %s", str(e))
            return pb2.GetServerStatusResponse(
                success=False,
                error_message=f'Internal error: {str(e)}'
            )
    
    def _update_stats(self, stat_name: str, processing_time: float, count: int = 1):
        """Update internal statistics"""
        with self._stats_lock:
            self.request_count += count
            self.total_processing_time += processing_time
            if stat_name in self.model_stats:
                self.model_stats[stat_name] += count


def create_server(port: int = 50051, max_workers: int = 10, max_batch_size: int = 32):
    """Create and configure gRPC server"""
    if not PROTO_AVAILABLE:
        logger.error("Protobuf files not available. Please run: python -m grpc_tools.protoc --python_out=. --grpc_python_out=. ml_models.proto")
        return None
    
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ('grpc.keepalive_time_ms', 60000),
            ('grpc.keepalive_timeout_ms', 5000),
            ('grpc.keepalive_permit_without_calls', True),
            ('grpc.http2.max_pings_without_data', 0),
            ('grpc.http2.min_time_between_pings_ms', 10000),
            ('grpc.http2.min_ping_interval_without_data_ms', 300000),
            ('grpc.max_receive_message_length', 16 * 1024 * 1024),  # 16MB
            ('grpc.max_send_message_length', 16 * 1024 * 1024),     # 16MB
        ]
    )
    
    # Add servicer
    servicer = MLModelServicer(max_batch_size=max_batch_size)
    pb2_grpc.add_MLModelServiceServicer_to_server(servicer, server)
    
    # Add port
    server.add_insecure_port(f'[::]:{port}')
    
    return server, servicer


def main():
    """Main server entry point"""
    parser = argparse.ArgumentParser(description='ML Models gRPC Server')
    parser.add_argument('--port', type=int, default=50051, help='Server port')
    parser.add_argument('--max-workers', type=int, default=10, help='Max worker threads')
    parser.add_argument('--max-batch-size', type=int, default=32, help='Max batch size')
    parser.add_argument('--warmup', action='store_true', help='Warmup models at startup')
    parser.add_argument('--warmup-models', nargs='+', help='Specific models to warmup')
    
    args = parser.parse_args()
    
    if not PROTO_AVAILABLE:
        logger.error("Please install grpcio-tools and generate protobuf files:")
        logger.error("pip install grpcio-tools")
        logger.error("python -m grpc_tools.protoc --python_out=. --grpc_python_out=. ml_models.proto")
        return 1
    
    # Create server
    server_result = create_server(
        port=args.port, 
        max_workers=args.max_workers,
        max_batch_size=args.max_batch_size
    )
    
    if server_result is None:
        return 1
    
    server, servicer = server_result
    
    # Warmup models if requested
    if args.warmup:
        warmup_models(args.warmup_models)
    
    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info("Received signal %d, shutting down...", signum)
        cleanup_models()
        server.stop(grace=30)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start server
    try:
        server.start()
        logger.info("gRPC server started on port %d with %d workers", args.port, args.max_workers)
        logger.info("Max batch size: %d", args.max_batch_size)
        
        # Log initial memory
        memory_info = get_model_memory_info()
        logger.info("Initial memory usage: RAM=%.2fGB", memory_info['ram_usage_gb'])
        
        # Keep server running
        server.wait_for_termination()
        
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error("Server error: %s", str(e))
        return 1
    finally:
        cleanup_models()
        logger.info("Server shutdown complete")
    
    return 0


if __name__ == '__main__':
    exit(main())
