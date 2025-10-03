#!/usr/bin/env python3
"""
ReqQuest Hybrid RAG Query Test Script

This script provides a simple and reliable way to test document retrieval 
from your ReqQuest hybrid RAG system. It handles PDF loading and hybrid 
search queries using both vector similarity and graph-based reasoning.

Usage:
    # Load PDF and query in one command
    python db_example.py --load-pdf book.pdf --query "machine learning"
    
    # Query existing documents with hybrid RAG
    python db_example.py --query "algorithms" --limit 5
    
    # Just load PDF
    python db_example.py --load-pdf book.pdf
    
    # Check what's available
    python db_example.py --status
"""

import argparse
import sys
import os
import time
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, '.')

from HybirdRAG.load_pdf import _ocr_pdf, _prepare_documents
from HybirdRAG.pipeline import HybridRAGPipeline


def load_pdf(pdf_path: str):
    """Load PDF documents into the hybrid RAG pipeline."""
    try:
        print(f"📖 Loading PDF: {pdf_path}")
        
        # Check file exists
        if not Path(pdf_path).exists():
            print(f"❌ File not found: {pdf_path}")
            return False
        pages = _ocr_pdf(Path(pdf_path))
        
        # Prepare documents
        documents = _prepare_documents(pages, Path(pdf_path))
        
        # Store in pipeline (try hybrid first, fallback to vector-only if needed)
        pipeline = HybridRAGPipeline()
        
        # Check if it's a hybrid pipeline or vector-only pipeline
        if hasattr(pipeline, 'vector_rag'):
            # HybridRAGPipeline - add_document expects a single document
            import time
            start_time = time.time()
            progress_bar_width = 20
            
            progress = 1 / len(documents) if len(documents) > 0 else 0
            filled = int(progress * progress_bar_width)
            bar = "█" * filled + "░" * (progress_bar_width - filled)
            percentage = progress * 100
            print(f"\r📄 Processing page {1}/{len(documents)} [{bar}] ({percentage:.1f}%)                                    ", end='', flush=True)
            
            for i, doc in enumerate(documents):
                # Calculate progress before processing this page
                progress = i / len(documents) if len(documents) > 0 else 0
                filled = int(progress * progress_bar_width)
                bar = "█" * filled + "░" * (progress_bar_width - filled)
                percentage = progress * 100
                
                # Show processing status with current progress
                
                
                # Process the document
                page_start = time.time()
                pipeline.add_document(doc)
                page_time = time.time() - page_start
                
                # Calculate final progress and ETA after processing
                elapsed = time.time() - start_time
                final_progress = (i + 1) / len(documents)
                
                # Calculate ETA based on average time per page
                avg_time_per_page = elapsed / (i + 1)
                remaining_pages = len(documents) - (i + 1)
                eta_seconds = avg_time_per_page * remaining_pages
                
                # Format ETA nicely
                if eta_seconds > 3600:  # More than 1 hour
                    eta_hours = int(eta_seconds // 3600)
                    eta_min = int((eta_seconds % 3600) // 60)
                    eta_str = f"{eta_hours}h {eta_min}m"
                elif eta_seconds > 60:  # More than 1 minute
                    eta_min, eta_sec = divmod(int(eta_seconds), 60)
                    eta_str = f"{eta_min}m {eta_sec}s"
                else:  # Less than 1 minute
                    eta_str = f"{int(eta_seconds)}s" if eta_seconds > 0 else "0s"
                
                # Update progress bar after completion
                final_filled = int(final_progress * progress_bar_width)
                final_bar = "█" * final_filled + "░" * (progress_bar_width - final_filled)
                final_percentage = final_progress * 100
                
                # Update the same line with completion status
                print(f"\r📄 Completed page {i + 1}/{len(documents)} [{final_bar}] ({final_percentage:.1f}%) - {page_time:.1f}s - ETA: {eta_str}                    ", end='', flush=True)
            
            print(f"✅ All {len(documents)} pages processed successfully in {elapsed:.1f}s!")
        else:
            # VectorRAGPipeline - add_document expects a list of documents
            pipeline.add_document(documents)
        
        print(f"✅ Successfully stored {len(documents)} documents")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading PDF: {e}")
        return False


def search_documents(query: str, limit: int = 5):
    """Search documents using hybrid RAG (vector + graph)."""
    try:
        pipeline = HybridRAGPipeline()
        results = pipeline.query(query, limit=limit)
        
        # Debug output
        
        if not results:
            print("❌ No documents found")
            return []
        
        for i, result in enumerate(results):
            print(f"\n📄 Result #{i + 1}:")
            
            if hasattr(result, 'entity'):
                # Milvus SearchResult object
                entity = result.entity
                score = getattr(result, 'score', 0.0)
                content = getattr(entity, 'content', 'No content')
                title = getattr(entity, 'title', 'Untitled')
                page = getattr(entity, 'page', 'Unknown')
                
                print(f"   Score: {score:.4f}")
                print(f"   Title: {title}")
                print(f"   Page: {page}")
                
                # Content preview
                if content and len(content) > 250:
                    print(f"   Content: {content[:250]}...")
                elif content:
                    print(f"   Content: {content}")
                else:
                    print(f"   Content: No content available")
            elif isinstance(result, str):
                # String result from hybrid pipeline
                print(f"   Result: {result[:2000]}{'...' if len(result) > 2000 else ''}")
            else:
                # Other types
                print(f"   Result: {result}")
            
            print("-" * 30)
        
        return results
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        return []


def check_status():
    """Check system status."""
    try:
        
        
        pipeline = HybridRAGPipeline()
        
        # Handle both hybrid and vector-only pipelines
        if hasattr(pipeline, 'vector_rag'):
            milvus = pipeline.vector_rag.milvus
            collection_name = pipeline.vector_rag.collection_name
        else:
            milvus = pipeline.milvus
            collection_name = pipeline.vector_rag.collection_name
        
        if milvus.has_collection(collection_name):
            try:
                stats = milvus.get_collection_stats(collection_name)
                count = stats.get('row_count', 0)
                print(f"✅ Database: Connected")
                print(f"📊 Documents: {count}")
                
                if count > 0:
                    print("🎉 Ready for queries!")
                    return True
                else:
                    print("⚠️ No documents - load PDF first")
                    return False
            except:
                print("✅ Database: Connected (ready for queries)")
                return True
        else:
            print("⚠️ No collection - load PDF first")
            return False
            
    except Exception as e:
        print(f"❌ Status check failed: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="ReqQuest Query Test Script",
        epilog="""
Examples:
  python db_example.py --load-pdf book.pdf --query "machine learning"
  python db_example.py --query "algorithms"
  python db_example.py --status
        """
    )
    
    parser.add_argument("--query", "-q", help="Search query")
    parser.add_argument("--limit", "-l", type=int, default=5, help="Result limit (default: 5)")
    parser.add_argument("--load-pdf", help="Load PDF file")
    parser.add_argument("--status", action="store_true", help="Check status")
    
    args = parser.parse_args()
    
    # Status check
    if args.status:
        check_status()
        return 0
    
    # Load PDF
    if args.load_pdf:
        if not load_pdf(args.load_pdf):
            return 1
        print()
    
    # Search
    if args.query:
        time_start = time.time()
        results = search_documents(args.query, args.limit)
        time_end = time.time()
        print(f"🔍 Query took {time_end - time_start:.2f} seconds")
        if results:
            print(f"\n🎉 Found {len(results)} results for '{args.query}'")
            return 0
        else:
            print(f"\n💡 No results for '{args.query}'")
            return 1
    elif args.load_pdf:
        print("✅ PDF loaded! Use --query to search.")
        return 0
    else:
        print("❌ Specify --query, --load-pdf, or --status")
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
