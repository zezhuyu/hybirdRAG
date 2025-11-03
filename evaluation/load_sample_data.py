#!/usr/bin/env python3
"""
Load sample data from evaluation datasets into HybridRAG pipeline.
This will populate both the vector database and graph database with relevant content.
"""

import json
import sys
import time
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, '.')

from HybirdRAG.pipeline import HybridRAGPipeline


def load_sample_from_hotpotqa(file_path: Path, limit: int = -1):
    """Load sample data from HotpotQA dataset."""
    print(f"📖 Loading sample data from {file_path}")
    
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit != -1 and i >= limit:
                break
            data = json.loads(line.strip())
            
            # Extract context information from HotpotQA
            if 'context' in data:
                context = data['context']
                if 'title' in context and 'sentences' in context:
                    # Create documents from the context
                    for title_idx, title in enumerate(context['title']):
                        if title_idx < len(context['sentences']):
                            sentences = context['sentences'][title_idx]
                            content = f"Title: {title}\n\n" + " ".join(sentences)
                            
                            doc = {
                                "content": content,
                                "source": "hotpotqa",
                                "title": title,
                                "page": 0,
                                "source_path": f"hotpotqa_{data.get('id', '')}",
                                "chapter": 0,
                                "section": 0
                            }
                            documents.append(doc)
    
    return documents


def load_sample_from_2wiki(file_path: Path, limit: int = -1):
    """Load sample data from 2WikiMultiHopQA dataset."""
    print(f"📖 Loading sample data from {file_path}")
    
    documents = []
    
    # Check if it's a JSONL file or regular JSON file
    if file_path.suffix.lower() in {'.jsonl', '.jl'}:
        # JSONL format
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit != -1 and i >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                documents.extend(_extract_documents_from_2wiki(data, "2wiki"))
    else:
        # Regular JSON format
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                for i, item in enumerate(data):
                    if limit != -1 and i >= limit:
                        break
                    documents.extend(_extract_documents_from_2wiki(item, "2wiki"))
            elif isinstance(data, dict):
                documents.extend(_extract_documents_from_2wiki(data, "2wiki"))
    
    return documents


def _extract_documents_from_2wiki(data: dict, source: str):
    """Extract documents from 2wiki data structure.
    
    Handles two formats:
    1. Dict format: {'title': [...], 'sentences': [[...], [...]]}
    2. List format: [['Title1', 'Text1'], ['Title2', 'Text2'], ...]
    """
    documents = []
    
    # Extract context information
    if 'context' in data:
        context = data['context']
        
        # Check if context is a list of [title, text] pairs (new format)
        if isinstance(context, list) and len(context) > 0:
            # Check if first element is a list (pair format)
            if isinstance(context[0], list) and len(context[0]) >= 2:
                # Format: [['Title', 'Text'], ['Title2', 'Text2'], ...]
                for item in context:
                    try:
                        if isinstance(item, list) and len(item) >= 2:
                            title = str(item[0]).strip() if item[0] is not None else ""
                            text = str(item[1]).strip() if item[1] is not None else ""
                            
                            if title and text:
                                content = f"Title: {title}\n\n{text}"
                                
                                doc = {
                                    "content": content,
                                    "source": source,
                                    "title": title,
                                    "page": 0,
                                    "source_path": f"{source}_{data.get('id', '')}",
                                    "chapter": 0,
                                    "section": 0
                                }
                                documents.append(doc)
                    except Exception as e:
                        # Skip malformed items but continue processing
                        continue
        
        # Check if context is a dict with 'title' and 'sentences' (old format)
        elif isinstance(context, dict) and 'title' in context and 'sentences' in context:
            for title_idx, title in enumerate(context['title']):
                if title_idx < len(context['sentences']):
                    sentences = context['sentences'][title_idx]
                    content = f"Title: {title}\n\n" + " ".join(sentences)
                    
                    doc = {
                        "content": content,
                        "source": source,
                        "title": title,
                        "page": 0,
                        "source_path": f"{source}_{data.get('id', '')}",
                        "chapter": 0,
                        "section": 0
                    }
                    documents.append(doc)
    
    return documents


def load_sample_from_musique(file_path: Path, limit: int = -1):
    """Load sample data from MuSiQue dataset."""
    print(f"📖 Loading sample data from {file_path}")
    
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit != -1 and i >= limit:
                break
            data = json.loads(line.strip())
            
            # MuSiQue uses 'paragraphs' instead of 'context'
            if 'paragraphs' in data:
                for para in data['paragraphs']:
                    title = para.get('title', '')
                    paragraph_text = para.get('paragraph_text', '')
                    
                    if title and paragraph_text:
                        content = f"Title: {title}\n\n{paragraph_text}"
                        
                        doc = {
                            "content": content,
                            "source": "musique",
                            "title": title,
                            "page": 0,
                            "source_path": f"musique_{data.get('id', '')}",
                            "chapter": 0,
                            "section": 0
                        }
                        documents.append(doc)
    
    return documents


def main():
    """Load sample data into HybridRAG pipeline."""
    # Load environment variables from .env file
    import os
    env_path = Path("HybirdRAG/.env")
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print(f"✅ Loaded environment from {env_path}")
    
    print("🚀 Initializing HybridRAG pipeline...")
    
    # Initialize pipeline with remote services
    pipeline = HybridRAGPipeline(collection_name="wiki_rag")
    
    all_documents = []
    
    # Load sample data from each dataset
    data_dir = Path("evaluation/data")
    
    # if (data_dir / "hotpot_dev.jsonl").exists():
    #     print(f"📖 Loading entire HotpotQA dataset (this may take a while)...")
    #     hotpot_docs = load_sample_from_hotpotqa(data_dir / "hotpot_dev.jsonl")
    #     all_documents.extend(hotpot_docs)
    #     print(f"✅ Loaded {len(hotpot_docs)} documents from entire HotpotQA dataset")
    # else:
    #     print(f"⚠️  HotpotQA file not found at {data_dir / 'hotpot_dev.jsonl'}")
    
    if (data_dir / "2wiki_dev_converted.json").exists():
        wiki_file = data_dir / "2wiki_dev_converted.json"
        wiki_docs = load_sample_from_2wiki(wiki_file)
        all_documents.extend(wiki_docs)
        print(f"✅ Loaded {len(wiki_docs)} documents from 2WikiMultiHopQA")
    else:
        print(f"⚠️  2Wiki file not found at {data_dir / '2wiki_dev_converted.json'}")
    
    # if (data_dir / "musique_dev.jsonl").exists():
    #     print(f"📖 Loading entire MuSiQue dataset (this may take a while)...")
    #     musique_docs = load_sample_from_musique(data_dir / "musique_dev.jsonl")  # Increased from 5
    #     all_documents.extend(musique_docs)
    #     print(f"✅ Loaded {len(musique_docs)} documents from MuSiQue")
    
    if not all_documents:
        print("❌ No documents found to load")
        return 1
    
    print(f"\n📚 Total documents to load: {len(all_documents)}")
    
    start_time = time.time()
    
    # Temporarily disable community building during loading for much faster performance
    # We'll build communities once at the end instead of after every batch
    print("⚙️  Disabling community building during loading (will build once at the end)\n")
    original_build_communities = pipeline.graph_rag.build_communities
    pipeline.graph_rag.build_communities = lambda: None
    
    # Process documents in batches for better progress monitoring
    batch_size = 100  # Process 100 documents at a time
    total_docs = len(all_documents)
    total_loaded = 0
    
    for i in range(0, total_docs, batch_size):
        batch = all_documents[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_docs + batch_size - 1) // batch_size

        # if batch_num < 119:
        #     continue
        
        print(f"📦 Processing batch {batch_num}/{total_batches} ({len(batch)} documents)...")
        batch_start = time.time()
        
        # Add batch to pipeline
        pipeline.add_documents(batch)
        
        batch_time = time.time() - batch_start
        total_loaded += len(batch)
        elapsed = time.time() - start_time
        
        # Calculate progress and ETA
        progress_pct = (total_loaded / total_docs) * 100
        avg_time_per_doc = elapsed / total_loaded
        remaining_docs = total_docs - total_loaded
        eta_seconds = avg_time_per_doc * remaining_docs
        
        print(f"   ✅ Batch completed in {batch_time:.1f}s ({batch_time/len(batch):.2f}s per doc)")
        print(f"   Progress: {total_loaded}/{total_docs} ({progress_pct:.1f}%)")
        print(f"   Elapsed: {elapsed/60:.1f}m | ETA: {eta_seconds/60:.1f}m\n")
    
    # Restore original method and build communities once at the end
    pipeline.graph_rag.build_communities = original_build_communities
    print(f"\n🏗️  Building graph communities for all {total_docs} documents...")
    print("   This is a one-time operation and may take several minutes...")
    community_start = time.time()
    pipeline.graph_rag.build_communities()
    community_time = time.time() - community_start
    print(f"   ✅ Communities built in {community_time/60:.1f} minutes\n")
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✅ All documents loaded successfully!")
    print(f"   Total documents: {total_docs}")
    print(f"   Total time: {elapsed_time/60:.1f} minutes")
    print(f"   Average: {elapsed_time/total_docs:.2f}s per document")
    print("🎉 HybridRAG pipeline is now ready with full dataset")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
