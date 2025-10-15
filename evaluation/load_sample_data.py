#!/usr/bin/env python3
"""
Load sample data from evaluation datasets into HybridRAG pipeline.
This will populate both the vector database and graph database with relevant content.
"""

import json
import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, '.')

from HybirdRAG.pipeline import HybridRAGPipeline


def load_sample_from_hotpotqa(file_path: Path, limit: int = 10):
    """Load sample data from HotpotQA dataset."""
    print(f"📖 Loading sample data from {file_path}")
    
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit:
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


def load_sample_from_2wiki(file_path: Path, limit: int = 10):
    """Load sample data from 2WikiMultiHopQA dataset."""
    print(f"📖 Loading sample data from {file_path}")
    
    documents = []
    
    # Check if it's a JSONL file or regular JSON file
    if file_path.suffix.lower() in {'.jsonl', '.jl'}:
        # JSONL format
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= limit:
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
                    if i >= limit:
                        break
                    documents.extend(_extract_documents_from_2wiki(item, "2wiki"))
            elif isinstance(data, dict):
                documents.extend(_extract_documents_from_2wiki(data, "2wiki"))
    
    return documents


def _extract_documents_from_2wiki(data: dict, source: str):
    """Extract documents from 2wiki data structure."""
    documents = []
    
    # Extract context information
    if 'context' in data:
        context = data['context']
        if 'title' in context and 'sentences' in context:
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


def load_sample_from_musique(file_path: Path, limit: int = 10):
    """Load sample data from MuSiQue dataset."""
    print(f"📖 Loading sample data from {file_path}")
    
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            data = json.loads(line.strip())
            
            # Extract context information
            if 'context' in data:
                context = data['context']
                if 'title' in context and 'sentences' in context:
                    for title_idx, title in enumerate(context['title']):
                        if title_idx < len(context['sentences']):
                            sentences = context['sentences'][title_idx]
                            content = f"Title: {title}\n\n" + " ".join(sentences)
                            
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
    print("🚀 Initializing HybridRAG pipeline...")
    
    # Initialize pipeline with remote services
    pipeline = HybridRAGPipeline()
    
    all_documents = []
    
    # Load sample data from each dataset
    data_dir = Path("data")
    
    if (data_dir / "hotpot_dev.jsonl").exists():
        hotpot_docs = load_sample_from_hotpotqa(data_dir / "hotpot_dev.jsonl", limit=50)  # Reduced for faster testing
        all_documents.extend(hotpot_docs)
        print(f"✅ Loaded {len(hotpot_docs)} documents from HotpotQA")
    
    if (data_dir / "2wiki_dev_converted.json").exists():
        wiki_docs = load_sample_from_2wiki(data_dir / "2wiki_dev_converted.json", limit=50)  # Increased from 5
        all_documents.extend(wiki_docs)
        print(f"✅ Loaded {len(wiki_docs)} documents from 2WikiMultiHopQA")
    
    if (data_dir / "musique_dev.jsonl").exists():
        musique_docs = load_sample_from_musique(data_dir / "musique_dev.jsonl", limit=50)  # Increased from 5
        all_documents.extend(musique_docs)
        print(f"✅ Loaded {len(musique_docs)} documents from MuSiQue")
    
    if not all_documents:
        print("❌ No documents found to load")
        return 1
    
    print(f"\n📚 Total documents to load: {len(all_documents)}")
    print("🔄 Adding documents to HybridRAG pipeline...")
    
    # Add documents to the pipeline (this will populate both vector and graph databases)
    pipeline.add_documents(all_documents)
    
    print("✅ All documents loaded successfully!")
    print("🎉 HybridRAG pipeline is now ready with sample data")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
