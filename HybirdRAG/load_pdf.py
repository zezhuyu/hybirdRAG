"""Utility script to ingest a PDF into the Hybrid RAG stack.

This script extracts text from PDFs using PyPDF (default) or OCR (optional),
then ingests the resulting documents into the HybridRAG pipeline.

Text Extraction Methods:
- PyPDF (default): Fast, reliable for text-based PDFs
- OCR (--use-ocr): For scanned PDFs, but may cause compatibility issues on ARM Mac

The script expects environment variables describing service endpoints:

    SERVICE_HOST=<grpc_host:port>
    MILVUS_HOST=<host[:port]>
    OPENAI_BASE_URL=<http url>
    OPENAI_API_KEY=<key>
    NEO4J_HOST=<host[:http_port]>
    NEO4J_USERNAME=<username>
    NEO4J_PASSWORD=<password>

Additional optional variables:

    NEO4J_URL=<bolt uri> (defaults to bolt://<host>:<bolt_port>)
    NEO4J_BOLT_PORT=<port> (defaults to 7687 if not supplied)
    NEO4J_DATABASE=<db name, defaults to "neo4j">
    GRAPH_VECTOR_COLLECTION=<Milvus collection for graph embeddings>
    GRAPH_VECTOR_DIM=<embedding dimension, default 1024>
    MILVUS_USERNAME / MILVUS_PASSWORD (if auth is enabled)

Usage:

    # Use PyPDF (default, recommended)
    python load_pdf.py /path/to/book.pdf --env .env
    
    # Use OCR for scanned PDFs (may cause bus errors on ARM Mac)
    python load_pdf.py /path/to/book.pdf --env .env --use-ocr

The script will extract text from the PDF, then add the resulting
documents to the configured `HybridRAGPipeline`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .pipeline import HybridRAGPipeline
from .ocr import OCRExtractor


def _ocr_pdf(pdf_path: Path, language: str = "en", use_ocr: bool = False) -> List[str]:
    """Extract text from PDF using OCR or PyPDF fallback."""
    if use_ocr:
        try:
            ocr_extractor = OCRExtractor(lang=language)
            return ocr_extractor.pdf_to_text(str(pdf_path))
        except ImportError as e:
            print(f"⚠️  OCR not available ({e}), falling back to PyPDF...")
            return _extract_text_with_pypdf(pdf_path)
        except Exception as e:
            print(f"⚠️  OCR failed ({e}), falling back to PyPDF...")
            return _extract_text_with_pypdf(pdf_path)
    else:
        return _extract_text_with_pypdf(pdf_path)

def _extract_text_with_pypdf(pdf_path: Path) -> List[str]:
    """Extract text from PDF using PyPDF as fallback."""
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text.strip():  # Only add non-empty pages
                    pages.append(text.strip())
            return pages
    except ImportError:
        print("PyPDF2 not available. Please install it with: pip install PyPDF2")
        raise
    except Exception as e:
        print(f"PyPDF extraction failed: {e}")
        raise


def _prepare_documents(pages: Iterable[str], source_path: Path) -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    title = source_path.stem.replace("_", " ")
    for idx, page_text in enumerate(pages, start=1):
        text = page_text.strip()
        if not text:
            continue
        documents.append(
            {
                "content": text,
                "title": title,
                "page": idx,
                "source": source_path.name,
                "source_path": str(source_path.resolve()),
            }
        )
    return documents


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR a PDF and ingest it into HybridRAG")
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF file to ingest")
    parser.add_argument(
        "--env",
        type=Path,
        default=Path(".env"),
        help="Path to a .env file with connection settings (default: .env)",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Language code for OCR (default: en)",
    )
    parser.add_argument(
        "--use-ocr",
        action="store_true",
        help="Enable OCR for text extraction (default: PyPDF for better compatibility)",
    )
    args = parser.parse_args()

    if not args.pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {args.pdf_path}")

    print("Building pipeline using environment configuration...")
    pipeline = HybridRAGPipeline()

    print(f"Running text extraction on {args.pdf_path} ...")
    pages = _ocr_pdf(args.pdf_path, language=args.lang, use_ocr=args.use_ocr)
    print(f"Extracted text from {len(pages)} pages")

    documents = _prepare_documents(pages, args.pdf_path)
    if not documents:
        raise RuntimeError("No text extracted from the PDF; aborting ingestion.")

    print(f"Adding {len(documents)} page documents to the pipeline...")
    pipeline.add_documents(documents)

    print("Rebuilding graph communities for updated knowledge base...")
    pipeline.graph_rag.build_communities()

    print("Ingestion complete.")


if __name__ == "__main__":
    main()
