#!/usr/bin/env python3
"""
Setup script for the ReqQuest project
"""

from setuptools import setup, find_packages
import os

# Read the requirements
def read_requirements():
    requirements = []
    if os.path.exists("HybirdRAG/comp/requirements.txt"):
        with open("HybirdRAG/comp/requirements.txt", "r") as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        # Default requirements for the entire project
        requirements = [
            "grpcio>=1.70.0",
            "grpcio-tools>=1.70.0", 
            "protobuf>=4.21.0",
            "pymilvus>=2.4.0",
            "spacy>=3.7.0",
            "langchain>=0.1.0",
            "llama-index>=0.10.0",
            "llama-index-core>=0.10.0",
            "llama-index-embeddings-openai>=0.1.0",
            "neo4j>=6.0.0",
            "schedule>=1.2.0",
            "pypdf>=3.17.0",
            "pytesseract>=0.3.10",
            "Pillow>=10.0.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "scikit-learn>=1.3.0",
            "transformers>=4.30.0",
            "torch>=2.0.0",
            "openai>=1.0.0",
            "python-dotenv>=1.0.0"
        ]
    return requirements

# Read README for long description
def read_readme():
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    return "ReqQuest - A hybrid RAG system combining vector and graph-based retrieval"

setup(
    name="reqquest",
    version="1.0.0",
    description="ReqQuest - A hybrid RAG system combining vector and graph-based retrieval",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/ReqQuest",
    
    # Package discovery and configuration
    packages=find_packages(where=".", exclude=["tests*", "QuickTA*", "docs*"]),
    package_dir={"": "."},
    
    # Include package data
    package_data={
        "HybirdRAG.comp": ["*.proto", "*.py"],
        "HybirdRAG": ["*.py"],
        "HybirdRAG.VectorRAG": ["*.py"],
        "HybirdRAG.GraphRAG": ["*.py"]
    },
    
    # Dependencies
    install_requires=read_requirements(),
    python_requires=">=3.8",
    
    # Entry points for CLI tools
    entry_points={
        'console_scripts': [
            'reqquest=HybirdRAG.cli_utils:main',
            'hybrid-rag=HybirdRAG.pipeline:main',
        ],
    },
    
    # Development dependencies
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.5.0',
        ],
        'jupyter': [
            'jupyter>=1.0.0',
            'ipykernel>=6.0.0',
            'notebook>=7.0.0',
        ]
    },
    
    # Metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    
    keywords="rag, retrieval, embeddings, vector-database, graph-rag, hybrid-rag",
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/yourusername/ReqQuest/issues",
        "Source": "https://github.com/yourusername/ReqQuest",
        "Documentation": "https://github.com/yourusername/ReqQuest/docs",
    },
)

