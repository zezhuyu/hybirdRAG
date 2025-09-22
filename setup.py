#!/usr/bin/env python3
"""
Setup script for the comp package
"""

from setuptools import setup, find_packages
import os

# Read the requirements
def read_requirements():
    requirements = []
    if os.path.exists("comp/requirements.txt"):
        with open("comp/requirements.txt", "r") as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        # Default requirements
        requirements = [
            "grpcio>=1.75.0",
            "grpcio-tools>=1.75.0", 
            "protobuf>=6.0.0"
        ]
    return requirements

setup(
    name="comp",
    version="1.0.0",
    description="ML Models gRPC Client Package",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="."),
    package_dir={"": "."},
    package_data={
        "comp": ["*.proto", "*.py"]
    },
    install_requires=read_requirements(),
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)

