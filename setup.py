"""
Setup script for RAG Document Q&A System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="rag-document-qa",
    version="1.0.0",
    description="Document Question Answering System using RAG (Retrieval-Augmented Generation)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/rag_document_qa",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "rag-qa=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="rag retrieval question-answering nlp deep-learning transformers",
    project_urls={
        "Documentation": "https://github.com/yourusername/rag_document_qa/blob/main/README.md",
        "Source": "https://github.com/yourusername/rag_document_qa",
        "Tracker": "https://github.com/yourusername/rag_document_qa/issues",
    },
)
