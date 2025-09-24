#!/usr/bin/env python3
"""
Setup script for the Astrophysics Methodology Classification and Clustering system.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="astronlp",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive pipeline for automatically identifying and clustering methodological approaches in astrophysics research papers",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/astronlp",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "gpu": [
            "cupy-cuda11x>=10.6.0",
            "cupy-cuda12x>=12.0.0",
        ],
        "dev": [
            "pytest>=7.1.0",
            "pytest-cov>=3.0.0",
            "pytest-mock>=3.7.0",
            "pytest-benchmark>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
        ],
        "docs": [
            "mkdocs>=1.4.0",
            "mkdocs-material>=8.5.0",
            "mkdocs-mermaid>=0.6.0",
            "mkdocs-macros>=0.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "astronlp-train=classification.bert_classifier_trainer:main",
            "astronlp-cluster=clustering.level_clustering.clustering_framework.thesis_master_runner:main",
            "astronlp-evaluate=evaluation.evaluate_multi:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.md", "*.txt"],
    },
    zip_safe=False,
)
