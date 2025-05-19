#!/usr/bin/env python
"""
Setup script for the sentiment analysis package.
"""

from setuptools import setup, find_packages

setup(
    name="imdb-sentiment-analysis",
    version="0.1.0",
    description="Sentiment analysis for IMDB movie reviews",
    author="Manish",
    author_email="manishbelsare2003@gmail.com",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "transformers>=4.18.0",
        "datasets>=2.0.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "tensorboard>=2.8.0",
    ],
    python_requires=">=3.8",
)