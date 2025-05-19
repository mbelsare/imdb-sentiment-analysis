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
        "torch==2.7.0",
        "transformers==4.51.3",
        "datasets==3.6.0",
        "scikit-learn==1.6.1",
        "pandas==2.2.3",
        "numpy==2.2.6",
        "matplotlib==3.10.3",
        "seaborn==0.13.2",
        "tqdm==4.67.1",
        "tensorboard==2.19.0",
        "protobuf==4.21.6",
        "urllib3==2.2.2",
        "hf-xet==1.1.2"
    ],
    python_requires=">=3.8",
)