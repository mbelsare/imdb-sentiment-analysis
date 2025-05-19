"""
Configuration parameters for the sentiment analysis project.
"""

import os
from pathlib import Path

# Base directories
ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
DATA_DIR = os.path.join(ROOT_DIR, "dataset")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Data files
IMDB_CSV_PATH = os.path.join(DATA_DIR, "imdb_dataset.csv")

# Model parameters
MAX_SEQUENCE_LENGTH = 512  # Maximum sequence length for BERT tokenizer
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_PROPORTION = 0.1
SAVE_STEPS = 1000
LOGGING_STEPS = 100

# Training parameters
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

# GPU settings
USE_CUDA = True
CUDA_DEVICE = 0

# Model configuration
MODEL_TYPE = "bert"
PRETRAINED_MODEL_NAME = "bert-base-uncased"
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "sentiment_model")

# Loss function parameters
LENGTH_WEIGHT_FACTOR = 0.2  # Factor for review length in custom loss
CONFIDENCE_PENALTY_FACTOR = 0.1  # Factor for confidence penalty in custom loss
CLASS_WEIGHTS = [1.0, 1.0]  # Weights for [negative, positive] classes

# Tokenizer settings
TOKENIZER_PARAMS = {
    "max_length": MAX_SEQUENCE_LENGTH,
    "padding": "max_length",
    "truncation": True,
    "return_tensors": "pt"
}