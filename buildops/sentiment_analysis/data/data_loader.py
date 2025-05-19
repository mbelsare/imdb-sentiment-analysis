"""
Dataset and DataLoader implementations for sentiment analysis.
"""

import logging
import os
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default configuration
MAX_SEQUENCE_LENGTH = 512
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42
BATCH_SIZE = 16
PRETRAINED_MODEL_NAME = "bert-base-uncased"

class IMDBDataset(Dataset):
    """
    Dataset class for IMDB sentiment analysis.

    This class handles loading and preprocessing the IMDB dataset,
    including tokenization using a pre-trained transformer tokenizer.
    """

    def __init__(self, texts, labels, tokenizer, max_length=MAX_SEQUENCE_LENGTH):
        """
        Initialize the dataset.

        Args:
            texts (list): List of review texts
            labels (list): List of sentiment labels (0 for negative, 1 for positive)
            tokenizer: Hugging Face tokenizer to use
            max_length (int): Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Calculate review lengths for potential use in loss weighting
        self.review_lengths = [len(text.split()) for text in texts]
        self.max_review_length = max(self.review_lengths)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        review_length = self.review_lengths[idx]

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Extract features from the tokenized input
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(label, dtype=torch.float).view(1),  # Ensure label is a 1D tensor
            "review_length": torch.tensor(review_length, dtype=torch.float32) / self.max_review_length  # Normalized length
        }

        return item


def load_and_preprocess_data(csv_path=None):
    """
    Load the IMDB dataset from CSV and preprocess it.

    Args:
        csv_path (str, optional): Path to the CSV file

    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    # If path not provided, look for "imdb_dataset.csv" in the dataset directory
    if csv_path is None:
        project_root = Path(__file__).parent.parent
        print(f"Project root: {project_root}")
        csv_path = project_root / "dataset" / "imdb_dataset.csv"
        print(f"CSV path: {csv_path}")

        if not os.path.exists(csv_path):
            csv_path = Path("imdb_dataset.csv")

    logger.info(f"Loading dataset from {csv_path}")

    # Load CSV dataset
    df = pd.read_csv(csv_path)

    # Basic preprocessing
    df["review"] = df["review"].str.replace("<br />", " ")  # Remove HTML line breaks
    df["review"] = df["review"].str.replace("[^a-zA-Z0-9\s]", "", regex=True)  # Keep only alphanumeric
    df["review"] = df["review"].str.lower()  # Convert to lowercase

    # Convert sentiment labels to numeric
    df["sentiment_label"] = df["sentiment"].apply(lambda x: 1 if x == "positive" else 0)

    logger.info(f"Loaded {len(df)} reviews")
    return df


def create_data_loaders(batch_size=BATCH_SIZE, csv_path=None, tokenizer_name=PRETRAINED_MODEL_NAME):
    """
    Create train, validation, and tests dataset loaders.

    Args:
        batch_size (int): Batch size
        csv_path (str, optional): Path to the CSV file
        tokenizer_name (str): Name of the tokenizer to use

    Returns:
        tuple: (train_loader, val_loader, test_loader, tokenizer)
    """
    # Load and preprocess dataset
    df = load_and_preprocess_data(csv_path)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Split dataset into train/val/tests
    X = df["review"].tolist()
    y = df["sentiment_label"].tolist()

    # First split into train and temp (val+tests)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(VAL_RATIO + TEST_RATIO), random_state=RANDOM_SEED
    )

    # Then split temp into val and tests
    test_ratio_adjusted = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_ratio_adjusted, random_state=RANDOM_SEED
    )

    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    # Create datasets
    train_dataset = IMDBDataset(X_train, y_train, tokenizer)
    val_dataset = IMDBDataset(X_val, y_val, tokenizer)
    test_dataset = IMDBDataset(X_test, y_test, tokenizer)

    # Create dataset loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, tokenizer