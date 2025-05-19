"""
Utility functions for sentiment analysis.
"""

import json
import logging
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer

from buildops.sentiment_analysis.config.model_config import RANDOM_SEED, LOGS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(LOGS_DIR, f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    ]
)

logger = logging.getLogger(__name__)


def set_seed(seed=RANDOM_SEED):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def load_tokenizer(tokenizer_name):
    """
    Load a tokenizer from Hugging Face.

    Args:
        tokenizer_name: Name of the tokenizer

    Returns:
        AutoTokenizer: The loaded tokenizer
    """
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return tokenizer


def save_metrics(metrics, file_name, directory=LOGS_DIR):
    """
    Save evaluation metrics to a JSON file.

    Args:
        metrics (dict): Metrics to save
        file_name (str): Name of the output file
        directory (str): Directory to save the file
    """
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Remove non-serializable objects
    serializable_metrics = {}
    for key, value in metrics.items():
        if key == "confusion_matrix":
            serializable_metrics[key] = value.tolist()
        elif key == "plots":
            # For plots, just save the file paths
            serializable_metrics[key] = value
        elif isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        elif isinstance(value, dict):
            serializable_metrics[key] = {k: v for k, v in value.items() if not isinstance(v, np.ndarray)}
        else:
            serializable_metrics[key] = value

    # Save to file
    file_path = os.path.join(directory, file_name)
    with open(file_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)

    logger.info(f"Metrics saved to {file_path}")


def visualize_attention(model, tokenizer, text, device, layer=-1, head=0):
    """
    Visualize attention patterns for a given text.

    Args:
        model: The model
        tokenizer: The tokenizer
        text (str): Input text
        device: Device to run on
        layer (int): Which transformer layer to visualize
        head (int): Which attention head to visualize

    Returns:
        str: Path to the saved attention visualization
    """
    # This function only works for transformer models with attention
    if not hasattr(model, 'transformer') or not hasattr(model.transformer, 'encoder'):
        logger.warning("Attention visualization is only supported for transformer models")
        return None

    # Tokenize text
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text, return_tensors='pt').to(device)

    # Get attention weights
    with torch.no_grad():
        outputs = model.transformer(token_ids, output_attentions=True)
        attentions = outputs.attentions

    # Get attention from specific layer and head
    if layer < 0:
        layer = len(attentions) + layer
    attention = attentions[layer][0, head].cpu().numpy()

    # Create visualization
    fig, ax = plt.figure(figsize=(10, 10)), plt.subplot(1, 1, 1)

    # Display matrix
    im = ax.matshow(attention)

    # Set labels
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(tokens)

    # Add colorbar
    plt.colorbar(im)

    # Set title
    plt.title(f"Attention visualization (Layer {layer + 1}, Head {head + 1})")
    plt.tight_layout()

    # Save figure
    os.makedirs(LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(LOGS_DIR, f"attention_vis_{timestamp}.png")
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Attention visualization saved to {output_path}")
    return output_path


def count_model_parameters(model):
    """
    Count the number of trainable parameters in a model.

    Args:
        model: The model

    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)