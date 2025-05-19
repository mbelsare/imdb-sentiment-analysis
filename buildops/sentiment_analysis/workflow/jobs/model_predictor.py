#!/usr/bin/env python
"""
Script to make predictions with the trained sentiment analysis model.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from buildops.sentiment_analysis.mlobjects.models.sentiment_analysis_model import get_model
from model_evaluator import predict_sentiment, visualize_attention
from buildops.sentiment_analysis.mlobjects.utils.model_utils import load_tokenizer
from buildops.sentiment_analysis.config.model_config import (
    MODEL_SAVE_PATH,
    PRETRAINED_MODEL_NAME
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Make predictions with the sentiment analysis model")

    parser.add_argument("--input", type=str, required=True,
                        help="Input text for sentiment analysis")
    parser.add_argument("--model_path", type=str, default=os.path.join(MODEL_SAVE_PATH, "best_model.pt"),
                        help="Path to the trained model checkpoint")
    parser.add_argument("--model_type", type=str, default="transformer", choices=["transformer", "lstm"],
                        help="Type of model to use")
    parser.add_argument("--tokenizer", type=str, default=PRETRAINED_MODEL_NAME,
                        help="Name of the tokenizer to use")
    parser.add_argument("--visualize_attention", action="store_true",
                        help="Whether to visualize attention weights (transformer only)")

    return parser.parse_args()


def main():
    """Main prediction function."""
    args = parse_args()

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer)

    # Load model
    if args.model_type == "transformer":
        model = get_model(model_type="transformer", pretrained_model_name=args.pretrained_model, num_classes=1)
    else:
        vocab_size = len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else tokenizer.vocab_size
        model = get_model(
            model_type="lstm",
            vocab_size=vocab_size,
            embedding_dim=300,
            hidden_dim=256,
            num_layers=2,
            num_classes=2,
        )

    # Load model weights
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logger.info(f"Model loaded from {args.model_path}")

    # Make prediction
    result = predict_sentiment(model, tokenizer, args.input, device)

    # Print prediction
    print("\nSentiment Analysis Result:")
    print(f"Text: {result['text']}")
    print(f"Predicted sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Probabilities: Negative: {result['probabilities']['Negative']:.4f}, "
          f"Positive: {result['probabilities']['Positive']:.4f}")

    # Generate attention visualization if requested (and if using transformer)
    if args.visualize_attention and args.model_type == "transformer":
        vis_path = visualize_attention(model, tokenizer, args.input, device)
        if vis_path:
            print(f"\nAttention visualization saved to {vis_path}")

    # Save result to file
    with open("prediction_result.json", "w") as f:
        json.dump(result, f, indent=4)

    logger.info("Prediction complete!")


if __name__ == "__main__":
    main()