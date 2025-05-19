#!/usr/bin/env python
"""
Script to train the sentiment analysis model.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from sympy.printing.pytorch import torch

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from buildops.sentiment_analysis.data.data_loader import create_data_loaders
from buildops.sentiment_analysis.mlobjects.models.sentiment_analysis_model import get_model
from buildops.sentiment_analysis.mlobjects.common.loss_function import get_loss_function
from buildops.sentiment_analysis.mlobjects.models.train import train_model, get_training_device
from buildops.sentiment_analysis.workflow.jobs.model_evaluator import detailed_evaluation
from buildops.sentiment_analysis.mlobjects.utils.model_utils import set_seed, count_model_parameters, save_metrics
from buildops.sentiment_analysis.config.model_config import (
    NUM_EPOCHS,
    TRAIN_BATCH_SIZE,
    LEARNING_RATE,
    PRETRAINED_MODEL_NAME,
    CLASS_WEIGHTS, MODEL_SAVE_PATH
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train sentiment analysis model")

    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=TRAIN_BATCH_SIZE,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE,
                        help="Learning rate")
    parser.add_argument("--model_type", type=str, default="transformer", choices=["transformer", "lstm"],
                        help="Type of model to train")
    parser.add_argument("--loss_type", type=str, default="combined",
                        choices=["standard", "length_aware", "confidence_penalty", "combined"],
                        help="Type of loss function to use")
    parser.add_argument("--pretrained_model", type=str, default=PRETRAINED_MODEL_NAME,
                        help="Name of pretrained model to use (for transformer)")
    parser.add_argument("--freeze_base", action="store_true",
                        help="Whether to freeze the base model parameters")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Get training device
    device = get_training_device()

    # Create dataset loaders
    train_loader, val_loader, test_loader, tokenizer = create_data_loaders()

    # Create model
    if args.model_type == "transformer":
        model = get_model(
            model_type="transformer",
            pretrained_model_name=args.pretrained_model,
            num_classes=2,
            freeze_base=args.freeze_base
        )
    else:
        # For LSTM model, we need vocabulary size which depends on the tokenizer
        vocab_size = len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else tokenizer.vocab_size
        model = get_model(
            model_type="lstm",
            vocab_size=vocab_size,
            embedding_dim=300,
            hidden_dim=256,
            num_layers=2,
            num_classes=2,
        )

    # Move model to device
    model.to(device)

    # Log model information
    num_params = count_model_parameters(model)
    logger.info(f"Model: {args.model_type}")
    logger.info(f"Number of trainable parameters: {num_params:,}")

    # Create loss function
    loss_fn = get_loss_function(
        loss_type=args.loss_type,
        length_weight_factor=0.2,
        confidence_penalty_factor=0.1,
        class_weights=CLASS_WEIGHTS
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # INSERT CHECKPOINT LOADING CODE HERE - before calling train_model
    if args.resume:
        checkpoint_dir = os.path.dirname(MODEL_SAVE_PATH + "/sentiment_model")
        logger.info(f"Looking for checkpoints in: {checkpoint_dir}")

        if not os.path.exists(checkpoint_dir):
            logger.warning(f"Checkpoint directory {checkpoint_dir} does not exist. Starting from scratch.")
        else:
            # List all files in the directory to verify
            all_files = os.listdir(checkpoint_dir)
            logger.info(f"Files in checkpoint directory: {all_files}")

            # Filter for checkpoint files
            checkpoints = [f for f in all_files if f.endswith('.pt') and f.startswith('checkpoint_epoch_')]
            logger.info(f"Found checkpoint files: {checkpoints}")

            if checkpoints:
                # Find the latest checkpoint by epoch number
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0]))
                checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

                logger.info(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']

                logger.info(f"Resuming training from epoch {start_epoch + 1}")

                # Update args.epochs to run the remaining epochs
                args.epochs = max(args.epochs, start_epoch + 1)
            else:
                logger.info("No checkpoints found. Starting from scratch.")

    # Train the model
    logger.info("Starting training...")
    training_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        device=device,
        num_epochs=args.epochs,
        start_epoch=(start_epoch + 1) if args.resume and 'start_epoch' in locals() else 0
    )

    # Evaluate on tests set
    logger.info("Evaluating on tests set...")
    metrics = detailed_evaluation(model, test_loader, device, class_names=["Negative", "Positive"])

    # Save evaluation metrics
    save_metrics(metrics, "test_metrics.json")

    logger.info("Training and evaluation complete!")


if __name__ == "__main__":
    main()