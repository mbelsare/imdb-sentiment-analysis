"""
Training utilities for sentiment analysis model.
"""

import logging
import os
import time

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from buildops.sentiment_analysis.config.model_config import (
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    WARMUP_PROPORTION,
    SAVE_STEPS,
    MODEL_SAVE_PATH,
    LOGS_DIR,
    USE_CUDA,
    CUDA_DEVICE
)

logger = logging.getLogger(__name__)


def train_model(model, train_loader, val_loader, loss_fn, device, num_epochs=NUM_EPOCHS, start_epoch=0):
    """
    Train the sentiment analysis model.

    Args:
        model: The model to train
        train_loader: DataLoader for training dataset
        val_loader: DataLoader for validation dataset
        loss_fn: Loss function to use
        device: Device to train on ('cuda' or 'cpu')
        num_epochs: Number of training epochs
        start_epoch: Starting epoch

    Returns:
        dict: Training history (losses, metrics)
    """
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Setup learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * WARMUP_PROPORTION)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Initialize history dictionary to track metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'val_f1': []
    }

    # Create directory for model checkpoints if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # Create log directory if it doesn't exist
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Track best model performance
    best_val_f1 = 0.0

    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs...")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_train_loss = 0
        epoch_train_preds = []
        epoch_train_labels = []

        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")

        for batch_idx, batch in enumerate(train_pbar):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            review_lengths = batch["review_length"].to(device)

            # Forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            print(f"Logits shape: {logits.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Logits sample: {logits[0]}")
            print(f"Labels sample: {labels[0]}")

            # Ensure shapes are compatible
            if len(logits.shape) == 2 and logits.shape[1] == 2:
                # If model outputs [batch_size, 2], extract positive class logit
                logits = logits[:, 1].view(-1, 1)

            # Make sure labels have right shape
            if len(labels.shape) == 1:
                labels = labels.float().view(-1, 1)

            # Prepare sample weights
            weights = {"review_lengths": review_lengths.to(device)} if "review_length" in batch else None

            # Compute loss with proper shape handling
            try:
                if weights is not None and hasattr(loss_fn,
                                                   'forward') and 'sample_weights' in loss_fn.forward.__code__.co_varnames:
                    loss = loss_fn(logits, labels, weights)
                else:
                    loss = loss_fn(logits, labels)
            except (ValueError, RuntimeError) as e:
                print(f"Error in loss calculation: {e}")
                print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")
                raise

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()

            # Track loss and predictions
            epoch_train_loss += loss.item()

            # Get predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            epoch_train_preds.extend(preds)
            epoch_train_labels.extend(labels.cpu().numpy())

            # Update progress bar
            train_pbar.set_postfix({'loss': loss.item()})

            # Save checkpoint periodically
            if (batch_idx + 1) % SAVE_STEPS == 0:
                checkpoint_path = os.path.join(
                    MODEL_SAVE_PATH,
                    f"checkpoint_epoch_{epoch + 1}_step_{batch_idx + 1}.pt"
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                logger.info(f"Checkpoint saved to {checkpoint_path}")

        # Calculate epoch metrics
        epoch_train_accuracy = accuracy_score(epoch_train_labels, epoch_train_preds)
        epoch_train_loss /= len(train_loader)

        history['train_loss'].append(epoch_train_loss)
        history['train_accuracy'].append(epoch_train_accuracy)

        # Validation
        val_loss, val_metrics = evaluate_model(model, val_loader, loss_fn, device)

        # Record validation metrics
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])

        # Log epoch results
        logger.info(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"Train Loss = {epoch_train_loss:.4f}, "
            f"Train Acc = {epoch_train_accuracy:.4f}, "
            f"Val Loss = {val_loss:.4f}, "
            f"Val Acc = {val_metrics['accuracy']:.4f}, "
            f"Val F1 = {val_metrics['f1']:.4f}"
        )

        # Save the best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_path = os.path.join(MODEL_SAVE_PATH, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, best_model_path)
            logger.info(f"New best model saved with F1 score: {best_val_f1:.4f}")

    # Save the final model
    final_model_path = os.path.join(MODEL_SAVE_PATH, "final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'val_metrics': val_metrics,
        'history': history
    }, final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    # Plot training history
    plot_training_history(history)

    return history


def evaluate_model(model, data_loader, loss_fn, device):
    """
    Evaluate the model on the given data loader.

    Args:
        model: The model to evaluate
        data_loader: DataLoader for evaluation data
        loss_fn: Loss function to use
        device: Device to evaluate on ('cuda' or 'cpu')

    Returns:
        tuple: (avg_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Ensure labels have the right shape for BCEWithLogitsLoss
            if len(labels.shape) == 1:
                labels = labels.float().view(-1, 1)

            # Forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # Ensure shapes are compatible
            if len(logits.shape) == 2 and logits.shape[1] == 2:
                # If model outputs [batch_size, 2], extract positive class logit
                logits = logits[:, 1].view(-1, 1)

            # Prepare sample weights if available
            weights = None
            if "review_length" in batch:
                review_lengths = batch["review_length"].to(device)
                weights = {"review_lengths": review_lengths}

            # Calculate loss with proper shape handling
            try:
                if weights is not None and hasattr(loss_fn, 'forward') and 'sample_weights' in loss_fn.forward.__code__.co_varnames:
                    loss = loss_fn(logits, labels, weights)
                else:
                    loss = loss_fn(logits, labels)
            except (ValueError, RuntimeError) as e:
                print(f"Error in loss calculation: {e}")
                print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")
                raise

            total_loss += loss.item()

            # Get predictions (apply sigmoid for binary classification)
            preds = torch.sigmoid(logits) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    avg_loss = total_loss / len(data_loader)

    metrics = {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

    return avg_loss, metrics


def plot_training_history(history):
    """
    Plot the training history metrics.

    Args:
        history (dict): Dictionary containing training history
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.figure(figsize=(12, 8))

    # Plot loss
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    # Plot accuracy
    plt.subplot(2, 1, 2)
    plt.plot(history['train_accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.plot(history['val_f1'], label='Validation F1 Score')
    plt.title('Model Metrics')
    plt.ylabel('Score')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')

    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(LOGS_DIR, f"training_history_{timestamp}.png")
    plt.savefig(plot_path)
    plt.close()

    logger.info(f"Training history plot saved to {plot_path}")


def get_training_device():
    """
    Determine the device to use for training.

    Returns:
        torch.device: Device to use
    """
    if USE_CUDA and torch.cuda.is_available():
        device = torch.device(f"cuda:{CUDA_DEVICE}")
        logger.info(f"Using CUDA device: {device}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    return device