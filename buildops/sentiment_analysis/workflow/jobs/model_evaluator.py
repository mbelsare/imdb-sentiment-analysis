"""
Evaluation utilities for sentiment analysis model.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

from buildops.sentiment_analysis.config.model_config import LOGS_DIR

logger = logging.getLogger(__name__)


def detailed_evaluation(model, test_loader, device, class_names=None):
    """
    Perform detailed evaluation of the model on tests dataset.

    Args:
        model: The model to evaluate
        test_loader: DataLoader for tests dataset
        device: Device to evaluate on ('cuda' or 'cpu')
        class_names (list): Names of the classes

    Returns:
        dict: Detailed evaluation metrics
    """
    if class_names is None:
        class_names = ["Negative", "Positive"]

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # Get predictions and probabilities
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            # Collect results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    # Calculate per-class metrics
    class_report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)

    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Calculate ROC curve and AUC (for positive class)
    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
    roc_auc = auc(fpr, tpr)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Ensure directory exists
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Save the plot
    cm_path = os.path.join(LOGS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Save the ROC plot
    roc_path = os.path.join(LOGS_DIR, "roc_curve.png")
    plt.savefig(roc_path)
    plt.close()

    # Create detailed metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm,
        'classification_report': class_report,
        'roc_auc': roc_auc,
        'plots': {
            'confusion_matrix': cm_path,
            'roc_curve': roc_path
        }
    }

    # Log results
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test F1 Score: {f1:.4f}")
    logger.info(f"Test Precision: {precision:.4f}")
    logger.info(f"Test Recall: {recall:.4f}")
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Classification Report:\n{pd.DataFrame(class_report).T}")

    return metrics


def analyze_prediction_errors(model, test_loader, device, tokenizer, top_n=10):
    """
    Analyze common prediction errors.

    Args:
        model: The model to evaluate
        test_loader: DataLoader for tests dataset
        device: Device to evaluate on ('cuda' or 'cpu')
        tokenizer: Tokenizer used for the model
        top_n: Number of top errors to analyze

    Returns:
        dict: Error analysis
    """
    model.eval()
    errors = []

    with torch.no_grad():
        for batch in test_loader:
            # Get original texts from input_ids (if available in the dataset)
            if hasattr(batch, "text"):
                texts = batch["text"]
            else:
                # Decode from input_ids
                input_ids = batch["input_ids"]
                texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]

            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            # Find errors
            for i, (pred, label) in enumerate(zip(preds, labels)):
                if pred != label:
                    confidence = probs[i, pred.item()].item()
                    errors.append({
                        'text': texts[i][:100] + "..." if len(texts[i]) > 100 else texts[i],
                        'true_label': label.item(),
                        'pred_label': pred.item(),
                        'confidence': confidence,
                        'error_margin': confidence - probs[i, label.item()].item()
                    })

    # Sort errors by confidence (most confident errors first)
    errors = sorted(errors, key=lambda x: x['confidence'], reverse=True)

    # Limit to top_n
    top_errors = errors[:top_n]

    # Log results
    logger.info(f"Total prediction errors: {len(errors)}")
    logger.info(f"Top {len(top_errors)} confident errors:")
    for i, err in enumerate(top_errors):
        logger.info(f"{i + 1}. Text: {err['text']}")
        logger.info(f"   True: {err['true_label']}, Pred: {err['pred_label']}, Confidence: {err['confidence']:.4f}")

    return {
        'total_errors': len(errors),
        'top_errors': top_errors
    }


def predict_sentiment(model, tokenizer, text, device):
    """
    Predict sentiment for a given text.

    Args:
        model: The trained model
        tokenizer: Tokenizer for the model
        text (str): Input text
        device: Device to run prediction on

    Returns:
        dict: Prediction results
    """
    model.eval()

    # Tokenize the text
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

    # Move to device
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Make prediction
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(logits, dim=1).item()

    # Get probabilities
    class_probs = probs[0].cpu().numpy()

    result = {
        'text': text,
        'sentiment': 'Positive' if pred_class == 1 else 'Negative',
        'confidence': class_probs[pred_class],
        'probabilities': {
            'Negative': class_probs[0],
            'Positive': class_probs[1]
        }
    }

    return result


def visualize_attention(model, tokenizer, text, device, layer=-1, head=0, output_dir=None):
    """
    Visualize attention patterns for a given text.

    Args:
        model: The model
        tokenizer: The tokenizer
        text (str): Input text
        device: Device to run on
        layer (int): Which transformer layer to visualize (-1 for last layer)
        head (int): Which attention head to visualize
        output_dir (Path, optional): Directory to save visualization

    Returns:
        str: Path to the saved attention visualization
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"visualizations/attention_{timestamp}")
        output_dir.mkdir(parents=True, exist_ok=True)
    elif isinstance(output_dir, str):
        output_dir = Path(output_dir)

    # Only works for transformer models
    if not hasattr(model, 'transformer') and not hasattr(model, 'get_attention_weights'):
        logger.warning("Attention visualization is only supported for transformer models")
        return None

    # Tokenize text
    encoded_input = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    # Get attention weights
    with torch.no_grad():
        # Check if model has a method to get attention weights
        if hasattr(model, 'get_attention_weights'):
            attention_weights = model.get_attention_weights(
                input_ids=encoded_input['input_ids'],
                attention_mask=encoded_input['attention_mask']
            )
        else:
            # Generic approach for Hugging Face transformers
            outputs = model.transformer(
                input_ids=encoded_input['input_ids'],
                attention_mask=encoded_input['attention_mask'],
                output_attentions=True
            )
            attention_weights = outputs.attentions

        # Get attention from specific layer and head
        if layer < 0:
            layer = len(attention_weights) + layer
        attention = attention_weights[layer][0, head].cpu().numpy()

    # Decode tokens
    tokens = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])

    # Create visualization - explicitly use plt.subplots to get a Figure and Axes object
    fig, ax = plt.subplots(figsize=(10, 8))

    # Display matrix - explicitly use the imshow method of the Axes object
    im = ax.imshow(attention, cmap='viridis')

    # Set labels
    tick_positions = list(range(len(tokens)))

    # Limit the number of tick labels to avoid overcrowding
    max_tokens_to_show = 30
    if len(tokens) > max_tokens_to_show:
        # Show a subset of tokens with ellipsis in between
        visible_indices = list(range(10)) + list(range(len(tokens) - 10, len(tokens)))
        visible_tokens = [tokens[i] if i in visible_indices else '...' for i in range(len(tokens))]
        # Remove duplicated ellipsis
        visible_tokens = [t for i, t in enumerate(visible_tokens) if
                          t != '...' or (i > 0 and visible_tokens[i - 1] != '...')]

        # Updated tick positions and labels
        tick_positions = [i for i, t in enumerate(visible_tokens) if t != '...']
        tick_labels = [t for t in visible_tokens if t != '...']

        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=8)
        ax.set_yticklabels(tick_labels, fontsize=8)
    else:
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tokens, rotation=90, fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)

    # Add colorbar
    fig.colorbar(im)

    # Set title
    ax.set_title(f"Attention visualization (Layer {layer + 1}, Head {head + 1})")
    plt.tight_layout()

    # Save figure
    output_path = output_dir / f"attention_layer{layer + 1}_head{head + 1}.png"
    plt.savefig(output_path)
    plt.close(fig)

    logger.info(f"Attention visualization saved to {output_path}")

    # Create a heatmap of all heads in the selected layer
    if hasattr(attention_weights[layer], 'shape') and len(attention_weights[layer].shape) == 4:
        # Get number of heads
        num_heads = attention_weights[layer].shape[1]

        # Create a grid of heatmaps
        fig, axes = plt.subplots(
            nrows=int(np.ceil(num_heads / 4)),
            ncols=min(num_heads, 4),
            figsize=(16, 3 * int(np.ceil(num_heads / 4))),
            squeeze=False
        )

        for h in range(num_heads):
            row, col = h // 4, h % 4
            ax = axes[row, col]

            # Get attention weights for this head
            head_attention = attention_weights[layer][0, h].cpu().numpy()

            # Plot heatmap
            im = ax.imshow(head_attention, cmap='viridis')
            ax.set_title(f"Head {h + 1}")

            # Minimal axis ticks to avoid clutter
            if len(tokens) > max_tokens_to_show:
                positions = [0, len(tokens) // 2, len(tokens) - 1]
                ax.set_xticks(positions)
                ax.set_yticks(positions)
                ax.set_xticklabels([tokens[p] for p in positions], rotation=90, fontsize=7)
                ax.set_yticklabels([tokens[p] for p in positions], fontsize=7)
            else:
                ax.set_xticks(range(len(tokens)))
                ax.set_yticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=90, fontsize=7)
                ax.set_yticklabels(tokens, fontsize=7)

        # Hide unused subplots if any
        for h in range(num_heads, axes.shape[0] * axes.shape[1]):
            row, col = h // 4, h % 4
            axes[row, col].axis('off')

        plt.tight_layout()

        # Save all-heads visualization
        all_heads_path = output_dir / f"attention_layer{layer + 1}_all_heads.png"
        plt.savefig(all_heads_path)
        plt.close(fig)

        logger.info(f"All heads attention visualization saved to {all_heads_path}")

    return str(output_path)