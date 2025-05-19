"""
Visualization utilities for sentiment analysis model.
"""

import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

logger = logging.getLogger(__name__)


def create_output_dir(base_dir="visualizations"):
    """Create output directory for visualizations with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def visualize_training_history(history, output_dir=None):
    """
    Visualize training metrics history.

    Args:
        history (dict): Dictionary containing training metrics history
        output_dir (Path, optional): Directory to save visualization

    Returns:
        str: Path to the saved visualization
    """
    if output_dir is None:
        output_dir = create_output_dir()

    plt.figure(figsize=(12, 10))

    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot F1 Score
    if 'val_f1' in history:
        plt.subplot(2, 2, 3)
        plt.plot(history['val_f1'], label='Validation F1 Score')
        if 'train_f1' in history:
            plt.plot(history['train_f1'], label='Train F1 Score')
        plt.title('F1 Score')
        plt.ylabel('F1 Score')
        plt.xlabel('Epoch')
        plt.legend()

    # Plot learning rate if available
    if 'learning_rate' in history:
        plt.subplot(2, 2, 4)
        plt.plot(history['learning_rate'])
        plt.title('Learning Rate')
        plt.ylabel('LR')
        plt.xlabel('Step')

    plt.tight_layout()

    # Save the figure
    output_path = output_dir / 'training_history.png'
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Training history visualization saved to {output_path}")
    return str(output_path)


def visualize_confusion_matrix(y_true, y_pred, class_names=None, output_dir=None):
    """
    Visualize confusion matrix.

    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        class_names (list, optional): Names of the classes
        output_dir (Path, optional): Directory to save visualization

    Returns:
        str: Path to the saved visualization
    """
    if output_dir is None:
        output_dir = create_output_dir()

    if class_names is None:
        class_names = ["Negative", "Positive"]

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Save the figure
    output_path = output_dir / 'confusion_matrix.png'
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Confusion matrix visualization saved to {output_path}")
    return str(output_path)


def visualize_roc_curve(y_true, y_prob, output_dir=None):
    """
    Visualize ROC curve.

    Args:
        y_true (array-like): True labels
        y_prob (array-like): Predicted probabilities for the positive class
        output_dir (Path, optional): Directory to save visualization

    Returns:
        str: Path to the saved visualization
    """
    if output_dir is None:
        output_dir = create_output_dir()

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Save the figure
    output_path = output_dir / 'roc_curve.png'
    plt.savefig(output_path)
    plt.close()

    logger.info(f"ROC curve visualization saved to {output_path}")
    return str(output_path)


def visualize_precision_recall_curve(y_true, y_prob, output_dir=None):
    """
    Visualize precision-recall curve.

    Args:
        y_true (array-like): True labels
        y_prob (array-like): Predicted probabilities for the positive class
        output_dir (Path, optional): Directory to save visualization

    Returns:
        str: Path to the saved visualization
    """
    if output_dir is None:
        output_dir = create_output_dir()

    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    # Save the figure
    output_path = output_dir / 'precision_recall_curve.png'
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Precision-recall curve visualization saved to {output_path}")
    return str(output_path)


def visualize_feature_embeddings(embeddings, labels, class_names=None, output_dir=None):
    """
    Visualize feature embeddings using t-SNE.

    Args:
        embeddings (array-like): Feature embeddings (n_samples, n_features)
        labels (array-like): Labels for each sample
        class_names (list, optional): Names of the classes
        output_dir (Path, optional): Directory to save visualization

    Returns:
        str: Path to the saved visualization
    """
    if output_dir is None:
        output_dir = create_output_dir()

    if class_names is None:
        class_names = ["Negative", "Positive"]

    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'label': [class_names[l] for l in labels]
    })

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x='x',
        y='y',
        hue='label',
        palette='viridis',
        alpha=0.7
    )
    plt.title('t-SNE Visualization of Feature Embeddings')
    plt.legend(title='Class')

    # Save the figure
    output_path = output_dir / 'feature_embeddings.png'
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Feature embeddings visualization saved to {output_path}")
    return str(output_path)


def visualize_attention_weights(attention_weights, tokens, output_dir=None):
    """
    Visualize attention weights for transformer-based models.

    Args:
        attention_weights (tensor): Attention weights (n_heads, seq_len, seq_len)
        tokens (list): List of tokens corresponding to the sequence
        output_dir (Path, optional): Directory to save visualization

    Returns:
        str: Path to the saved visualization
    """
    if output_dir is None:
        output_dir = create_output_dir()

    n_heads = attention_weights.shape[0]

    # Create subplots for each attention head
    fig, axes = plt.subplots(
        nrows=int(np.ceil(n_heads / 2)),
        ncols=2,
        figsize=(16, n_heads * 3)
    )
    axes = axes.flatten()

    for i in range(n_heads):
        # Extract attention weights for this head
        head_weights = attention_weights[i].numpy()

        # Plot heatmap
        im = axes[i].imshow(head_weights, cmap='viridis')

        # Set labels
        axes[i].set_xticks(range(len(tokens)))
        axes[i].set_yticks(range(len(tokens)))
        axes[i].set_xticklabels(tokens, rotation=90, fontsize=8)
        axes[i].set_yticklabels(tokens, fontsize=8)

        # Add colorbar
        plt.colorbar(im, ax=axes[i])

        # Set title
        axes[i].set_title(f'Attention Head {i + 1}')

    plt.tight_layout()

    # Save the figure
    output_path = output_dir / 'attention_weights.png'
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Attention weights visualization saved to {output_path}")
    return str(output_path)


def visualize_loss_comparison(loss_types, train_metrics, val_metrics, output_dir=None):
    """
    Visualize comparison of different loss functions.

    Args:
        loss_types (list): List of loss function names
        train_metrics (dict): Dictionary with loss names as keys and lists of training metrics as values
        val_metrics (dict): Dictionary with loss names as keys and lists of validation metrics as values
        output_dir (Path, optional): Directory to save visualization

    Returns:
        str: Path to the saved visualization
    """
    if output_dir is None:
        output_dir = create_output_dir()

    metrics = ['loss', 'accuracy', 'f1_score']

    plt.figure(figsize=(15, 12))

    for i, metric in enumerate(metrics):
        plt.subplot(3, 1, i + 1)

        for loss_type in loss_types:
            # Plot training metrics
            if f'{loss_type}_train_{metric}' in train_metrics:
                plt.plot(
                    train_metrics[f'{loss_type}_train_{metric}'],
                    linestyle='-',
                    label=f'{loss_type} (Train)'
                )

            # Plot validation metrics
            if f'{loss_type}_val_{metric}' in val_metrics:
                plt.plot(
                    val_metrics[f'{loss_type}_val_{metric}'],
                    linestyle='--',
                    label=f'{loss_type} (Val)'
                )

        plt.title(f'{metric.capitalize()} Comparison')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:  # Only create legend if there are labeled elements
            plt.legend()

    plt.tight_layout()

    # Save the figure
    output_path = output_dir / 'loss_comparison.png'
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Loss comparison visualization saved to {output_path}")
    return str(output_path)


def visualize_error_cases(texts, true_labels, pred_labels, probabilities, n_samples=5, output_dir=None):
    """
    Visualize error cases with highest confidence.

    Args:
        texts (list): List of input texts
        true_labels (array-like): True labels
        pred_labels (array-like): Predicted labels
        probabilities (array-like): Predicted probabilities
        n_samples (int): Number of samples to visualize
        output_dir (Path, optional): Directory to save visualization

    Returns:
        str: Path to the saved visualization
    """
    if output_dir is None:
        output_dir = create_output_dir()

    # Find error cases
    errors = []
    for i, (true, pred, prob) in enumerate(zip(true_labels, pred_labels, probabilities)):
        if true != pred:
            confidence = prob[pred]
            errors.append({
                'index': i,
                'text': texts[i],
                'true_label': true,
                'pred_label': pred,
                'confidence': confidence
            })

    # Sort by confidence (highest first)
    errors = sorted(errors, key=lambda x: x['confidence'], reverse=True)

    # Take top n_samples
    top_errors = errors[:n_samples]

    # Create DataFrame for visualization
    df = pd.DataFrame(top_errors)

    # Save to CSV
    output_path = output_dir / 'high_confidence_errors.csv'
    df.to_csv(output_path, index=False)

    logger.info(f"Error cases saved to {output_path}")
    return str(output_path)


def get_all_visualizations(results, output_dir=None):
    """
    Generate all visualizations for model results.

    Args:
        results (dict): Dictionary containing model results
        output_dir (Path, optional): Directory to save visualizations

    Returns:
        dict: Paths to all generated visualizations
    """
    if output_dir is None:
        output_dir = create_output_dir()

    # Extract dataset from results
    y_true = np.array(results['true_labels'])
    y_pred = np.array(results['pred_labels'])
    y_prob = np.array(results['probabilities'])

    # Generate visualizations
    paths = {}

    # 1. Training history
    if 'history' in results:
        paths['training_history'] = visualize_training_history(results['history'], output_dir)

    # 2. Confusion matrix
    paths['confusion_matrix'] = visualize_confusion_matrix(y_true, y_pred, output_dir=output_dir)

    # 3. ROC curve
    if y_prob.shape[1] > 1:  # If we have class probabilities
        paths['roc_curve'] = visualize_roc_curve(y_true, y_prob[:, 1], output_dir=output_dir)

    # 4. Precision-recall curve
    if y_prob.shape[1] > 1:
        paths['precision_recall_curve'] = visualize_precision_recall_curve(
            y_true, y_prob[:, 1], output_dir=output_dir
        )

    # 5. Feature embeddings
    if 'embeddings' in results:
        paths['feature_embeddings'] = visualize_feature_embeddings(
            results['embeddings'], y_true, output_dir=output_dir
        )

    # 6. Attention weights
    if 'attention_weights' in results and 'tokens' in results:
        paths['attention_weights'] = visualize_attention_weights(
            results['attention_weights'], results['tokens'], output_dir=output_dir
        )

    # 7. Error cases
    if 'texts' in results:
        paths['error_cases'] = visualize_error_cases(
            results['texts'], y_true, y_pred, y_prob, output_dir=output_dir
        )

    # 8. Loss comparison
    if 'loss_comparison' in results:
        paths['loss_comparison'] = visualize_loss_comparison(
            results['loss_comparison']['loss_types'],
            results['loss_comparison']['train_metrics'],
            results['loss_comparison']['val_metrics'],
            output_dir=output_dir
        )

    return paths