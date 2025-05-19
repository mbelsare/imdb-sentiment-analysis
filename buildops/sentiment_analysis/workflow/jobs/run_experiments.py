"""
Script to run experiments comparing different loss functions.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from buildops.sentiment_analysis.data.data_loader import create_data_loaders
from buildops.sentiment_analysis.mlobjects.models.sentiment_analysis_model import get_model
from buildops.sentiment_analysis.mlobjects.common.loss_function import get_loss_function
from buildops.sentiment_analysis.mlobjects.utils.visualization_utils import (
    visualize_training_history,
    visualize_confusion_matrix,
    visualize_roc_curve,
    visualize_loss_comparison
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare different loss functions")

    parser.add_argument("--model_type", type=str, default="transformer",
                        choices=["transformer", "lstm"],
                        help="Type of model to use")
    parser.add_argument("--pretrained_model", type=str, default="bert-base-uncased",
                        help="Pretrained model name (for transformer)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--loss_types", type=str, nargs="+",
                        default=["standard", "length_aware", "confidence_penalty", "weighted", "combined"],
                        help="Loss functions to compare")
    parser.add_argument("--output_dir", type=str, default="experiment_results",
                        help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--use_existing_model", action="store_true",
                        help="Use existing model instead of training from scratch")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to existing model checkpoint")

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, optimizer, loss_fn, device, sample_weights=True):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    try:
        for batch in tqdm(dataloader, desc="Training"):
            # Move data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Get labels and ensure correct shape
            labels = batch["label"].to(device)

            # Fix label shape: [16] -> [16, 1]
            if len(labels.shape) == 1:
                labels = labels.float().unsqueeze(1)  # Shape: [batch_size, 1]

            # Fix label shape: [16, 1, 1] -> [16, 1] (remove extra dimension)
            if len(labels.shape) == 3:
                labels = labels.squeeze(1)  # Remove middle dimension

            # Forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # Handle model output shape mismatch
            if logits.shape[1] == 2 and labels.shape[1] == 1:
                # Extract positive class logit to match labels shape
                logits = logits[:, 1:2]  # Use slicing to keep dim: [batch_size, 1]

            # Prepare sample weights if using custom loss
            weights = None
            if sample_weights and "review_length" in batch:
                weights = {"review_lengths": batch["review_length"].to(device)}

            # Compute loss with improved error handling
            try:
                if weights is not None and hasattr(loss_fn, 'forward') and 'sample_weights' in loss_fn.forward.__code__.co_varnames:
                    loss = loss_fn(logits, labels, weights)
                else:
                    loss = loss_fn(logits, labels)
            except Exception as e:
                print(f"Error in loss calculation: {e}")
                print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")
                if weights:
                    print(f"Weights keys: {weights.keys()}")
                    for k, v in weights.items():
                        print(f"  {k} shape: {v.shape}")
                raise

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()

            # Get predictions (adjust based on model output shape)
            if logits.shape[1] == 2:
                # For 2-class output, get class with highest probability
                preds = torch.argmax(logits, dim=1).cpu().numpy()
            else:
                # For binary output with BCE loss
                preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()

            # Ensure labels are in right format for metrics
            batch_labels = labels.view(-1).cpu().numpy() if len(labels.shape) > 1 else labels.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(batch_labels)

        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return avg_loss, accuracy, f1

    except Exception as e:
        print(f"Error in train_epoch: {e}")
        import traceback
        traceback.print_exc()
        # Return default values for error case
        return 0.0, 0.0, 0.0

def evaluate(model, dataloader, loss_fn, device, sample_weights=True):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move dataset to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device).float().unsqueeze(1)  # Shape: [batch_size, 1]

            # Forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # Prepare sample weights if using custom loss
            weights = None
            if sample_weights and "review_length" in batch:
                weights = {"review_lengths": batch["review_length"].to(device)}

            # Compute loss
            if weights is not None:
                loss = loss_fn(logits, labels, weights)
            else:
                loss = loss_fn(logits, labels)

            # Track metrics
            total_loss += loss.item()

            # Get predictions and probabilities
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int().cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    }

    return metrics, np.array(all_labels), np.array(all_preds), np.array(all_probs)


def run_experiment(args):
    """Run experiment comparing different loss functions."""
    # Set random seed
    set_seed(args.seed)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to {output_dir}")

    # Create dataset loaders
    train_loader, val_loader, test_loader, tokenizer = create_data_loaders(
        batch_size=args.batch_size
    )

    # Dictionary to store results
    results = {
        'train_history': {},
        'val_metrics': {},
        'test_metrics': {}
    }

    # Dictionary to store training and validation metrics for visualization
    train_metrics = {}
    val_metrics = {}

    # Train and evaluate with each loss function
    for loss_type in args.loss_types:
        logger.info(f"\n{'=' * 50}\nTraining with loss function: {loss_type}\n{'=' * 50}")

        # Create model
        if args.model_type == "transformer":
            model = get_model(
                model_type="transformer",
                pretrained_model_name=args.pretrained_model,
                num_classes=2,  # Binary classification
            )
        else:
            # For LSTM model
            vocab_size = len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else tokenizer.vocab_size
            model = get_model(
                model_type="lstm",
                vocab_size=vocab_size,
                embedding_dim=300,
                hidden_dim=256,
                num_layers=2,
                num_classes=2,  # Binary classification
            )

        model.to(device)

        # Instead of training from scratch, load your existing model
        if args.use_existing_model:
            checkpoint_path = args.model_path
            logger.info(f"Loading model from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])

            # Skip training and go straight to evaluation
            logger.info("Skipping training, using existing model")
        else:

            # Create loss function
            loss_fn = get_loss_function(loss_type)

            # Create optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

            # Training history
            history = {
                'train_loss': [],
                'train_accuracy': [],
                'train_f1': [],
                'val_loss': [],
                'val_accuracy': [],
                'val_f1': []
            }

            # Training loop
            for epoch in range(args.epochs):
                logger.info(f"Epoch {epoch + 1}/{args.epochs}")

                # Train
                train_loss, train_accuracy, train_f1 = train_epoch(
                    model, train_loader, optimizer, loss_fn, device,
                    sample_weights=(loss_type != "standard")
                )

                # Evaluate on validation set
                val_metrics_epoch, _, _, _ = evaluate(
                    model, val_loader, loss_fn, device,
                    sample_weights=(loss_type != "standard")
                )

                # Update history
                history['train_loss'].append(train_loss)
                history['train_accuracy'].append(train_accuracy)
                history['train_f1'].append(train_f1)
                history['val_loss'].append(val_metrics_epoch['loss'])
                history['val_accuracy'].append(val_metrics_epoch['accuracy'])
                history['val_f1'].append(val_metrics_epoch['f1_score'])

                # Log progress
                logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
                logger.info(f"Val Loss: {val_metrics_epoch['loss']:.4f}, Val Accuracy: {val_metrics_epoch['accuracy']:.4f}")

            # Store training history
            results['train_history'][loss_type] = history

            # Evaluate on tests set
            test_metrics, test_labels, test_preds, test_probs = evaluate(
                model, test_loader, loss_fn, device,
                sample_weights=(loss_type != "standard")
            )

            # Store tests metrics
            results['test_metrics'][loss_type] = {
                'metrics': test_metrics,
                'true_labels': test_labels.tolist(),
                'pred_labels': test_preds.tolist(),
                'probabilities': test_probs.tolist()
            }

            logger.info(f"Test Metrics for {loss_type} loss:")
            for metric, value in test_metrics.items():
                logger.info(f"{metric}: {value:.4f}")

            # Save model
            model_path = output_dir / f"model_{loss_type}.pt"
            torch.save(model.state_dict(), model_path)

            # Collect metrics for visualization
            for metric in ['loss', 'accuracy', 'f1_score']:
                train_metrics[f'{loss_type}_train_{metric}'] = history[f'train_{metric}']
                val_metrics[f'{loss_type}_val_{metric}'] = history[f'val_{metric}']

            # Visualize confusion matrix
            visualize_confusion_matrix(
                test_labels,
                test_preds,
                output_dir=output_dir / loss_type
            )

            # Visualize ROC curve
            visualize_roc_curve(
                test_labels,
                test_probs[:, 0],
                output_dir=output_dir / loss_type
            )

            # Visualize training history
            visualize_training_history(
                history,
                output_dir=output_dir / loss_type
            )

    # Compare loss functions
    loss_comparison_data = {
        'loss_types': args.loss_types,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics
    }

    # Visualize loss comparison
    visualize_loss_comparison(
        args.loss_types,
        train_metrics,
        val_metrics,
        output_dir=output_dir
    )

    # Create summary table
    summary = []
    for loss_type in args.loss_types:
        if loss_type in results['test_metrics']:  # Check if this loss type has metrics
            metrics = results['test_metrics'][loss_type]['metrics']
            summary.append({
                'Loss Function': loss_type,
                'Accuracy': metrics['accuracy'],
                'F1 Score': metrics['f1_score'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall']
            })
        else:
            print(f"Skipping {loss_type} - no test metrics available")

    # Create summary table and visualization if we have data
    if summary:
        try:
            # Create summary DataFrame
            summary_df = pd.DataFrame(summary)
            summary_path = output_dir / "summary.csv"
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"Summary saved to {summary_path}")

            # Log the DataFrame contents for debugging
            logger.info(f"Summary DataFrame columns: {summary_df.columns.tolist()}")
            logger.info(f"Summary DataFrame shape: {summary_df.shape}")

            # Check if DataFrame has the required columns before creating charts
            required_columns = ['Loss Function'] + ['Accuracy', 'F1 Score', 'Precision', 'Recall']
            missing_columns = [col for col in required_columns if col not in summary_df.columns]

            if not missing_columns and len(summary_df) > 0:
                # Create summary charts
                plt.figure(figsize=(12, 8))
                metrics_to_plot = ['Accuracy', 'F1 Score', 'Precision', 'Recall']

                for i, metric in enumerate(metrics_to_plot):
                    plt.subplot(2, 2, i + 1)
                    sns.barplot(data=summary_df, x='Loss Function', y=metric)
                    plt.title(f'{metric} by Loss Function')
                    plt.xticks(rotation=45)

                plt.tight_layout()
                plt.savefig(output_dir / "summary_charts.png")
                plt.close()
                logger.info("Summary charts created successfully")
            else:
                if missing_columns:
                    logger.warning(f"Cannot create summary charts - missing columns: {missing_columns}")
                else:
                    logger.warning("Cannot create summary charts - summary DataFrame is empty")
        except Exception as e:
            logger.error(f"Error creating summary visualizations: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    else:
        logger.warning("No summary created - no completed evaluation runs")

    logger.info(f"Experiment completed. Results saved to {output_dir}")
    return results, output_dir


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)