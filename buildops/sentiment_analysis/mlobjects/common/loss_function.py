"""
Custom loss functions for sentiment analysis model.
"""

import torch
import torch.nn as nn


class SentimentWeightedLoss(nn.Module):
    """
    Custom loss function for sentiment analysis that weights samples differently
    based on specific criteria.

    This implementation includes three weighting mechanisms:
    1. Review length weighting - Gives different importance to reviews based on their length
    2. Confidence penalty - Penalizes highly confident incorrect predictions
    3. Class weighting - Handles potential class imbalance
    """
    def __init__(self, length_weight_factor=0.2, confidence_penalty_factor=0.1, class_weights=None):
        """
        Initialize the custom loss function.

        Args:
            length_weight_factor (float): Controls how much review length affects the loss weighting
                                         (0 = no effect, higher = stronger effect)
            confidence_penalty_factor (float): Controls penalty for overconfident wrong predictions
                                             (0 = no penalty, higher = stronger penalty)
            class_weights (list, optional): Weights for each class [neg_weight, pos_weight]
        """
        super(SentimentWeightedLoss, self).__init__()
        self.length_weight_factor = length_weight_factor
        self.confidence_penalty_factor = confidence_penalty_factor

        # Initialize with BCEWithLogitsLoss as requested in starter code
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.base_loss = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([class_weights[1] / class_weights[0]]),
                reduction='none'
            )
        else:
            self.class_weights = None
            self.base_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, predictions, targets, sample_weights=None):
        """
        Calculate the weighted loss.

        Args:
            predictions (torch.Tensor): Model predictions (batch_size, 1)
            targets (torch.Tensor): True labels (batch_size, 1)
            sample_weights (dict, optional): Dictionary containing additional weighting factors:
                - 'review_lengths': Normalized review lengths (batch_size)

        Returns:
            torch.Tensor: Weighted loss
        """
        # Ensure predictions and targets have compatible shapes
        if predictions.shape != targets.shape:
            if len(predictions.shape) == 2 and predictions.shape[1] == 2 and len(targets.shape) == 1:
                # If predictions are [batch_size, 2] and targets are [batch_size]
                # Extract the positive class logit or reshape targets
                predictions = predictions[:, 1].view(-1, 1)  # Extract positive class logit
            elif len(predictions.shape) == 2 and predictions.shape[1] == 1 and len(targets.shape) == 1:
                # If predictions are [batch_size, 1] and targets are [batch_size]
                targets = targets.view(-1, 1)  # Reshape targets

        # Calculate base loss
        base_loss = self.base_loss(predictions, targets)

        # Initialize combined weight tensor (defaults to 1.0 for all samples)
        batch_size = predictions.size(0)
        weights = torch.ones(batch_size, device=predictions.device)

        # Apply review length weighting if provided
        if sample_weights is not None and 'review_lengths' in sample_weights:
            review_lengths = sample_weights['review_lengths']
            # Map lengths to weights in range [0.8, 1.2] to avoid extreme weighting
            length_weights = 1.0 + (review_lengths - 0.5) * self.length_weight_factor
            weights = weights * length_weights

        # Apply confidence penalty for incorrect predictions
        if self.confidence_penalty_factor > 0:
            # Convert logits to probabilities
            probs = torch.sigmoid(predictions)

            # Determine correctness of predictions (1 if correct, 0 if wrong)
            pred_classes = (probs > 0.5).float()
            correct_predictions = (pred_classes == targets).float()

            # Calculate confidence as distance from decision boundary (0.5)
            confidence = torch.abs(probs - 0.5) * 2  # Scale to [0, 1]

            # Confidence penalty applies only to incorrect predictions
            confidence_penalty = (1 - correct_predictions) * confidence * self.confidence_penalty_factor

            # Add penalty to weights
            weights = weights + confidence_penalty

        # Apply final weighting to base loss
        weighted_loss = base_loss * weights

        # Return mean of weighted losses
        return weighted_loss.mean()


class ReviewLengthLoss(SentimentWeightedLoss):
    """
    Loss function variant that only implements review length weighting.
    """
    def __init__(self, length_weight_factor=0.2, class_weights=None):
        super(ReviewLengthLoss, self).__init__(
            length_weight_factor=length_weight_factor,
            confidence_penalty_factor=0,  # Disable confidence penalty
            class_weights=class_weights
        )


class ConfidencePenaltyLoss(SentimentWeightedLoss):
    """
    Loss function variant that only implements confidence penalty.
    """
    def __init__(self, confidence_penalty_factor=0.1, class_weights=None):
        super(ConfidencePenaltyLoss, self).__init__(
            length_weight_factor=0,  # Disable length weighting
            confidence_penalty_factor=confidence_penalty_factor,
            class_weights=class_weights
        )


def get_loss_function(loss_type="weighted", **kwargs):
    """
    Factory function to create loss function instances.

    Args:
        loss_type (str): Type of loss function to create
                        ("weighted", "length_aware", "confidence_penalty", or "standard")
        **kwargs: Additional arguments to pass to the loss function constructor

    Returns:
        nn.Module: Loss function instance
    """
    if loss_type == "weighted":
        return SentimentWeightedLoss(**kwargs)
    elif loss_type == "length_aware":
        return ReviewLengthLoss(**kwargs)
    elif loss_type == "confidence_penalty":
        return ConfidencePenaltyLoss(**kwargs)
    elif loss_type == "combined":
        return SentimentWeightedLoss(**kwargs)
    elif loss_type == "standard":
        if "class_weights" in kwargs and kwargs["class_weights"] is not None:
            pos_weight = torch.tensor([kwargs["class_weights"][1] / kwargs["class_weights"][0]])
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")