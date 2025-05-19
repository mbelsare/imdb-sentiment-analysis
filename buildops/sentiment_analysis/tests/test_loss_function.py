"""
Unit tests for custom loss functions.
"""

import unittest
import torch
import numpy as np
from buildops.sentiment_analysis.mlobjects.common.loss_function import (
    SentimentWeightedLoss,
    ReviewLengthLoss,
    ConfidencePenaltyLoss,
    get_loss_function
)


class TestSentimentWeightedLoss(unittest.TestCase):
    """Test cases for the SentimentWeightedLoss class."""

    def setUp(self):
        """Set up tests fixtures."""
        # Create sample dataset
        self.predictions = torch.tensor([0.8, -0.5, 0.2, -0.7], dtype=torch.float32).reshape(-1, 1)
        self.targets = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.float32).reshape(-1, 1)

        # Sample review lengths (normalized)
        self.review_lengths = torch.tensor([0.2, 0.5, 0.8, 0.3], dtype=torch.float32)

        # Initialize loss functions
        self.standard_loss = torch.nn.BCEWithLogitsLoss()
        self.weighted_loss = SentimentWeightedLoss(
            length_weight_factor=0.2,
            confidence_penalty_factor=0.1
        )

    def test_standard_vs_weighted_loss_shape(self):
        """Test that standard and weighted loss return tensors of same shape."""
        standard_loss_val = self.standard_loss(self.predictions, self.targets)
        weighted_loss_val = self.weighted_loss(
            self.predictions,
            self.targets,
            {'review_lengths': self.review_lengths}
        )

        self.assertEqual(standard_loss_val.shape, weighted_loss_val.shape)

    def test_length_weighting(self):
        """Test that length weighting affects the loss value."""
        # Create loss function with only length weighting
        length_only_loss = SentimentWeightedLoss(
            length_weight_factor=0.5,
            confidence_penalty_factor=0.0
        )

        # Compute loss without and with length weighting
        no_length_loss = self.standard_loss(self.predictions, self.targets)
        with_length_loss = length_only_loss(
            self.predictions,
            self.targets,
            {'review_lengths': self.review_lengths}
        )

        # Losses should be different
        self.assertNotEqual(no_length_loss.item(), with_length_loss.item())

    def test_confidence_penalty(self):
        """Test that confidence penalty affects the loss value."""
        # Create loss function with only confidence penalty
        confidence_only_loss = SentimentWeightedLoss(
            length_weight_factor=0.0,
            confidence_penalty_factor=0.5
        )

        # Compute loss without and with confidence penalty
        no_confidence_loss = self.standard_loss(self.predictions, self.targets)
        with_confidence_loss = confidence_only_loss(self.predictions, self.targets)

        # Losses should be different
        self.assertNotEqual(no_confidence_loss.item(), with_confidence_loss.item())

    def test_class_weights(self):
        """Test that class weights affect the loss value."""
        # Create loss function with class weights
        class_weighted_loss = SentimentWeightedLoss(
            length_weight_factor=0.0,
            confidence_penalty_factor=0.0,
            class_weights=[0.7, 1.3]  # Penalize positive class more
        )

        # Compute loss without and with class weights
        no_weights_loss = self.standard_loss(self.predictions, self.targets)
        with_weights_loss = class_weighted_loss(self.predictions, self.targets)

        # Losses should be different
        self.assertNotEqual(no_weights_loss.item(), with_weights_loss.item())

    def test_combined_weighting(self):
        """Test that combined weighting produces expected behavior."""
        # Create a fully combined loss
        combined_loss = SentimentWeightedLoss(
            length_weight_factor=0.2,
            confidence_penalty_factor=0.1,
            class_weights=[0.7, 1.3]
        )

        # Get loss value
        loss_val = combined_loss(
            self.predictions,
            self.targets,
            {'review_lengths': self.review_lengths}
        )

        # Check that loss is positive and finite
        self.assertGreater(loss_val.item(), 0)
        self.assertTrue(np.isfinite(loss_val.item()))


class TestLossFunctionFactory(unittest.TestCase):
    """Test cases for the loss function factory."""

    def test_get_standard_loss(self):
        """Test that get_loss_function returns standard loss correctly."""
        loss_fn = get_loss_function("standard")
        self.assertIsInstance(loss_fn, torch.nn.BCEWithLogitsLoss)

    def test_get_weighted_loss(self):
        """Test that get_loss_function returns weighted loss correctly."""
        loss_fn = get_loss_function("weighted")
        self.assertIsInstance(loss_fn, SentimentWeightedLoss)

    def test_get_length_aware_loss(self):
        """Test that get_loss_function returns length aware loss correctly."""
        loss_fn = get_loss_function("length_aware")
        self.assertIsInstance(loss_fn, ReviewLengthLoss)

    def test_get_confidence_penalty_loss(self):
        """Test that get_loss_function returns confidence penalty loss correctly."""
        loss_fn = get_loss_function("confidence_penalty")
        self.assertIsInstance(loss_fn, ConfidencePenaltyLoss)

    def test_invalid_loss_type(self):
        """Test that get_loss_function raises ValueError for invalid loss type."""
        with self.assertRaises(ValueError):
            get_loss_function("invalid_loss_type")


class TestLossGradients(unittest.TestCase):
    """Test cases for loss function gradients."""

    def setUp(self):
        """Set up tests fixtures."""
        # Create random dataset
        torch.manual_seed(42)
        self.predictions = torch.randn(16, 1, requires_grad=True)
        self.targets = torch.randint(0, 2, (16, 1)).float()
        self.review_lengths = torch.rand(16)

    def test_gradient_flow(self):
        """Test that gradients flow through the custom loss."""
        loss_fn = SentimentWeightedLoss()

        # Forward pass
        loss = loss_fn(
            self.predictions,
            self.targets,
            {'review_lengths': self.review_lengths}
        )

        # Check that loss requires grad
        self.assertTrue(loss.requires_grad)

        # Backward pass
        loss.backward()

        # Check that gradients are computed
        self.assertIsNotNone(self.predictions.grad)
        self.assertTrue(torch.all(torch.isfinite(self.predictions.grad)))

    def test_zero_length_weight_factor(self):
        """Test that setting length_weight_factor to 0 disables length weighting."""
        # Create loss functions with and without length weighting
        standard_loss = torch.nn.BCEWithLogitsLoss()
        no_length_weight_loss = SentimentWeightedLoss(
            length_weight_factor=0.0,
            confidence_penalty_factor=0.0
        )

        # Compute losses
        standard_loss_val = standard_loss(self.predictions, self.targets)
        no_length_weight_loss_val = no_length_weight_loss(
            self.predictions,
            self.targets,
            {'review_lengths': self.review_lengths}
        )

        # Losses should be very close (may not be exactly equal due to numerical precision)
        self.assertAlmostEqual(
            standard_loss_val.item(),
            no_length_weight_loss_val.item(),
            places=5
        )

    def test_zero_confidence_penalty_factor(self):
        """Test that setting confidence_penalty_factor to 0 disables confidence penalty."""
        # Create loss functions with and without confidence penalty
        standard_loss = torch.nn.BCEWithLogitsLoss()
        no_confidence_penalty_loss = SentimentWeightedLoss(
            length_weight_factor=0.0,
            confidence_penalty_factor=0.0
        )

        # Compute losses
        standard_loss_val = standard_loss(self.predictions, self.targets)
        no_confidence_penalty_loss_val = no_confidence_penalty_loss(self.predictions, self.targets)

        # Losses should be very close
        self.assertAlmostEqual(
            standard_loss_val.item(),
            no_confidence_penalty_loss_val.item(),
            places=5
        )


class TestEdgeCases(unittest.TestCase):
    """Test edge cases for the custom loss functions."""

    def test_all_correct_predictions(self):
        """Test behavior when all predictions are correct."""
        # Create dataset where predictions perfectly match targets
        predictions = torch.tensor([-10.0, 10.0], dtype=torch.float32).reshape(-1, 1)  # Very negative/positive logits
        targets = torch.tensor([0.0, 1.0], dtype=torch.float32).reshape(-1, 1)

        # Loss function with confidence penalty
        loss_fn = SentimentWeightedLoss(confidence_penalty_factor=0.5)

        # Compute loss
        loss_val = loss_fn(predictions, targets)

        # Loss should be positive but small
        self.assertTrue(loss_val.item() > 0)
        self.assertTrue(loss_val.item() < 0.1)  # An arbitrary small threshold

    def test_all_incorrect_predictions(self):
        """Test behavior when all predictions are incorrect."""
        # Create dataset where predictions are opposite of targets
        predictions = torch.tensor([10.0, -10.0], dtype=torch.float32).reshape(-1, 1)  # Very positive/negative logits
        targets = torch.tensor([0.0, 1.0], dtype=torch.float32).reshape(-1, 1)

        # Loss function with confidence penalty
        loss_fn = SentimentWeightedLoss(confidence_penalty_factor=0.5)

        # Compute loss
        loss_val = loss_fn(predictions, targets)

        # Loss should be large due to incorrect confident predictions
        self.assertTrue(loss_val.item() > 5.0)  # An arbitrary large threshold

    def test_uniform_review_lengths(self):
        """Test behavior with uniform review lengths."""
        # Create dataset with uniform review lengths
        predictions = torch.randn(8, 1)
        targets = torch.randint(0, 2, (8, 1)).float()
        uniform_lengths = torch.ones(8) * 0.5  # All reviews have same length

        # Loss function with length weighting
        loss_fn = SentimentWeightedLoss(length_weight_factor=0.5, confidence_penalty_factor=0.0)

        # Compute losses with and without length weighting
        standard_loss = torch.nn.BCEWithLogitsLoss()(predictions, targets)
        weighted_loss = loss_fn(predictions, targets, {'review_lengths': uniform_lengths})

        # Losses should be same since all lengths are equal
        self.assertAlmostEqual(standard_loss.item(), weighted_loss.item(), places=5)

    def test_extreme_length_differences(self):
        """Test behavior with extreme differences in review lengths."""
        # Create dataset with extreme length differences
        predictions = torch.tensor([0.5, 0.5], dtype=torch.float32).reshape(-1, 1)
        targets = torch.tensor([1.0, 1.0], dtype=torch.float32).reshape(-1, 1)
        extreme_lengths = torch.tensor([0.01, 0.99])  # Very short and very long

        # Loss function with high length weight factor
        loss_fn = SentimentWeightedLoss(length_weight_factor=1.0, confidence_penalty_factor=0.0)

        # Compute loss
        loss = loss_fn(predictions, targets, {'review_lengths': extreme_lengths})

        # Loss should be positive and finite
        self.assertTrue(loss.item() > 0)
        self.assertTrue(np.isfinite(loss.item()))


if __name__ == '__main__':
    unittest.main()