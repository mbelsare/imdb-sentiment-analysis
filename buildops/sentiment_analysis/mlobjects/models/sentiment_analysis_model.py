"""
Model architecture for sentiment analysis.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class SentimentTransformer(nn.Module):
    """
    Transformer-based model for sentiment analysis.

    Uses a pre-trained transformer model (e.g., BERT) as backbone,
    with a classification head on top for binary sentiment classification.
    """

    def __init__(
            self,
            pretrained_model_name="bert-base-uncased",
            num_classes=1,
            dropout_prob=0.1,
            freeze_base=False
    ):
        """
        Initialize the model.

        Args:
            pretrained_model_name (str): Name of the pre-trained model to use
            num_classes (int): Number of output classes (1 for binary with BCEWithLogitsLoss)
            dropout_prob (float): Dropout probability for the classification head
            freeze_base (bool): Whether to freeze the pre-trained model parameters
        """
        super(SentimentTransformer, self).__init__()

        # Load pre-trained model configuration and model
        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        self.transformer = AutoModel.from_pretrained(pretrained_model_name)

        # Freeze the transformer parameters if specified
        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False

        # Classification head
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)

        # Initialize weights for the classifier
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier.bias.data.zero_()

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.

        Args:
            input_ids (torch.Tensor): Token IDs
            attention_mask (torch.Tensor): Attention mask

        Returns:
            torch.Tensor: Logits (batch_size, num_classes)
        """
        # Pass input through the transformer model
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Get the [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]

        # Apply dropout and classifier
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

    def get_attention_weights(self, input_ids, attention_mask):
        """
        Get attention weights for visualization.

        Args:
            input_ids (torch.Tensor): Token IDs
            attention_mask (torch.Tensor): Attention mask

        Returns:
            torch.Tensor: Attention weights
        """
        # Forward pass with output_attentions=True
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )

        # Get attention weights from last layer
        attention_weights = outputs.attentions[-1]  # Shape: [batch_size, num_heads, seq_len, seq_len]

        return attention_weights


class SentimentLSTM(nn.Module):
    """
    LSTM-based model for sentiment analysis.

    Uses an embedding layer followed by bidirectional LSTM layers
    and a classification head for binary sentiment prediction.
    """

    def __init__(
            self,
            vocab_size,
            embedding_dim=300,
            hidden_dim=256,
            num_layers=2,
            num_classes=1,
            dropout_prob=0.3,
            bidirectional=True
    ):
        """
        Initialize the LSTM model.

        Args:
            vocab_size (int): Size of the vocabulary
            embedding_dim (int): Dimension of the word embeddings
            hidden_dim (int): Dimension of the LSTM hidden state
            num_layers (int): Number of LSTM layers
            num_classes (int): Number of output classes (1 for binary)
            dropout_prob (float): Dropout probability
            bidirectional (bool): Whether to use a bidirectional LSTM
        """
        super(SentimentLSTM, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Calculate the size of the LSTM output
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Classification head
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_ids):
        """
        Forward pass through the model.

        Args:
            input_ids (torch.Tensor): Token IDs
        Returns:
            torch.Tensor: Logits (batch_size, num_classes)
        """
        # Get embeddings
        embeddings = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]

        # Pass through LSTM
        lstm_out, (hidden, _) = self.lstm(embeddings)  # lstm_out: [batch_size, seq_len, hidden_dim*directions]

        # Get the final hidden state
        if self.lstm.bidirectional:
            # Concatenate the final forward and backward hidden states
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1]

        # Apply dropout and classifier
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)

        return logits


def get_model(model_type="transformer", **kwargs):
    """
    Factory function to create model instances.

    Args:
        model_type (str): Type of model to create ("transformer" or "lstm")
        **kwargs: Additional arguments to pass to the model constructor

    Returns:
        nn.Module: Model instance
    """
    if model_type == "transformer":
        return SentimentTransformer(**kwargs)
    elif model_type == "lstm":
        return SentimentLSTM(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")