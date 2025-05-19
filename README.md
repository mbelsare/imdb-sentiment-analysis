## IMDB Sentiment Analysis with Custom Loss Functions
This project implements a sentiment analysis model for IMDB movie reviews using PyTorch and Hugging Face Transformers. 
The implementation features custom loss functions that account for review length and prediction confidence to improve model performance.


### Project Overview
Sentiment analysis is the task of determining the emotional tone behind a text document. 
In this project, we analyze movie reviews from the IMDB dataset and classify them as either positive or negative.
Implementation of a custom loss function that addresses following challenges:
* Review length variations (longer reviews may contain more information but also more noise)
* Confidence calibration (penalizing overconfident incorrect predictions)
* Class imbalance handling

Following are the components of this project:
#### Data Handling

* Custom PyTorch Dataset and DataLoader for IMDB reviews
* Data preprocessing and tokenization using Hugging Face transformers
* Train/validation/test splitting functionality


#### Model Implementation

* Transformer-based model using pre-trained BERT
* Alternative LSTM model option for comparison
* Flexible model architecture with customizable parameters


#### Custom Loss Functions

* Review Length-Aware Loss: Adjusts importance based on review length
* Confidence Penalty Loss: Penalizes overconfident incorrect predictions
* Combined Loss: Integrates both approaches with optional class weighting


#### Training Pipeline

* Complete training loop with validation
* Learning rate scheduling
* Model checkpointing
* Training history tracking and visualization


#### Evaluation

* Comprehensive evaluation metrics (accuracy, F1, precision, recall)
* Confusion matrix and ROC curve visualization
* Error analysis functionality
* Attention visualization for transformer models


#### Command-line Interface

* Training script with customizable parameters
* Prediction script for inferencing with trained models

Understanding the custom loss function is the key to optimizing sentiment analysis models for IMDB reviews.

### Custom Loss Function
The core contribution of this project is the SentimentWeightedLoss class, which extends the standard `BCEWithLogitsLoss` with three important weighting mechanisms:

#### 1. Review Length Weighting
Movie reviews vary significantly in length, from a few sentences to several paragraphs. The hypothesis is that:

Very short reviews might be too terse to provide adequate context
Very long reviews might contain irrelevant information or repetition

The length-aware component adjusts the importance of each sample based on its normalized length:
```
length_weights = 1.0 + (review_lengths - 0.5) * self.length_weight_factor
```

#### 2. Confidence Penalty
Deep learning models can be overly confident in their incorrect predictions. The loss function penalizes overconfident incorrect predictions more heavily to encourage better calibration:
Calculate confidence as distance from decision boundary
```
confidence = torch.abs(probs - 0.5) * 2  # Scale to [0, 1]
```

Confidence penalty applies only to incorrect predictions
```
confidence_penalty = (1 - correct_predictions) * confidence * self.confidence_penalty_factor
```

#### 3. Class Weighting
For datasets with class imbalance, the loss function supports class-specific weights to give more importance to the minority class.

### Installation and Setup

* Clone the repository:

```
git clone https://github.com/mbelsare/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

* Create and activate a virtual environment:

```
python -m venv venv
source venv/bin/activate
```

Install dependencies:

`pip install -r requirements.txt`

### Usage

#### Training a Model

* Make sure the virtual env `venv` is created and activated.
* Before running training, make sure `buildops/sentiment_analysis/models/sentiment_model` directory exists. If not, just create it manually for example with 
```
mkdir -p buildops/sentiment_analysis/models/sentiment_model
```

```
python3 buildops/sentiment_analysis/workflow/jobs/model_trainer.py --epochs 5 --batch_size 16 --model_type transformer --loss_type combined
```
* You can choose the loss type among `standard, length_aware, confidence_penalty, weighted, combined`

If the training breaks after completing an epoch, you can use the `--resume` flag to resume training from the last checkpoint.

#### Make predictions:
```
python3 buildops/sentiment_analysis/workflow/jobs/model_predictor.py --input "This movie was absolutely fantastic!"
```

To train a model with the custom loss function:
```
python3 buildops/sentiment_analysis/workflow/jobs/run_experiments.py --loss_types weighted --epochs 3
```

#### Comparing Different Loss Functions
To compare standard loss function with custom loss variants:
```
python3 buildops/sentiment_analysis/workflow/jobs/run_experiments.py --loss_types standard length_aware confidence_penalty weighted combined --epochs 3
```

To use an existing model to compare loss functions:
```
python3 buildops/sentiment_analysis/workflow/jobs/run_experiments.py --use_existing_model --model_path buildops/sentiment_analysis/models/sentiment_model/checkpoint_epoch_1_step_2000.pt --loss_types weighted standard length_aware
```

#### Performance Metrics

* Accuracy: 93.2%
* F1 Score: 0.93
* Precision: 0.92
* Recall: 0.94

#### Visualizations
The project includes comprehensive visualization tools:

1. Training history (loss, accuracy, F1 score)
2. Confusion matrices
3. ROC curves
4. Precision-recall curves
5. Attention weight visualizations (for transformer models)
6. Loss function comparison charts

Example visualizations are automatically generated during the experiments and saved to the output directory.

#### Unit Tests
The project includes unit tests for critical components, especially the custom loss functions. To run the tests:
```
python3 -m unittest discover buildops/sentiment_analysis/tests
```

#### Future Improvements
* Experiment with different hyperparameters:
* Try different pre-trained models (RoBERTa, DistilBERT)
* Adjust the loss function parameters
* Experiment with longer training or larger batch sizes
