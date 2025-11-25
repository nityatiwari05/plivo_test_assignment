# PII Entity Recognition for STT Transcripts

## Overview
This project aims to build a machine learning model for token-level Named Entity Recognition (NER) that identifies and classifies Personally Identifiable Information (PII) entities within noisy speech-to-text (STT) transcripts. The model is designed to detect various entity types, including credit card numbers, phone numbers, emails, personal names, dates, cities, and locations.

## Project Structure
The project is organized as follows:

```
pii_ner_assignment
├── src
│   ├── dataset.py        # Handles loading and preprocessing of the dataset
│   ├── labels.py         # Defines BIO labeling scheme and PII mappings
│   ├── model.py          # Implements the token classification model
│   ├── train.py          # Contains the training loop for the model
│   ├── predict.py        # Runs inference and generates predictions
│   ├── eval_span_f1.py   # Evaluates model predictions against gold annotations
│   └── measure_latency.py # Measures inference latency of the model
├── data
│   ├── train.jsonl       # Training dataset in JSONL format
│   ├── dev.jsonl         # Development dataset for validation
│   ├── test.jsonl        # Test dataset for final evaluation
│   └── stress.jsonl      # Adversarial dataset for robustness evaluation
├── out
│   ├── dev_pred.json     # Predictions for the development dataset
│   ├── stress_pred.json  # Predictions for the stress-test dataset
│   └── config.json       # Configuration settings for the model
├── requirements.txt       # Lists project dependencies
└── README.md              # Documentation for the project
```

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd pii_ner_assignment
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. **Training the Model:**
   To train the model, run the following command:
   ```
   python src/train.py --model_name distilbert-base-uncased --train data/train.jsonl --dev data/dev.jsonl --out_dir out
   ```

2. **Running Inference:**
   After training, you can run inference on the development and stress-test datasets:
   ```
   python src/predict.py --model_dir out --input data/dev.jsonl --output out/dev_pred.json
   python src/predict.py --model_dir out --input data/stress.jsonl --output out/stress_pred.json
   ```

3. **Evaluating Predictions:**
   To evaluate the model's predictions, use:
   ```
   python src/eval_span_f1.py --gold data/dev.jsonl --pred out/dev_pred.json
   python src/eval_span_f1.py --gold data/stress.jsonl --pred out/stress_pred.json
   ```

4. **Measuring Latency:**
   To measure the inference latency, run:
   ```
   python src/measure_latency.py --model_dir out --input data/dev.jsonl --runs 50
   ```

## Evaluation Metrics
The model's performance will be evaluated based on:
- Precision, Recall, and F1 Score for PII entities (CREDIT_CARD, PHONE, EMAIL, PERSON_NAME, DATE).
- Latency metrics (p50 and p95) for inference speed.

## Conclusion
This project aims to provide a robust solution for detecting PII in STT transcripts, ensuring both high precision and fast inference.