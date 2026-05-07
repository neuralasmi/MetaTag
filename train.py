import torch
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from seqeval.metrics import f1_score, classification_report

MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 9  # PER, LOC, ORG, THEME, MOOD, GENRE, RATING, TAG, O

label2id = {"O": 0, "PER": 1, "LOC": 2, "ORG": 3, "THEME": 4, "MOOD": 5, "GENRE": 6, "RATING": 7, "TAG": 8}
id2label = {v: k for k, v in label2id.items()}

def align_labels(tokens, labels, tokenizer, label2id):
    """Align word-level labels to subword tokens."""
    aligned_labels = []
    for tok in tokens:
        subwords = tokenizer.tokenize(tok)
        aligned_labels.extend([label2id.get(labels[len(aligned_labels)], -100)] * len(subwords))
    return aligned_labels

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    true_preds = [[id2label[p] for p, l in zip(pred, label) if l != -100] for pred, label in zip(predictions, labels)]
    return {"f1": f1_score(true_labels, true_preds)}

def train():
    model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=5e-5,
        evaluation_strategy="epoch",
        logging_steps=100,
        report_to="none",
    )
    
    print("Training MetaTag NER model...")
    print("Load your dataset and format with tokens + BIO labels")
    print("Example: python train.py --dataset my_data.csv --epochs 3")
    return model, tokenizer

if __name__ == "__main__":
    train()
