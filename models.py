from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd

# HF model repo ID
repo_id = "Sooryanjali/early-classification-models"

# Load trained DistilBERT models directly from Hugging Face Hub
model_paths = [
    "saved_model30",
    "saved_model50",
    "saved_model70",
]

models = [
    DistilBertForSequenceClassification.from_pretrained(f"{repo_id}", subfolder=subfolder)
    for subfolder in model_paths
]

# Load tokenizer from base model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Load pre-split datasets
dataset1 = pd.read_csv("test_output30.csv")
dataset2 = pd.read_csv("test_output50.csv")
dataset3 = pd.read_csv("test_output70.csv")

# Ensure the datasets have the required columns
for dataset in [dataset1, dataset2, dataset3]:
    dataset["text"] = dataset["text"].astype(str)
    if "label" not in dataset.columns:
        raise ValueError("One of the datasets is missing the 'label' column.")

# Store the datasets as lists
full_datasets = [
    dataset1.values.tolist(),
    dataset2.values.tolist(),
    dataset3.values.tolist()
]
