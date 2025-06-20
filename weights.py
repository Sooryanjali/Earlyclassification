import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch
from models import models, tokenizer, model_paths  # ✅ Import model paths to print model names

# Load the common validation dataset
validation_df = pd.read_csv("validation.csv")
# Ensure text column is properly formatted as strings
validation_df["text"] = validation_df["text"].astype(str)

# Check if the dataset has the required columns
if "label" not in validation_df.columns:
    raise ValueError("Dataset is missing the 'label' column.")

# Split the dataset into different fractions for each model
split_ratios = [0.33, 0.33, 0.34]  # Adjust based on the number of models
full_datasets = []

# Shuffle dataset before splitting to ensure randomness
validation_df = validation_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Assign different dataset fractions to each model
start_idx = 0
for ratio in split_ratios:
    split_size = int(len(validation_df) * ratio)
    subset = validation_df.iloc[start_idx : start_idx + split_size]
    full_datasets.append((subset["text"].tolist(), subset["label"].tolist()))  # Convert to lists
    start_idx += split_size

# Placeholder for classifier accuracy scores
classifier_weights = []
model_accuracies = []  # ✅ Store accuracies for printing later

# Function to get model probability outputs
def get_model_output(model, text):
    """Returns probability scores from a DistilBERT model"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits  # Get raw logits
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # Convert logits to probabilities
    return probs

# Evaluate each model on its assigned validation set fraction
for model_idx, (model, model_name, (validation_texts, validation_labels)) in enumerate(zip(models, model_paths, full_datasets)):
    correct = 0
    total = 0

    print(f"🔄 Evaluating Model {model_idx+1}/{len(models)}: {model_name}")

    for i, (text, label) in enumerate(zip(validation_texts, validation_labels)):
        probs = get_model_output(model, text)  # Get model predictions
        predicted_label = np.argmax(probs)  # Get class with highest probability

        if predicted_label == label:
            correct += 1
        total += 1

        if i % 1000 == 0:  # Print progress every 1000 samples
            print(f"   ✅ Processed {i}/{len(validation_texts)} samples for Model {model_idx+1}")

    accuracy = correct / total if total > 0 else 0  # Avoid division by zero
    classifier_weights.append(accuracy)
    model_accuracies.append((model_name, accuracy))  # ✅ Store model name and accuracy

# Normalize weights so they sum to 1
if sum(classifier_weights) > 0:
    classifier_weights = np.array(classifier_weights) / sum(classifier_weights)
else:
    classifier_weights = np.array([1/len(models)] * len(models))  # Equal weights if all accuracies are zero

# Save the computed weights for later use
np.save("classifier_weights1.npy", classifier_weights)

# ✅ Print model accuracies and final weights
print("\n📊 **Model Accuracies and Weights**")
for (model_name, accuracy), weight in zip(model_accuracies, classifier_weights):
    print(f"🟢 Model: {model_name} | Accuracy: {accuracy:.4f} | Final Weight: {weight:.4f}")

print("\n✅ Classifier Weights Computed & Saved Successfully!")
