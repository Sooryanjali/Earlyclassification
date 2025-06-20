import torch
import re
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load datasets
data_30 = pd.read_csv("test_output30.csv")
data_50 = pd.read_csv("test_output50.csv")
data_70 = pd.read_csv("test_output70.csv")

# Combine datasets
test_set = pd.concat([data_30, data_50, data_70], ignore_index=True)

# Load classifiers
def load_classifier(model_path):
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    return model, tokenizer

C_30_model, C_30_tokenizer = load_classifier('saved_model30')
C_50_model, C_50_tokenizer = load_classifier('saved_model50')
C_70_model, C_70_tokenizer = load_classifier('saved_model70')

# Offensive Word Dictionary
offensive_dict = {
    'bitch': ['bitch', 'bitches', 'bit', 'bich', 'bish'],
    'nigga': ['nigga', 'nigger', 'niggah', 'nigg', 'nig'],
    'fuck': ['fuck', 'fucking', 'fucker', 'fuk', 'fukk', 'motherfucker'],
    'ass': ['ass', 'asshole', 'azz', 'dumbass', 'dumb'],
    'shit': ['shit', 'shitty', 'shyt'],
    'faggot': ['faggot', 'fag', 'faag'],
    'cunt': ['cunt', 'cnt'],
    'hoe': ['hoe', 'hoes', 'ho'],
    'pussy': ['pussy', 'pusy', 'pussi']
}

# Function to check for offensive words
def contains_offensive_word(text):
    text = text.lower()
    for variations in offensive_dict.values():
        for variant in variations:
            if variant in text:  # Check if the word appears in the text
                return True
    return False

# Function to get softmax score from models
def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    return probs[0][1].item()  # Assuming index 1 is the offensive class

# Hierarchical Decision Fusion function with Immediate Flagging
def hierarchical_decision_fusion(input_text, C_30_model, C_30_tokenizer, C_50_model, C_50_tokenizer, C_70_model, C_70_tokenizer, threshold=0.7, alpha=0.7, beta=0.7):
    # Immediate flagging if any offensive word is found
    if contains_offensive_word(input_text):
        print(f"⚠️ Offensive word detected. Automatically classified as Offensive.")
        return "Offensive"

    # Step 1: Get softmax score from C_30
    S_30 = predict(input_text, C_30_model, C_30_tokenizer)
    if S_30 >= threshold:
        return "Offensive"
    
    # Step 2: Get softmax score from C_50 and compute Y_50
    S_50 = predict(input_text, C_50_model, C_50_tokenizer)
    Y_50 = alpha * S_50 + (1 - alpha) * S_30
    if Y_50 >= threshold:
        return "Offensive"
    
    # Step 3: Get softmax score from C_70 and compute Y_70
    S_70 = predict(input_text, C_70_model, C_70_tokenizer)
    Y_70 = beta * S_70 + (1 - beta) * Y_50
    return "Offensive" if Y_70 >= threshold else "Non-Offensive"

# Evaluate on combined dataset
y_true = []
y_pred = []

for _, row in test_set.iterrows():
    text = str(row["text"]) if pd.notna(row["text"]) else ""
    true_label = "Offensive" if row["label"] == 1 else "Non-Offensive"

    print(f"Processing: {text}")

    predicted_label = hierarchical_decision_fusion(
        text, C_30_model, C_30_tokenizer, C_50_model, C_50_tokenizer, C_70_model, C_70_tokenizer
    )

    y_true.append(true_label)
    y_pred.append(predicted_label)

# Compute evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, pos_label="Offensive")
recall = recall_score(y_true, y_pred, pos_label="Offensive")
f1 = f1_score(y_true, y_pred, pos_label="Offensive")

print("\nEvaluation Results:")
print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.2%}")
