import os
import pickle
import re
import torch
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU execution

print("[INFO] Initializing tokenizer and models...", flush=True)

# Load Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load Models
models = {
    'model30': DistilBertForSequenceClassification.from_pretrained('saved_model30').to('cpu'),
    'model50': DistilBertForSequenceClassification.from_pretrained('saved_model50').to('cpu'),
    'model70': DistilBertForSequenceClassification.from_pretrained('saved_model70').to('cpu')
}

for model in models.values():
    model.eval()

print("[INFO] Models loaded successfully.", flush=True)

def get_predictions(model, texts):
    texts = [str(text) if isinstance(text, str) else "" for text in texts]  # Ensure valid input
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to('cpu')
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    preds = torch.argmax(probs, axis=1).tolist()
    confidence = probs[:, 1].tolist()
    return preds, confidence

def markov_fusion(temp_results, text):
    model_predictions = {model: result[0][0] for model, result in temp_results.items()}
    model_confidences = {model: result[1][0] for model, result in temp_results.items()}
    
    total_confidence = sum(model_confidences.values())
    weighted_prediction = sum(model_predictions[model] * model_confidences[model] for model in model_predictions) / total_confidence
    
    return 1 if weighted_prediction > 0.5 else 0

def load_test_datasets():
    files = ['test_output30.csv', 'test_output50.csv', 'test_output70.csv']
    datasets = [pd.read_csv(f).sample(frac=0.25, random_state=42) for f in files]
    return pd.concat(datasets)

test_data = load_test_datasets()
test_data.dropna(subset=['text', 'label'], inplace=True)
test_data['text'] = test_data['text'].astype(str)
test_data['label'] = test_data['label'].astype(int)

print("[INFO] Running evaluation...")

y_true = []
y_pred = []
y_scores = []

for _, row in test_data.iterrows():
    temp_results = {name: get_predictions(model, [row['text']]) for name, model in models.items()}
    prediction = markov_fusion(temp_results, row['text'])
    confidence_score = sum(temp_results[model][1][0] for model in models) / len(models)
    y_true.append(row['label'])
    y_pred.append(prediction)
    y_scores.append(confidence_score)

print("[INFO] Evaluation Metrics:")
print(classification_report(y_true, y_pred, digits=4))

# Compute ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Compute and Display Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

print("[INFO] Saving model...")
with open("markov_classifier.pkl", "wb") as f:
    pickle.dump(markov_fusion, f)
print("[INFO] Model saved as 'markov_classifier.pkl'")

if _name_ == "_main_":
    while True:
        text = input("Enter a sentence to classify (or type 'exit' to quit): ")
        if text.lower() == 'exit':
            break
        temp_results = {name: get_predictions(model, [text]) for name, model in models.items()}
        prediction = markov_fusion(temp_results, text)
        print("Prediction:", "Offensive" if prediction == 1 else "Non-Offensive")


