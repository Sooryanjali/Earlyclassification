import numpy as np
import torch.nn.functional as F
import torch
from models import models, tokenizer, model_paths  # ✅ Import model names too!

# Load Decision Templates
DT_0 = np.load("decision_template_0.npy")  # Non-Offensive DT
DT_1 = np.load("decision_template_1.npy")  # Offensive DT

# Function to get model probability outputs
def get_model_output(model, text):
    """Returns probability scores from a DistilBERT model"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits  # Get raw logits
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # Convert logits to probabilities
    return probs  # [p(non-offensive), p(offensive)]

# Function to classify text using Decision Templates & print model outputs
def classify_text(text):
    """Classifies a given text based on Decision Profiles and prints model predictions"""
    print(f"\n🔍 Classifying: \"{text}\"")
    
    dp = []  # Decision Profile storage
    model_predictions = []  # Store model-wise predictions

    # Loop through models & collect outputs
    for model, model_name in zip(models, model_paths):
        probs = get_model_output(model, text)  # Get model's probability scores
        predicted_label = np.argmax(probs)  # Get class prediction (0 or 1)
        
        dp.append(probs)  # Store in DP
        model_predictions.append((model_name, predicted_label, probs))  # Store for display
    
    dp = np.array(dp)  # Convert to numpy array

    # Compute Euclidean Distance to each Decision Template
    dist_0 = np.linalg.norm(dp - DT_0)  # Distance to Non-Offensive DT
    dist_1 = np.linalg.norm(dp - DT_1)  # Distance to Offensive DT

    # Final class decision
    final_prediction = 0 if dist_0 < dist_1 else 1  

    # Print individual model predictions
    print("\n📊 **Model Predictions:**")
    for model_name, pred, probs in model_predictions:
        class_label = "Non-Offensive" if pred == 0 else "Offensive"
        print(f"🟢 {model_name} → Prediction: {class_label} | Probabilities: {probs}")

    # Print final fused decision
    print(f"\n🔮 **Final Decision:** {'Offensive' if final_prediction == 1 else 'Non-Offensive'} (based on Decision Templates)")

    return final_prediction  # Return final predicted class

# Example Classification
text_sample="People here are so fun" 


prediction = classify_text(text_sample)
