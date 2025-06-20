from flask import Flask, request, jsonify, send_file
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

app = Flask(__name__)

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

def contains_offensive_word(text):
    text = text.lower()
    for variations in offensive_dict.values():
        for variant in variations:
            if variant in text:
                return True
    return False

# Function to get softmax score from models
def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    return probs[0][1].item()  # Assuming index 1 is the offensive class

# Decision Fusion Function
def hierarchical_decision_fusion(input_text, threshold=0.7, alpha=0.7, beta=0.7):
    # Immediate flagging if any offensive word is found
    if contains_offensive_word(input_text):
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

# Serve HTML Page
@app.route("/")
def home():
    return send_file("index.html")  # Ensure index.html is in the same folder as app.py

# API Endpoint
@app.route("/classify", methods=["POST"])
def classify_text():
    data = request.json
    text = data.get("text", "")

    # Run hierarchical decision fusion
    prediction = hierarchical_decision_fusion(text)

    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
