from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
import os
import requests

app = Flask(__name__)

# Define the path to your model directory
model_path = "./model"

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Set model to evaluation mode
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Receive the JSON
    sentences = data['sentences']  # Extract sentences

    predictions = []
    for sentence in sentences:
        # Preprocess and tokenize the sentence
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get the predicted probabilities using softmax
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)  # Convert logits to probabilities
        
        # Find the maximum probability and corresponding label
        max_prob, predicted_class = torch.max(probabilities, dim=1)
        max_prob = max_prob.item()  # Convert tensor to a float value
        predicted_class = predicted_class.item()  # Convert tensor to int
        
        # Check if the max probability is greater than 70%
        if max_prob > 0.6 and predicted_class != 10:
            # Return the predicted label and probability
            predictions.append({
                "sentence": sentence,
                "label": predicted_class,
                "probability": round(max_prob * 100 * 0.65, 2)  # Return as a percentage
            })

    return jsonify(predictions)
