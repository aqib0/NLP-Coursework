from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import numpy as np
import joblib
import os
import concurrent.futures
import requests

# Define the label encoding
label_encoding = {"B-O": 0, "B-AC": 1, "B-LF": 2, "I-LF": 3}
label_decoding = {v: k for k, v in label_encoding.items()}

# Load the model and tokenizer
model_path = "trained_model.joblib"
tokenizer_path = "tokenizer.joblib"

if not os.path.isfile(model_path) or not os.path.isfile(tokenizer_path):
    raise FileNotFoundError("Model or tokenizer not found. Please ensure the paths are correct.")

model = joblib.load(model_path)
tokenizer = joblib.load(tokenizer_path)

# Determine the device to use (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_ner(tokens, model, tokenizer, device):
    tokenized_input = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True).to(device)
    model.eval()
    with torch.no_grad():
        output = model(**tokenized_input)
    predictions = np.argmax(output.logits.cpu().numpy(), axis=2)
    predicted_labels = [label_decoding[label] for label in predictions[0]]
    token_label_pairs = list(zip(tokens, predicted_labels))
    return token_label_pairs

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    tokens = data.get("tokens")
    if not tokens:
        return jsonify({"error": "No tokens provided"}), 400

    predictions = predict_ner(tokens, model, tokenizer, device)
    if predictions is None:
        return jsonify({"error": "Prediction error"}), 500

    # Write predictions to the file
    with open("results.txt", "a") as file:
        file.write("Prediction:\n")
        for token, label in predictions:
            file.write(f"{token}: {label}\n")
        file.write("\n")

    return jsonify(predictions)

@app.route('/stress_test', methods=['POST'])
def stress_test():
    data = request.json
    tokens = data.get("tokens")
    if not tokens:
        return jsonify({"error": "No tokens provided"}), 400

    all_predictions = []
    with open("results.txt", "a") as file:
        file.write("Stress Test:\n")
        for i in range(50):
            predictions = predict_ner(tokens, model, tokenizer, device)
            if predictions is None:
                return jsonify({"error": "Prediction error during stress test"}), 500
            all_predictions.append(predictions)
            file.write(f"Iteration {i+1}:\n")
            for token, label in predictions:
                file.write(f"{token}: {label}\n")
            file.write("\n")

    return jsonify(all_predictions)

@app.route('/run_stress_test', methods=['POST'])
def run_stress_test():
    data = request.json
    num_requests = data.get("num_requests", 50)
    url = "http://localhost:5003/predict"
    payload = {
        "tokens": ["For", "this", "purpose", "the", "Gothenburg", "Young", "Persons", "Empowerment", "Scale", "(", "GYPES", ")", "was", "developed", "."]
    }
    headers = {
        "Content-Type": "application/json"
    }

    times = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(send_request, url, payload, headers) for _ in range(num_requests)]
        for future in concurrent.futures.as_completed(futures):
            status_code, elapsed_time = future.result()
            times.append(elapsed_time)

    avg_time = sum(times) / len(times)
    max_time = max(times)
    min_time = min(times)

    with open("results.txt", "a") as file:
        file.write(f"Stress Test Results: {num_requests} requests\n")
        file.write(f"Average Response Time: {avg_time} seconds\n")
        file.write(f"Max Response Time: {max_time} seconds\n")
        file.write(f"Min Response Time: {min_time} seconds\n")

    return jsonify({
        "average_time": avg_time,
        "max_time": max_time,
        "min_time": min_time
    })

@app.route('/get_latest_results', methods=['GET'])
def get_latest_results():
    if not os.path.isfile("results.txt"):
        return jsonify({"error": "Results file not found"}), 404

    with open("results.txt", "r") as file:
        results = file.read()

    return jsonify({"results": results})

@app.route('/clear_results', methods=['POST'])
def clear_results():
    open("results.txt", "w").close()
    return jsonify({"message": "Results file cleared"}), 200

def send_request(url, payload, headers):
    response = requests.post(url, json=payload, headers=headers)
    return response.status_code, response.elapsed.total_seconds()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)

