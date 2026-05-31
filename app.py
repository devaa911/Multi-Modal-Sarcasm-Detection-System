from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel, AutoTokenizer, AutoModel
import re

# Initialize Flask app
app = Flask(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load RoBERTa Tokenizer and Model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaModel.from_pretrained("roberta-base").to(device)

# Load Emoji-BERT Tokenizer and Model
emoji_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emoji")
emoji_model = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-emoji").to(device)

# Define Sarcasm Detection Model
class SarcasmModel(nn.Module):
    def __init__(self, input_dim):
        super(SarcasmModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Load the trained model
model = SarcasmModel(input_dim=1536).to(device)
model.load_state_dict(torch.load("sarcasm_detection_model.pth", map_location=device))
model.eval()

# Function to preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|pic\.twitter\.com/\S+", "", text)  # Remove links
    text = re.sub(r"[^\w\s#@]", "", text)  # Remove special characters except hashtags and mentions
    return text

# Function to extract emojis
def extract_emojis(text):
    emojis = re.findall(r"[^\w\s,]", text)
    return "".join(emojis) if emojis else "[NO_EMOJI]"

# Function to get RoBERTa text embeddings
def get_text_embedding(text):
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        output = roberta_model(**tokens).last_hidden_state[:, 0, :]
    return output

# Function to get Emoji-BERT emoji embeddings
def get_emoji_embedding(emoji_text):
    tokens = emoji_tokenizer(emoji_text, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        output = emoji_model(**tokens).last_hidden_state[:, 0, :]
    return output

# Function to predict sarcasm
def predict_sarcasm(tweet):
    cleaned_text = clean_text(tweet)
    extracted_emojis = extract_emojis(tweet)

    text_embedding = get_text_embedding(cleaned_text)
    emoji_embedding = get_emoji_embedding(extracted_emojis)

    # Normalize and concatenate embeddings
    text_embedding = torch.nn.functional.normalize(text_embedding, p=2, dim=1)
    emoji_embedding = torch.nn.functional.normalize(emoji_embedding, p=2, dim=1)
    combined_embedding = torch.cat((text_embedding, emoji_embedding), dim=1)

    # Get model prediction
    with torch.no_grad():
        output = model(combined_embedding).item()
    
    sarcasm_label = "Sarcastic" if output > 0.5 else "Not Sarcastic"
    confidence = output if sarcasm_label == "Sarcastic" else 1 - output
    return sarcasm_label, confidence

# Home Route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_input = data.get("text", "")
    
    if not user_input:
        return jsonify({"error": "No text provided"}), 400

    label, confidence = predict_sarcasm(user_input)

    return jsonify({"label": label, "confidence": round(confidence, 4)})

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
