import os
import json
import logging
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("TransformerAPI")

# ============================================================
# DEVICE
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Running on {DEVICE}")

# ============================================================
# PATHS – all files must be in the same directory as app.py
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best_transformer_model.pt")
CONFIG_PATH = os.path.join(BASE_DIR, "model_configuration.json")
TOKEN_PATH = os.path.join(BASE_DIR, "token_dictionary.json")
THRESHOLD_PATH = os.path.join(BASE_DIR, "optimized_thresholds.json")

# ============================================================
# LOAD CONFIGURATION FILES
# ============================================================
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

try:
    CONFIG = load_json(CONFIG_PATH)
    TOKEN_DICTIONARY = load_json(TOKEN_PATH)
    THRESHOLDS = load_json(THRESHOLD_PATH)
except FileNotFoundError as e:
    logger.error(f"Missing file: {e}")
    raise

CONFIDENCE_THRESHOLD = float(THRESHOLDS.get("optimal_threshold", 0.55))

logger.info("Configuration loaded successfully")
logger.info(f"Vocabulary size : {CONFIG['vocab_size']}")
logger.info(f"Sequence length : {CONFIG['max_seq_len']}")
logger.info(f"Confidence threshold : {CONFIDENCE_THRESHOLD}")

# ============================================================
# TRANSFORMER MODEL (must match training architecture)
# ============================================================
class TransformerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(
            CONFIG["vocab_size"],
            CONFIG["d_model"]
        )
        self.position_embedding = nn.Parameter(
            torch.randn(1, CONFIG["max_seq_len"], CONFIG["d_model"])
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=CONFIG["d_model"],
            nhead=CONFIG["n_heads"],
            dim_feedforward=CONFIG["d_ff"],
            dropout=CONFIG["dropout"],
            activation=CONFIG.get("activation", "gelu"),
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=CONFIG["n_layers"]
        )
        self.classifier = nn.Linear(
            CONFIG["d_model"],
            CONFIG["num_classes"]
        )

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.position_embedding[:, :x.size(1), :]
        x = self.encoder(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits

# ============================================================
# LOAD TRAINED MODEL
# ============================================================
logger.info("Loading trained model...")
model = TransformerClassifier()
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()
logger.info("Model loaded successfully")

# ============================================================
# FLASK APP
# ============================================================
app = Flask(__name__)
CORS(app)
logger.info("Flask server initialized")

# ============================================================
# HEALTH ENDPOINT
# ============================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": "Transformer V3",
        "device": str(DEVICE),
        "vocab_size": CONFIG["vocab_size"],
        "sequence_length": CONFIG["max_seq_len"],
        "confidence_threshold": CONFIDENCE_THRESHOLD
    })

# ============================================================
# PREDICT ENDPOINT
# ============================================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        if payload is None:
            return jsonify({"error": "No JSON payload received."}), 400

        token_sequence = payload.get("token_sequence")
        asset = payload.get("asset", "UNKNOWN")
        timeframe = payload.get("timeframe", 60)

        if token_sequence is None:
            return jsonify({"error": "token_sequence missing."}), 400

        if len(token_sequence) != CONFIG["max_seq_len"]:
            return jsonify({
                "error": f"Expected {CONFIG['max_seq_len']} rows."
            }), 400

        for row in token_sequence:
            if len(row) != 8:
                return jsonify({
                    "error": "Each row must contain exactly 8 token IDs."
                }), 400

        x = torch.tensor(token_sequence, dtype=torch.long, device=DEVICE).unsqueeze(0)

        with torch.no_grad():
            logits = model(x)
            probabilities = torch.softmax(logits, dim=1)[0]

        p_down = float(probabilities[0])
        p_up = float(probabilities[1])

        if p_up >= CONFIDENCE_THRESHOLD:
            action = "BUY"
            confidence = p_up
        elif p_down >= CONFIDENCE_THRESHOLD:
            action = "SELL"
            confidence = p_down
        else:
            action = "NO_TRADE"
            confidence = max(p_up, p_down)

        return jsonify({
            "success": True,
            "asset": asset,
            "timeframe": timeframe,
            "action": action,
            "confidence": round(confidence, 4),
            "p_up": round(p_up, 4),
            "p_down": round(p_down, 4),
            "threshold": CONFIDENCE_THRESHOLD
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

# ============================================================
# ROOT ENDPOINT
# ============================================================
@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "i90 Transformer API",
        "status": "running",
        "version": "3.0"
    })

# ============================================================
# LOCAL DEVELOPMENT (ignored by Gunicorn)
# ============================================================
if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting development server on port {PORT}")
    app.run(host="0.0.0.0", port=PORT)
