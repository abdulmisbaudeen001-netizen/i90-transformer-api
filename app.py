# ============================================================
# i90 TRANSFORMER API SERVER
# Flask + PyTorch
#
# Architecture matches Cell 7A/7B of training notebook exactly:
#   token_embedding  (Embedding)
#   pos_encoding     (nn.Parameter, learnable)
#   encoder          (TransformerEncoder, norm_first=True)
#   embed_dropout    (Dropout)
#   classifier       (Sequential: LayerNorm → Linear)
#
# Forward pass:
#   Input: (batch, seq_len, num_features) — integer token IDs
#   For each of 8 feature channels: embed → sum across channels
#   + learnable positional encoding
#   → TransformerEncoder
#   → mean pooling over seq dimension
#   → LayerNorm + Linear → logits (2 classes)
#
# Receives from browser:
#   POST /predict { token_sequence: [[...8 ids...] x8], asset, timeframe }
# ============================================================

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
log = logging.getLogger(__name__)

# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log.info("=" * 60)
log.info("Starting i90 Transformer API...")
log.info(f"Running on {DEVICE}")
log.info("=" * 60)

# ============================================================
# FILES
# ============================================================

MODEL_PATH     = "best_transformer_model.pt"
CONFIG_PATH    = "model_configuration.json"
TOKEN_PATH     = "token_dictionary.json"
THRESHOLD_PATH = "optimized_thresholds.json"

# ============================================================
# LOAD CONFIG
# ============================================================

log.info("Loading configuration...")

with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

with open(TOKEN_PATH, "r") as f:
    TOKEN_DICTIONARY = json.load(f)

with open(THRESHOLD_PATH, "r") as f:
    THRESHOLDS = json.load(f)

CONFIDENCE_THRESHOLD = float(THRESHOLDS.get("optimal_threshold", 0.55))

log.info(f"Vocabulary size      : {CONFIG['vocab_size']}")
log.info(f"Sequence length      : {CONFIG['max_seq_len']}")
log.info(f"Confidence threshold : {CONFIDENCE_THRESHOLD}")

# ============================================================
# MODEL ARCHITECTURE
# Must match Cell 7A of the training notebook exactly.
# Layer names must match the saved state_dict keys.
# ============================================================

class MarketTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.vocab_size       = config["vocab_size"]
        self.d_model          = config["d_model"]
        self.max_seq_len      = config["max_seq_len"]
        self.n_heads          = config["n_heads"]
        self.n_layers         = config["n_layers"]
        self.d_ff             = config["d_ff"]
        self.dropout_rate     = config["dropout"]
        self.num_classes      = config["num_classes"]
        self.activation       = config.get("activation", "gelu")
        self.use_positional   = config.get("use_positional", True)
        self.positional_type  = config.get("positional_type", "learnable")

        # Named "token_embedding" — must match saved state_dict
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)

        # Named "pos_encoding" — must match saved state_dict
        if self.use_positional and self.positional_type == "learnable":
            self.pos_encoding = nn.Parameter(
                torch.randn(1, self.max_seq_len, self.d_model) * 0.02
            )
        else:
            self.pos_encoding = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_ff,
            dropout=self.dropout_rate,
            activation=self.activation,
            batch_first=True,
            norm_first=True,         # matches training
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layers,
        )

        # Named "classifier" Sequential with LayerNorm then Linear
        # Matches: nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.num_classes),
        )

        self.embed_dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        """
        x: (batch, seq_len, num_features)  — integer token IDs
        Each of num_features token IDs is embedded and summed per position.
        """
        batch_size, seq_len, num_features = x.shape

        # Embed each feature channel and sum
        embedded = torch.zeros(batch_size, seq_len, self.d_model, device=x.device)
        for f in range(num_features):
            token_ids = x[:, :, f].long()
            embedded  = embedded + self.token_embedding(token_ids)

        embedded = self.embed_dropout(embedded)

        # Add learnable positional encoding
        if self.pos_encoding is not None:
            embedded = embedded + self.pos_encoding[:, :seq_len, :]

        # Transformer encoder
        encoded = self.encoder(embedded)   # (batch, seq_len, d_model)

        # Mean pooling over sequence dimension
        pooled = encoded.mean(dim=1)       # (batch, d_model)

        # Classification head
        logits = self.classifier(pooled)   # (batch, num_classes)

        return logits


# ============================================================
# LOAD MODEL
# ============================================================

log.info("Loading trained model...")

model = MarketTransformer(CONFIG)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# Handle both plain state_dict and wrapped checkpoint
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

log.info("Model loaded successfully.")

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)

log.info("Flask server initialized.")


# ============================================================
# HEALTH
# ============================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":               "ok",
        "model":                "MarketTransformer V3",
        "device":               str(DEVICE),
        "vocab_size":           CONFIG["vocab_size"],
        "sequence_length":      CONFIG["max_seq_len"],
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    })


# ============================================================
# PREDICT
# Server receives the (8,8) token sequence from the browser.
# All preprocessing (ZigZag → structure → pressure → momentum
# → tokens) was done in the browser. Server runs inference only.
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)

        if payload is None:
            return jsonify({"error": "No JSON payload received."}), 400

        token_sequence = payload.get("token_sequence")
        asset          = payload.get("asset", "UNKNOWN")
        timeframe      = payload.get("timeframe", 60)

        if token_sequence is None:
            return jsonify({"error": "token_sequence missing."}), 400

        if len(token_sequence) != CONFIG["max_seq_len"]:
            return jsonify({
                "error": f"Expected {CONFIG['max_seq_len']} rows, got {len(token_sequence)}."
            }), 400

        for i, row in enumerate(token_sequence):
            if len(row) != 8:
                return jsonify({
                    "error": f"Row {i} must contain exactly 8 token IDs, got {len(row)}."
                }), 400

        # Shape: (1, seq_len, num_features) = (1, 8, 8)
        x = torch.tensor(
            token_sequence,
            dtype=torch.long,
            device=DEVICE,
        ).unsqueeze(0)

        with torch.no_grad():
            logits        = model(x)
            probabilities = torch.softmax(logits, dim=1)[0]

        p_down = float(probabilities[0])
        p_up   = float(probabilities[1])

        if p_up >= CONFIDENCE_THRESHOLD:
            action     = "BUY"
            confidence = p_up
        elif p_down >= CONFIDENCE_THRESHOLD:
            action     = "SELL"
            confidence = p_down
        else:
            action     = "NO_TRADE"
            confidence = max(p_up, p_down)

        log.info(
            f"Predict | asset={asset} tf={timeframe} "
            f"action={action} conf={confidence:.4f} "
            f"p_up={p_up:.4f} p_down={p_down:.4f}"
        )

        return jsonify({
            "success":    True,
            "asset":      asset,
            "timeframe":  timeframe,
            "action":     action,
            "confidence": round(confidence, 4),
            "p_up":       round(p_up, 4),
            "p_down":     round(p_down, 4),
            "threshold":  CONFIDENCE_THRESHOLD,
        })

    except Exception as e:
        log.error(f"Predict error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================================
# ROOT
# ============================================================

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "i90 Transformer API",
        "status":  "running",
        "version": "4.0",
    })


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log.info("=" * 60)
    log.info("Transformer API ready.")
    log.info(f"Listening on port {port}")
    log.info("=" * 60)
    app.run(host="0.0.0.0", port=port)
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
