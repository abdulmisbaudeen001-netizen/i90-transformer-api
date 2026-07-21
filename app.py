import os
import json
import logging
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log.info("Starting i90 Transformer API...")
log.info(f"Running on {DEVICE}")

MODEL_PATH     = "best_transformer_model.pt"
CONFIG_PATH    = "model_configuration.json"
TOKEN_PATH     = "token_dictionary.json"
THRESHOLD_PATH = "optimized_thresholds.json"

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


class MarketTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.d_model         = config["d_model"]
        self.max_seq_len     = config["max_seq_len"]
        self.use_positional  = config.get("use_positional", True)
        self.positional_type = config.get("positional_type", "learnable")
        self.num_classes     = config["num_classes"]
        self.dropout_rate    = config["dropout"]

        self.token_embedding = nn.Embedding(
            config["vocab_size"],
            config["d_model"]
        )

        if self.use_positional and self.positional_type == "learnable":
            self.pos_encoding = nn.Parameter(
                torch.randn(1, config["max_seq_len"], config["d_model"]) * 0.02
            )
        else:
            self.pos_encoding = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["d_model"],
            nhead=config["n_heads"],
            dim_feedforward=config["d_ff"],
            dropout=config["dropout"],
            activation=config.get("activation", "gelu"),
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config["n_layers"]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(config["d_model"]),
            nn.Linear(config["d_model"], config["num_classes"])
        )

        self.embed_dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        batch_size, seq_len, num_features = x.shape
        embedded = torch.zeros(batch_size, seq_len, self.d_model, device=x.device)
        for f in range(num_features):
            embedded = embedded + self.token_embedding(x[:, :, f].long())
        embedded = self.embed_dropout(embedded)
        if self.pos_encoding is not None:
            embedded = embedded + self.pos_encoding[:, :seq_len, :]
        encoded = self.encoder(embedded)
        pooled  = encoded.mean(dim=1)
        return self.classifier(pooled)


log.info("Loading trained model...")
log.info(f"Model path: {MODEL_PATH}")
log.info(f"File exists: {os.path.exists(MODEL_PATH)}")
log.info(f"File size: {os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 'N/A'} bytes")

model = MarketTransformer(CONFIG)
log.info("MarketTransformer architecture instantiated.")

# weights_only=False required for PyTorch >= 2.0 with legacy checkpoints.
# Without this, torch.load may hang or raise UnpicklingError on CPU.
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
log.info("Checkpoint loaded from disk.")

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
    log.info("State dict loaded from checkpoint dict.")
else:
    model.load_state_dict(checkpoint)
    log.info("State dict loaded directly.")

model.to(DEVICE)
model.eval()
log.info("Model loaded successfully.")

app = Flask(__name__)
CORS(app)

log.info("Flask server initialized.")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":               "ok",
        "model":                "MarketTransformer V3",
        "device":               str(DEVICE),
        "vocab_size":           CONFIG["vocab_size"],
        "sequence_length":      CONFIG["max_seq_len"],
        "confidence_threshold": CONFIDENCE_THRESHOLD
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # force=True: parses JSON regardless of Content-Type (handles text/plain from extension)
        payload = request.get_json(force=True)

        if payload is None:
            return jsonify({"error": "No JSON payload received."}), 400

        token_sequence = payload.get("token_sequence")
        asset          = payload.get("asset", "UNKNOWN")
        timeframe      = payload.get("timeframe", 60)

        if token_sequence is None:
            return jsonify({"error": "token_sequence missing."}), 400

        if len(token_sequence) != CONFIG["max_seq_len"]:
            return jsonify({"error": f"Expected {CONFIG['max_seq_len']} rows, got {len(token_sequence)}."}), 400

        for i, row in enumerate(token_sequence):
            if len(row) != 8:
                return jsonify({"error": f"Row {i} must have 8 token IDs, got {len(row)}."}), 400

        x = torch.tensor(
            token_sequence,
            dtype=torch.long,
            device=DEVICE
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

        log.info(f"asset={asset} action={action} conf={confidence:.4f} p_up={p_up:.4f} p_down={p_down:.4f}")

        return jsonify({
            "success":    True,
            "asset":      asset,
            "timeframe":  timeframe,
            "action":     action,
            "confidence": round(confidence, 4),
            "p_up":       round(p_up, 4),
            "p_down":     round(p_down, 4),
            "threshold":  CONFIDENCE_THRESHOLD
        })

    except Exception as e:
        log.error(f"Predict error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "i90 Transformer API",
        "status":  "running",
        "version": "4.0"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log.info(f"Listening on port {port}")
    app.run(host="0.0.0.0", port=port)
        
