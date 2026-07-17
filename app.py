import os
import json
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS

# =====================================================
# CONFIG
# =====================================================

MODEL_PATH = "models/best_transformer_model.pt"
CONFIG_PATH = "models/model_configuration.json"
TOKEN_PATH = "tokens/token_dictionary.json"
THRESHOLD_PATH = "models/optimized_thresholds.json"

DEVICE = torch.device("cpu")

# =====================================================
# LOAD CONFIG
# =====================================================

with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

with open(TOKEN_PATH, "r") as f:
    TOKEN_DICT = json.load(f)

with open(THRESHOLD_PATH, "r") as f:
    THRESHOLD = json.load(f)

CONFIDENCE_THRESHOLD = float(
    THRESHOLD.get("optimal_threshold", 0.55)
)

# =====================================================
# MODEL
# =====================================================

class TransformerClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            CONFIG["vocab_size"],
            CONFIG["d_model"]
        )

        self.position = nn.Parameter(
            torch.randn(
                1,
                CONFIG["max_seq_len"],
                CONFIG["d_model"]
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=CONFIG["d_model"],
            nhead=CONFIG["n_heads"],
            dim_feedforward=CONFIG["d_ff"],
            dropout=CONFIG["dropout"],
            activation=CONFIG.get("activation", "gelu"),
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            CONFIG["n_layers"]
        )

        self.classifier = nn.Linear(
            CONFIG["d_model"],
            CONFIG["num_classes"]
        )

    def forward(self, x):

        x = self.embedding(x)

        x = x + self.position[:, :x.size(1), :]

        x = self.encoder(x)

        x = x.mean(dim=1)

        return self.classifier(x)

# =====================================================
# LOAD MODEL
# =====================================================

model = TransformerClassifier()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("Transformer loaded successfully.")

# =====================================================
# APP
# =====================================================

app = Flask(__name__)
CORS(app)

# =====================================================
# HEALTH
# =====================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "model": "Transformer V3",
        "threshold": CONFIDENCE_THRESHOLD
    })

# =====================================================
# PREDICT
# =====================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        payload = request.get_json(force=True)

        token_sequence = payload.get("token_sequence")

        if token_sequence is None:

            return jsonify({
                "error": "token_sequence missing"
            }), 400

        x = torch.tensor(
            token_sequence,
            dtype=torch.long
        ).unsqueeze(0)

        with torch.no_grad():

            logits = model(x)

            probs = torch.softmax(
                logits,
                dim=1
            )[0]

        p_down = float(probs[0])
        p_up = float(probs[1])

        if p_up >= CONFIDENCE_THRESHOLD:

            action = "BUY"
            confidence = p_up

        elif p_down >= CONFIDENCE_THRESHOLD:

            action = "SELL"
            confidence = p_down

        else:

            action = "NO_TRADE"
            confidence = max(
                p_up,
                p_down
            )

        return jsonify({

            "action": action,

            "confidence": round(
                confidence,
                4
            ),

            "p_up": round(
                p_up,
                4
            ),

            "p_down": round(
                p_down,
                4
            )

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

# =====================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
