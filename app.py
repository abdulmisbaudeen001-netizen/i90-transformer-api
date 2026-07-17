"""
==========================================================
i90 Transformer Prediction Server
Production Version 1.0

Receives:
    token_sequence (8 × 8 integer matrix)

Returns:
    BUY
    SELL
    NO_TRADE

Framework:
    Flask
    PyTorch

Author:
    i90 Transformer Engine

==========================================================
"""

import os
import json
import logging

from typing import Any
from typing import Dict

import torch
import torch.nn as nn

from flask import Flask
from flask import jsonify
from flask import request

from flask_cors import CORS

############################################################
# LOGGING
############################################################

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s"

)

logger = logging.getLogger("TransformerAPI")

############################################################
# DEVICE
############################################################

DEVICE = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"

)

logger.info(f"Running on {DEVICE}")

############################################################
# PATHS
############################################################

BASE_DIR = os.path.dirname(

    os.path.abspath(__file__)

)

MODEL_PATH = os.path.join(

    BASE_DIR,

    "best_transformer_model.pt"

)

CONFIG_PATH = os.path.join(

    BASE_DIR,

    "model_configuration.json"

)

TOKEN_PATH = os.path.join(

    BASE_DIR,

    "token_dictionary.json"

)

THRESHOLD_PATH = os.path.join(

    BASE_DIR,

    "optimized_thresholds.json"

)

############################################################
# LOAD JSON
############################################################

def load_json(path: str):

    with open(path, "r") as f:

        return json.load(f)

CONFIG = load_json(CONFIG_PATH)

TOKEN_DICTIONARY = load_json(TOKEN_PATH)

THRESHOLDS = load_json(THRESHOLD_PATH)

CONFIDENCE_THRESHOLD = float(

    THRESHOLDS.get(

        "optimal_threshold",

        0.55

    )

)

logger.info("Configuration Loaded")

logger.info(

    f"Vocabulary Size : {CONFIG['vocab_size']}"

)

logger.info(

    f"Sequence Length : {CONFIG['max_seq_len']}"

)

logger.info(

    f"Confidence Threshold : {CONFIDENCE_THRESHOLD}"

)
############################################################
# TRANSFORMER MODEL
############################################################

class TransformerClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Token Embedding
        # ----------------------------------------------------

        self.embedding = nn.Embedding(

            num_embeddings=CONFIG["vocab_size"],

            embedding_dim=CONFIG["d_model"]

        )

        # ----------------------------------------------------
        # Learnable Positional Embedding
        # ----------------------------------------------------

        self.position_embedding = nn.Parameter(

            torch.randn(

                1,

                CONFIG["max_seq_len"],

                CONFIG["d_model"]

            )

        )

        # ----------------------------------------------------
        # Transformer Encoder Layer
        # ----------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(

            d_model=CONFIG["d_model"],

            nhead=CONFIG["n_heads"],

            dim_feedforward=CONFIG["d_ff"],

            dropout=CONFIG["dropout"],

            activation=CONFIG.get(

                "activation",

                "gelu"

            ),

            batch_first=True,

            norm_first=True

        )

        # ----------------------------------------------------
        # Transformer Encoder
        # ----------------------------------------------------

        self.encoder = nn.TransformerEncoder(

            encoder_layer,

            num_layers=CONFIG["n_layers"]

        )

        # ----------------------------------------------------
        # Layer Normalization
        # ----------------------------------------------------

        self.norm = nn.LayerNorm(

            CONFIG["d_model"]

        )

        # ----------------------------------------------------
        # Classification Head
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(

                CONFIG["d_model"],

                CONFIG["d_model"] // 2

            ),

            nn.GELU(),

            nn.Dropout(

                CONFIG["dropout"]

            ),

            nn.Linear(

                CONFIG["d_model"] // 2,

                CONFIG["num_classes"]

            )

        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x):

        # x shape:
        # (batch, seq_len)

        x = self.embedding(x)

        x = x + self.position_embedding[:, :x.size(1), :]

        x = self.encoder(x)

        x = self.norm(x)

        # Mean Pooling

        x = torch.mean(

            x,

            dim=1

        )

        logits = self.classifier(x)

        return logits
