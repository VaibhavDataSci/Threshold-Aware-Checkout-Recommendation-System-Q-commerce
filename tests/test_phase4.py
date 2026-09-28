"""
Phase 4 Verification Script.
Validates model architecture, checkpoint loading, forward pass, output probability constraints [0, 1],
and prediction behavior.
"""

import sys
import os
import torch

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.recommendation_model import EmbeddingRecommendationModel

def run_phase4_validation():
    print("==================================================")
    print("RUNNING PHASE 4 MODEL ARCHITECTURE & LOAD VALIDATION")
    print("==================================================")

    model_path = "model/saved_model.pt"
    errors = []

    # 1. Check file existence
    if not os.path.exists(model_path):
        errors.append(f"Model file not found at {model_path}")
        print("FAIL: Model file missing.")
        sys.exit(1)
    print(f"1. Checkpoint File Existence: Found ({model_path})")

    # 2. Load Checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    required_keys = ["model_state_dict", "num_users", "num_products", "user2idx", "prod2idx"]
    for key in required_keys:
        if key not in checkpoint:
            errors.append(f"Missing key '{key}' in saved model checkpoint!")
    print("2. Checkpoint Keys Check: Passed")

    # 3. Instantiate Model and Load State Dict
    num_users = checkpoint['num_users']
    num_products = checkpoint['num_products']
    emb_dim = checkpoint['embedding_dim']
    hid_dim = checkpoint['hidden_dim']

    model = EmbeddingRecommendationModel(
        num_users=num_users,
        num_products=num_products,
        embedding_dim=emb_dim,
        hidden_dim=hid_dim
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("3. Model Re-instantiation & Weights Load: Passed")

    # 4. Single Pair Prediction & Output Bound Check [0, 1]
    score_single = model.predict_pair(0, 0)
    print(f"4. Sample Single Prediction (User 0, Product 0): {score_single:.4f}")
    if not (0.0 <= score_single <= 1.0):
        errors.append(f"Prediction score out of bounds [0, 1]: {score_single}")

    # 5. Batch Inference Check
    u_batch = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    p_batch = torch.tensor([10, 11, 12, 13, 14], dtype=torch.long)
    with torch.no_grad():
        batch_scores = model(u_batch, p_batch).numpy()

    print(f"5. Sample Batch Predictions (5 items): {np.round(batch_scores, 4)}")
    if len(batch_scores) != 5:
        errors.append(f"Expected 5 batch predictions, got {len(batch_scores)}")
    if (batch_scores < 0.0).any() or (batch_scores > 1.0).any():
        errors.append("Batch prediction scores out of bounds [0, 1]!")

    # 6. Check determinism
    score_again = model.predict_pair(0, 0)
    if score_single != score_again:
        errors.append("Inference is non-deterministic!")
    print("6. Determinism Check: Passed (Identical output on repeat call)")

    print("--------------------------------------------------")
    if errors:
        print("PHASE 4 VALIDATION FAILED:")
        for err in errors:
            print(f" - {err}")
        sys.exit(1)
    else:
        print("ALL PHASE 4 VALIDATION CHECKS PASSED SUCCESSFULLY!")
        print("==================================================")

if __name__ == "__main__":
    import numpy as np
    run_phase4_validation()
