"""
Phase 5 Evaluation Script.
Evaluates trained EmbeddingRecommendationModel on the validation dataset.
Calculates Accuracy, Precision, Recall, F1 Score, ROC-AUC, Confusion Matrix,
runs Top-5 sanity checks for representative personas, and analyzes loss patterns.
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import get_prepared_data, load_raw_data
from model.recommendation_model import EmbeddingRecommendationModel

def evaluate_model():
    print("==================================================")
    print("RUNNING PHASE 5 MODEL EVALUATION")
    print("==================================================")

    model_path = "model/saved_model.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # 1. Load Checkpoint & Re-instantiate Model
    checkpoint = torch.load(model_path, map_location="cpu")
    num_users = checkpoint['num_users']
    num_products = checkpoint['num_products']
    user2idx = checkpoint['user2idx']
    idx2user = checkpoint['idx2user']
    prod2idx = checkpoint['prod2idx']
    idx2prod = checkpoint['idx2prod']
    history = checkpoint['history']

    model = EmbeddingRecommendationModel(
        num_users=num_users,
        num_products=num_products,
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['hidden_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Load Validation Data & Raw Data
    prepared_data = get_prepared_data(data_dir="data", test_size=0.2, seed=42)
    val_df = prepared_data['val_df']
    products_df, interactions_df = load_raw_data("data")

    # 3. Generate Validation Predictions
    val_user_tensor = torch.tensor(val_df['user_idx'].values, dtype=torch.long)
    val_prod_tensor = torch.tensor(val_df['product_idx'].values, dtype=torch.long)
    true_targets = val_df['target'].values

    with torch.no_grad():
        predicted_probs = model(val_user_tensor, val_prod_tensor).numpy()

    # Probability threshold = 0.5 for classification
    binary_preds = (predicted_probs >= 0.5).astype(float)

    # 4. Compute Classification Metrics
    acc = accuracy_score(true_targets, binary_preds)
    prec = precision_score(true_targets, binary_preds, zero_division=0)
    rec = recall_score(true_targets, binary_preds, zero_division=0)
    f1 = f1_score(true_targets, binary_preds, zero_division=0)
    try:
        roc_auc = roc_auc_score(true_targets, predicted_probs)
    except Exception as e:
        roc_auc = None

    cm = confusion_matrix(true_targets, binary_preds)
    tn, fp, fn, tp = cm.ravel()

    print(f"Validation Samples: {len(val_df)} (Positives: {(true_targets==1).sum()}, Negatives: {(true_targets==0).sum()})")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}" if roc_auc else "ROC-AUC: N/A")
    print("\nConfusion Matrix:")
    print(f"  TP: {tp:3d} | FP: {fp:3d}")
    print(f"  FN: {fn:3d} | TN: {tn:3d}")
    print("--------------------------------------------------")

    # 5. Top-5 Recommendation Sanity Check for Representative Personas
    representative_users = [
        ("U005", "Dairy & Bakery Persona"),
        ("U025", "Fruits & Vegetables Persona"),
        ("U045", "Snacks & Beverages Persona"),
        ("U065", "Staples & Household Persona"),
        ("U085", "Personal Care & Household Persona"),
        ("U095", "Mixed Preferences Persona")
    ]

    print("\nTOP-5 RECOMMENDATION SANITY CHECKS:")
    
    # Pre-build historical purchase lookup dict
    user_purchased_dict = interactions_df[interactions_df['interaction'] == 'purchase'].groupby('user_id')['product_id'].apply(set).to_dict()

    rep_results = []
    for uid, persona in representative_users:
        if uid not in user2idx:
            continue
            
        u_idx = user2idx[uid]
        purchased_pids = user_purchased_dict.get(uid, set())

        # Score all 50 products for this user
        all_prod_ids = [idx2prod[p] for p in range(num_products)]
        all_prod_indices = torch.tensor(list(range(num_products)), dtype=torch.long)
        user_indices = torch.tensor([u_idx] * num_products, dtype=torch.long)

        with torch.no_grad():
            scores = model(user_indices, all_prod_indices).numpy()

        # Combine into DataFrame & sort descending by model score
        scored_prods = products_df.copy()
        scored_prods['score'] = scores
        scored_prods['previously_purchased'] = scored_prods['product_id'].apply(lambda pid: "Yes" if pid in purchased_pids else "No")
        scored_prods = scored_prods.sort_values(by='score', ascending=False).reset_index(drop=True)

        top5 = scored_prods.head(5)
        print(f"\nUser: {uid} ({persona}) | Total Historical Purchases: {len(purchased_pids)}")
        print(top5[['product_name', 'category', 'price', 'score', 'previously_purchased']].to_string(index=False))

        rep_results.append({
            "user_id": uid,
            "persona": persona,
            "top5": top5
        })

    # Save summary report text
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "phase5_evaluation.md")

    with open(report_path, "w") as f:
        f.write("# Phase 5 Model Evaluation Results\n\n")
        f.write("## Classification Metrics\n")
        f.write(f"- **Accuracy:** {acc:.4f}\n")
        f.write(f"- **Precision:** {prec:.4f}\n")
        f.write(f"- **Recall:** {rec:.4f}\n")
        f.write(f"- **F1 Score:** {f1:.4f}\n")
        f.write(f"- **ROC-AUC:** {roc_auc:.4f}\n\n")
        f.write("## Confusion Matrix\n")
        f.write(f"- True Positives (TP): {tp}\n")
        f.write(f"- True Negatives (TN): {tn}\n")
        f.write(f"- False Positives (FP): {fp}\n")
        f.write(f"- False Negatives (FN): {fn}\n\n")
        f.write("## Loss Progression\n")
        f.write(f"- Epoch 1 Train Loss: {history['train_loss'][0]:.4f} | Val Loss: {history['val_loss'][0]:.4f}\n")
        f.write(f"- Epoch 20 Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {history['val_loss'][-1]:.4f}\n")

    print(f"\nSaved evaluation report artifact to: {report_path}")
    print("==================================================")
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "history": history,
        "rep_results": rep_results
    }

if __name__ == "__main__":
    evaluate_model()
