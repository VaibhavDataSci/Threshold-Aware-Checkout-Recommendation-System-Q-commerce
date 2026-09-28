"""
Recommendation Engine & Hybrid Ranking Module.
Connects trained neural recommendation model with threshold-aware candidate filtering and ranking logic.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.recommendation_model import EmbeddingRecommendationModel
from src.data_loader import load_raw_data
from src.threshold import (
    FREE_DELIVERY_THRESHOLD,
    calculate_cart_total,
    calculate_threshold_gap,
    is_threshold_reached,
    get_threshold_status
)

def load_trained_model(model_path="model/saved_model.pt"):
    """
    Loads saved PyTorch recommendation model checkpoint and metadata.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")
    num_users = checkpoint['num_users']
    num_products = checkpoint['num_products']

    model = EmbeddingRecommendationModel(
        num_users=num_users,
        num_products=num_products,
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['hidden_dim']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    products_df, _ = load_raw_data("data")

    return {
        "model": model,
        "user2idx": checkpoint['user2idx'],
        "idx2user": checkpoint['idx2user'],
        "prod2idx": checkpoint['prod2idx'],
        "idx2prod": checkpoint['idx2prod'],
        "products_df": products_df
    }

def calculate_threshold_score(product_price, gap):
    """
    Calculates threshold gap relevance score [0, 1].
    Products closer to the remaining threshold gap receive higher scores.
    :param product_price: Price of candidate product in INR.
    :param gap: Remaining threshold gap in INR.
    :return: Float score in range [0.0, 1.0].
    """
    if gap <= 0:
        return 0.0

    price_diff = abs(float(product_price) - float(gap))
    # Normalized score: 1.0 when price exactly equals gap
    score = 1.0 - (price_diff / max(float(gap), 1.0))
    return max(0.0, min(1.0, float(score)))

def get_top_recommendations(
    user_id,
    cart_product_ids=None,
    top_k=3,
    model_path="model/saved_model.pt",
    model_weight=0.7,
    threshold_weight=0.3
):
    """
    Generates top threshold-aware personalized product recommendations for a customer checkout session.
    
    :param user_id: String user ID (e.g., 'U001').
    :param cart_product_ids: List of product IDs currently in customer's cart.
    :param top_k: Number of recommendations to return (default: 3).
    :param model_path: Path to saved PyTorch model checkpoint.
    :param model_weight: Weight for neural model relevance score (default: 0.7).
    :param threshold_weight: Weight for threshold gap score (default: 0.3).
    :return: Dict containing session summary, cart total, remaining gap, threshold status, and recommendations.
    """
    if cart_product_ids is None:
        cart_product_ids = []

    # 1. Load Model & Metadata
    model_data = load_trained_model(model_path=model_path)
    model = model_data['model']
    user2idx = model_data['user2idx']
    prod2idx = model_data['prod2idx']
    idx2prod = model_data['idx2prod']
    products_df = model_data['products_df']

    # Validate User ID
    if user_id not in user2idx:
        raise ValueError(f"User ID '{user_id}' not found in model user index mapping.")

    # 2. Calculate Cart Total & Threshold Status
    cart_total = calculate_cart_total(cart_product_ids, products_df)
    threshold_reached = is_threshold_reached(cart_total, threshold=FREE_DELIVERY_THRESHOLD)
    gap = calculate_threshold_gap(cart_total, threshold=FREE_DELIVERY_THRESHOLD)
    status_msg = get_threshold_status(cart_total, threshold=FREE_DELIVERY_THRESHOLD)

    # 3. Handle Threshold Reached Case
    if threshold_reached:
        return {
            "user_id": user_id,
            "cart_total": cart_total,
            "free_delivery_threshold": FREE_DELIVERY_THRESHOLD,
            "remaining_gap": 0.0,
            "threshold_reached": True,
            "status_message": status_msg,
            "recommendations": []
        }

    # 4. Candidate Generation (Exclude products currently in cart)
    cart_set = set(cart_product_ids)
    candidate_rows = products_df[~products_df['product_id'].isin(cart_set)].copy()

    if candidate_rows.empty:
        return {
            "user_id": user_id,
            "cart_total": cart_total,
            "free_delivery_threshold": FREE_DELIVERY_THRESHOLD,
            "remaining_gap": gap,
            "threshold_reached": False,
            "status_message": status_msg,
            "recommendations": []
        }

    # 5. Compute Neural Model Scores
    u_idx = user2idx[user_id]
    cand_pids = candidate_rows['product_id'].values
    cand_indices = [prod2idx[pid] for pid in cand_pids if pid in prod2idx]

    user_tensor = torch.tensor([u_idx] * len(cand_indices), dtype=torch.long)
    prod_tensor = torch.tensor(cand_indices, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        model_scores = model(user_tensor, prod_tensor).numpy()

    # Map scores back to candidate DataFrame
    pid_to_model_score = {idx2prod[idx]: score for idx, score in zip(cand_indices, model_scores)}
    candidate_rows['model_score'] = candidate_rows['product_id'].map(pid_to_model_score).fillna(0.0)

    # 6. Compute Threshold Relevance Scores & Hybrid Final Score
    candidate_rows['threshold_score'] = candidate_rows['price'].apply(
        lambda p: calculate_threshold_score(p, gap)
    )

    candidate_rows['final_score'] = (
        model_weight * candidate_rows['model_score'] +
        threshold_weight * candidate_rows['threshold_score']
    )

    # 7. Rank candidates by final_score descending
    ranked_candidates = candidate_rows.sort_values(by='final_score', ascending=False).reset_index(drop=True)

    # Select top_k recommendations
    top_candidates = ranked_candidates.head(top_k)

    recommendations = []
    for _, row in top_candidates.iterrows():
        recommendations.append({
            "product_id": row['product_id'],
            "product_name": row['product_name'],
            "category": row['category'],
            "price": float(row['price']),
            "model_score": float(round(row['model_score'], 4)),
            "threshold_score": float(round(row['threshold_score'], 4)),
            "final_score": float(round(row['final_score'], 4))
        })

    return {
        "user_id": user_id,
        "cart_total": cart_total,
        "free_delivery_threshold": FREE_DELIVERY_THRESHOLD,
        "remaining_gap": gap,
        "threshold_reached": False,
        "status_message": status_msg,
        "recommendations": recommendations
    }
