from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_raw_data
from src.threshold import (
    FREE_DELIVERY_THRESHOLD,
    calculate_cart_total,
    calculate_threshold_gap,
    is_threshold_reached,
    get_threshold_status,
    get_initial_cart
)
from src.recommendation import get_top_recommendations

app = FastAPI(title="Threshold-Aware Checkout API")

# Add CORS middleware for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data at startup
try:
    products_df, interactions_df = load_raw_data("data")
    all_users = sorted(interactions_df['user_id'].unique().tolist())
except Exception as e:
    print(f"Error loading data: {e}")
    products_df, interactions_df = None, None
    all_users = []

class CartRequest(BaseModel):
    user_id: str
    cart_product_ids: List[str]

@app.get("/api/users")
def get_users():
    return {"users": all_users}

@app.get("/api/user/{user_id}/initial-cart")
def get_user_initial_cart(user_id: str):
    if user_id not in all_users:
        raise HTTPException(status_code=404, detail="User not found")
    cart = get_initial_cart(user_id, products_df, interactions_df)
    return {"cart_product_ids": cart}

@app.post("/api/cart/details")
def get_cart_details(request: CartRequest):
    cart_ids = request.cart_product_ids
    
    # Get product details
    price_dict = products_df.set_index('product_id')['price'].to_dict()
    name_dict = products_df.set_index('product_id')['product_name'].to_dict()
    cat_dict = products_df.set_index('product_id')['category'].to_dict()
    
    items = []
    for pid in cart_ids:
        items.append({
            "product_id": pid,
            "product_name": name_dict.get(pid, pid),
            "category": cat_dict.get(pid, "N/A"),
            "price": price_dict.get(pid, 0.0)
        })
        
    cart_total = calculate_cart_total(cart_ids, products_df)
    gap = calculate_threshold_gap(cart_total, threshold=FREE_DELIVERY_THRESHOLD)
    threshold_unlocked = is_threshold_reached(cart_total, threshold=FREE_DELIVERY_THRESHOLD)
    status_msg = get_threshold_status(cart_total, threshold=FREE_DELIVERY_THRESHOLD)
    
    return {
        "items": items,
        "summary": {
            "cart_total": cart_total,
            "free_delivery_threshold": FREE_DELIVERY_THRESHOLD,
            "remaining_gap": gap if not threshold_unlocked else 0.0,
            "threshold_unlocked": threshold_unlocked,
            "status_message": status_msg
        }
    }

@app.post("/api/recommendations")
def get_recommendations(request: CartRequest):
    try:
        rec_data = get_top_recommendations(
            user_id=request.user_id,
            cart_product_ids=request.cart_product_ids,
            top_k=3
        )
        return {"recommendations": rec_data.get("recommendations", [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
