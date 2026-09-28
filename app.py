"""
Streamlit Web Application Entry Point.
Provides an interactive web interface for the Threshold-Aware Checkout Recommendation System.
Reuses existing backend modules in src/ & model/ without duplicating logic.
"""

import sys
import os
import streamlit as st
import pandas as pd

# Add root directory to sys.path
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

# Page Configuration
st.set_page_config(
    page_title="Threshold-Aware Checkout Recommendation System",
    page_icon="🛒",
    layout="centered"
)

# Cache raw data loading to optimize Streamlit performance
@st.cache_data
def get_cached_raw_data():
    return load_raw_data("data")

try:
    products_df, interactions_df = get_cached_raw_data()
    all_users = sorted(interactions_df['user_id'].unique().tolist())
except Exception as e:
    st.error(f"Error loading project datasets: {e}")
    st.stop()

# Streamlit Session State Initialization
if "selected_user" not in st.session_state:
    st.session_state.selected_user = all_users[0]

if "cart_product_ids" not in st.session_state:
    st.session_state.cart_product_ids = get_initial_cart(
        st.session_state.selected_user, products_df, interactions_df
    )

# Header Section
st.title("🛒 Threshold-Aware Checkout Recommendation System")
st.caption("AI-powered personalized product recommendations to help customers reach the ₹150 free-delivery threshold.")

st.markdown("---")

# Section 1: Customer Selection
col_user, col_reset = st.columns([3, 1])

with col_user:
    selected_user_input = st.selectbox(
        "Select Customer / User Profile:",
        options=all_users,
        index=all_users.index(st.session_state.selected_user) if st.session_state.selected_user in all_users else 0
    )

# Handle user profile switching
if selected_user_input != st.session_state.selected_user:
    st.session_state.selected_user = selected_user_input
    st.session_state.cart_product_ids = get_initial_cart(
        selected_user_input, products_df, interactions_df
    )
    st.rerun()

with col_reset:
    st.write("") # Alignment spacing
    if st.button("Reset Cart", use_container_width=True, help="Reset cart to default state"):
        st.session_state.cart_product_ids = get_initial_cart(
            st.session_state.selected_user, products_df, interactions_df
        )
        st.rerun()

# Section 2: Current Cart Display
st.subheader("🛍️ Your Shopping Cart")

current_cart_ids = st.session_state.cart_product_ids

if not current_cart_ids:
    st.info("Your cart is currently empty.")
else:
    price_dict = products_df.set_index('product_id')['price'].to_dict()
    name_dict = products_df.set_index('product_id')['product_name'].to_dict()
    cat_dict = products_df.set_index('product_id')['category'].to_dict()

    for idx, pid in enumerate(current_cart_ids):
        pname = name_dict.get(pid, pid)
        pcat = cat_dict.get(pid, "N/A")
        pprice = price_dict.get(pid, 0.0)

        c1, c2, c3, c4 = st.columns([3, 2, 2, 1])
        with c1:
            st.write(f"**{pname}**")
        with c2:
            st.caption(f"Category: {pcat}")
        with c3:
            st.write(f"₹{pprice:.2f}")
        with c4:
            if st.button("❌", key=f"remove_{pid}_{idx}", help=f"Remove {pname}"):
                st.session_state.cart_product_ids.pop(idx)
                st.rerun()

st.markdown("---")

# Section 3: Checkout & Threshold Summary
cart_total = calculate_cart_total(current_cart_ids, products_df)
gap = calculate_threshold_gap(cart_total, threshold=FREE_DELIVERY_THRESHOLD)
threshold_unlocked = is_threshold_reached(cart_total, threshold=FREE_DELIVERY_THRESHOLD)
status_msg = get_threshold_status(cart_total, threshold=FREE_DELIVERY_THRESHOLD)

st.subheader("💳 Checkout Summary")

col_t1, col_t2, col_t3 = st.columns(3)
col_t1.metric("Cart Total", f"₹{cart_total:.2f}")
col_t2.metric("Free Delivery Threshold", f"₹{FREE_DELIVERY_THRESHOLD:.2f}")
col_t3.metric("Remaining Gap", f"₹{gap:.2f}" if not threshold_unlocked else "₹0.00")

if threshold_unlocked:
    st.success("🎉 **FREE DELIVERY UNLOCKED!** You have reached the ₹150 threshold.")
else:
    st.warning(f"🚚 **{status_msg}**")

st.markdown("---")

# Section 4: AI Recommendations
if not threshold_unlocked:
    st.subheader("🤖 AI Checkout Recommendations")
    st.caption("Personalized low-cost items recommended by our deep learning model to help you reach free delivery.")

    try:
        rec_data = get_top_recommendations(
            user_id=st.session_state.selected_user,
            cart_product_ids=current_cart_ids,
            top_k=3
        )
        recommendations = rec_data.get("recommendations", [])
    except Exception as rec_err:
        st.error(f"Error generating AI recommendations: {rec_err}")
        recommendations = []

    if not recommendations:
        st.info("No eligible recommendations available.")
    else:
        rec_cols = st.columns(len(recommendations))
        for col, rec in zip(rec_cols, recommendations):
            with col:
                st.markdown(f"### {rec['product_name']}")
                st.caption(f"Category: **{rec['category']}**")
                st.markdown(f"#### **₹{rec['price']:.2f}**")
                st.write(f"🎯 **AI Relevance:** `{rec['model_score'] * 100:.1f}%`")
                st.info(f"💡 *Helps close ₹{gap:.2f} gap*")

                if st.button(f"➕ Add to Cart", key=f"add_{rec['product_id']}", use_container_width=True):
                    st.session_state.cart_product_ids.append(rec['product_id'])
                    st.rerun()
else:
    st.info("No further recommendations needed as free delivery has been unlocked! Proceed to final checkout.")

st.markdown("---")

# Optional Expandable Section: Prototype Details
with st.expander("ℹ️ About this Prototype"):
    st.markdown("""
    **Threshold-Aware Checkout Recommendation System (Academic MVP)**
    - **Domain:** Quick Commerce / E-commerce
    - **Fixed Business Rule:** ₹150 Free Delivery Threshold
    - **Deep Learning Component:** PyTorch Embedding Neural Network ($0.7 \times \text{Model Relevance} + 0.3 \times \text{Threshold Gap Score}$)
    - **Dataset:** 100 Synthetic Users, 50 Products, 1,832 Interactions
    """)
