"""
Phase 7 Verification Script.
Validates Streamlit backend integration, deterministic initial cart loading,
cart state updates, threshold unlock state, and recommendation engine integration.
"""

import sys
import os

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def run_phase7_validation():
    print("==================================================")
    print("RUNNING PHASE 7 STREAMLIT APP INTEGRATION TESTS")
    print("==================================================")

    errors = []

    # 1. Test Dataset Loading
    products_df, interactions_df = load_raw_data("data")
    if products_df.empty or interactions_df.empty:
        errors.append("Dataset load returned empty DataFrames!")
    print("1. Dataset Loading Test: Passed")

    # 2. Test Deterministic Initial Cart for U001 (< ₹150) and U100 (>= ₹150)
    cart_u1 = get_initial_cart("U001", products_df, interactions_df)
    total_u1 = calculate_cart_total(cart_u1, products_df)
    print(f"2a. Initial Cart for U001: Items={cart_u1}, Total=₹{total_u1:.2f}")
    if total_u1 >= FREE_DELIVERY_THRESHOLD:
        errors.append(f"U001 initial cart should be < ₹150, got ₹{total_u1:.2f}")

    cart_u100 = get_initial_cart("U100", products_df, interactions_df)
    total_u100 = calculate_cart_total(cart_u100, products_df)
    print(f"2b. Initial Cart for U100: Items={cart_u100}, Total=₹{total_u100:.2f}")
    if total_u100 < FREE_DELIVERY_THRESHOLD:
        errors.append(f"U100 initial cart should be >= ₹150, got ₹{total_u100:.2f}")

    # 3. Test Recommendations Call for U001
    rec_res = get_top_recommendations("U001", cart_u1, top_k=3)
    recs = rec_res.get("recommendations", [])
    print(f"3. Recommendation Integration Test: Top {len(recs)} items returned.")
    if len(recs) != 3:
        errors.append(f"Expected 3 recommendations for U001, got {len(recs)}")

    # 4. Test Add-to-Cart State Simulation
    top_rec_pid = recs[0]['product_id']
    updated_cart = cart_u1 + [top_rec_pid]
    updated_total = calculate_cart_total(updated_cart, products_df)
    print(f"4. Add-to-Cart State Update Test: Added {top_rec_pid}, New Total=₹{updated_total:.2f}")
    if updated_total <= total_u1:
        errors.append("Adding product should increase cart total!")

    # 5. Test Threshold Unlocked State (U100)
    rec_u100 = get_top_recommendations("U100", cart_u100, top_k=3)
    print(f"5. Threshold Unlocked Test (U100): Reached={rec_u100['threshold_reached']}, Recommendations={len(rec_u100['recommendations'])}")
    if not rec_u100['threshold_reached'] or len(rec_u100['recommendations']) != 0:
        errors.append("Unlocked cart should return 0 recommendations and threshold_reached=True!")

    # 6. Test App Module Import
    try:
        import app
        print("6. Streamlit app.py Import Test: Passed")
    except Exception as imp_err:
        errors.append(f"Failed to import app.py: {imp_err}")

    print("--------------------------------------------------")
    if errors:
        print("PHASE 7 VALIDATION FAILED WITH ERRORS:")
        for err in errors:
            print(f" - {err}")
        sys.exit(1)
    else:
        print("ALL PHASE 7 STREAMLIT INTEGRATION TESTS PASSED SUCCESSFULLY!")
        print("==================================================")

if __name__ == "__main__":
    run_phase7_validation()
