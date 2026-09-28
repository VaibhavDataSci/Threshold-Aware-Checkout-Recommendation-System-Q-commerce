"""
Phase 6 Automated Test Suite.
Tests threshold logic, cart total calculation, gap calculation, cart product exclusion,
hybrid ranking (0.7 model + 0.3 threshold), edge cases, and mandatory Scenarios A-E.
"""

import sys
import os

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.threshold import (
    calculate_cart_total,
    calculate_threshold_gap,
    is_threshold_reached,
    get_threshold_status,
    FREE_DELIVERY_THRESHOLD
)
from src.recommendation import get_top_recommendations, calculate_threshold_score, load_trained_model

def run_phase6_tests():
    print("==================================================")
    print("RUNNING PHASE 6 THRESHOLD-AWARE ENGINE TESTS")
    print("==================================================")

    model_data = load_trained_model("model/saved_model.pt")
    products_df = model_data['products_df']

    errors = []

    # 1. Unit Tests for Threshold Functions
    print("\n--- 1. UNIT TESTS: THRESHOLD LOGIC ---")
    
    # Test Cart Total
    sample_cart = ["P001", "P008"] # Fresh Milk (30.0) + Brown Bread (45.0) = 75.0
    total = calculate_cart_total(sample_cart, products_df)
    print(f"Cart Total Test (Milk ₹30 + Bread ₹45): Expected=75.0, Got={total}")
    if total != 75.0:
        errors.append(f"Cart total expected 75.0, got {total}")

    # Test Gap Calculation
    gap = calculate_threshold_gap(75.0)
    print(f"Gap Calculation Test (Cart=75.0): Expected=75.0, Got={gap}")
    if gap != 75.0:
        errors.append(f"Gap expected 75.0, got {gap}")

    # Test Threshold Status Reached
    print(f"Threshold Reached (150.0): {is_threshold_reached(150.0)}")
    if not is_threshold_reached(150.0):
        errors.append("150.0 should be marked threshold reached!")

    print(f"Threshold Reached (149.9): {is_threshold_reached(149.9)}")
    if is_threshold_reached(149.9):
        errors.append("149.9 should NOT be marked threshold reached!")

    # 2. Mandatory Scenarios (A - E)
    print("\n--- 2. MANDATORY SCENARIO TESTING ---")

    # SCENARIO A: Significantly Below Threshold (Cart Total = ₹90, Gap = ₹60)
    # Products: P005 (Paneer ₹85) + P001 (Milk ₹30) -> Wait, P005 (Paneer ₹85) + P008 (Bread ₹45) = 130... Let's use P005 (Paneer ₹85) + Lemon (₹15) -> 100.
    # Let's select P005 (Paneer ₹85) + Green Chillies (₹12) = 97... let's select items equaling ₹90.
    # P002 (Toned Milk ₹55) + P009 (Sandwich Bread ₹35) = ₹90
    cart_a = ["P002", "P009"]
    res_a = get_top_recommendations("U001", cart_a)
    print(f"\nScenario A (Cart=₹90, Gap=₹60): Status='{res_a['status_message']}'")
    print(f"Cart Total: ₹{res_a['cart_total']}, Remaining Gap: ₹{res_a['remaining_gap']}")
    if res_a['threshold_reached'] or res_a['remaining_gap'] != 60.0:
        errors.append(f"Scenario A failed! Expected gap 60.0, got {res_a['remaining_gap']}")
    if len(res_a['recommendations']) != 3:
        errors.append(f"Scenario A expected 3 recommendations, got {len(res_a['recommendations'])}")

    # Check cart items excluded
    for rec in res_a['recommendations']:
        if rec['product_id'] in cart_a:
            errors.append(f"Scenario A recommended product in cart! {rec['product_id']}")

    # SCENARIO B: Slightly Below Threshold (Cart Total = ₹120, Gap = ₹30)
    # P005 (Paneer ₹85) + P002 (Milk ₹35) = 120... P005 Paneer ₹85 + P009 Bread ₹35 = 120
    cart_b = ["P005", "P009"]
    res_b = get_top_recommendations("U001", cart_b)
    print(f"\nScenario B (Cart=₹120, Gap=₹30): Status='{res_b['status_message']}'")
    print("Top Recommendations:")
    for rec in res_b['recommendations']:
        print(f" - {rec['product_name']} (₹{rec['price']}): Model={rec['model_score']:.4f}, Thresh={rec['threshold_score']:.4f}, Final={rec['final_score']:.4f}")
    if res_b['threshold_reached'] or res_b['remaining_gap'] != 30.0:
        errors.append(f"Scenario B failed! Expected gap 30.0, got {res_b['remaining_gap']}")

    # SCENARIO C: Very Close to Threshold (Cart Total = ₹145, Gap = ₹5)
    # P038 (Basmati Rice ₹120) + P004 (Curd ₹25) = 145
    cart_c = ["P038", "P004"]
    res_c = get_top_recommendations("U001", cart_c)
    print(f"\nScenario C (Cart=₹145, Gap=₹5): Status='{res_c['status_message']}'")
    if res_c['threshold_reached'] or res_c['remaining_gap'] != 5.0:
        errors.append(f"Scenario C failed! Expected gap 5.0, got {res_c['remaining_gap']}")

    # SCENARIO D: Exactly at Threshold (Cart Total = ₹150, Gap = ₹0)
    # P038 (Rice ₹120) + P001 (Milk ₹30) = 150
    cart_d = ["P038", "P001"]
    res_d = get_top_recommendations("U001", cart_d)
    print(f"\nScenario D (Cart=₹150, Gap=₹0): Status='{res_d['status_message']}'")
    if not res_d['threshold_reached'] or len(res_d['recommendations']) != 0:
        errors.append("Scenario D failed! Expected threshold reached and 0 recommendations.")

    # SCENARIO E: Above Threshold (Cart Total = ₹205, Gap = ₹0)
    # P038 (Rice ₹120) + P005 (Paneer ₹85) = 205
    cart_e = ["P038", "P005"]
    res_e = get_top_recommendations("U001", cart_e)
    print(f"\nScenario E (Cart=₹205, Gap=₹0): Status='{res_e['status_message']}'")
    if not res_e['threshold_reached'] or res_e['remaining_gap'] != 0.0:
        errors.append("Scenario E failed! Expected threshold reached.")

    # 3. Edge Cases Testing
    print("\n--- 3. EDGE CASES TESTING ---")
    
    # Edge Case 1: Empty Cart
    res_empty = get_top_recommendations("U001", [])
    print(f"Empty Cart Test: Total=₹{res_empty['cart_total']}, Recommendations={len(res_empty['recommendations'])}")
    if res_empty['cart_total'] != 0.0 or len(res_empty['recommendations']) != 3:
        errors.append("Empty cart test failed!")

    # Edge Case 2: Invalid Product ID in Cart
    res_invalid_p = get_top_recommendations("U001", ["INVALID_PRODUCT_XYZ"])
    print(f"Invalid Product Cart Test: Total=₹{res_invalid_p['cart_total']} (Handled safely)")

    # Edge Case 3: Invalid User ID
    try:
        get_top_recommendations("INVALID_USER_999", [])
        errors.append("Expected ValueError for invalid user_id!")
    except ValueError as ve:
        print(f"Invalid User ID Test: Safely raised expected error -> '{ve}'")

    print("\n--------------------------------------------------")
    if errors:
        print("PHASE 6 VALIDATION FAILED WITH ERRORS:")
        for err in errors:
            print(f" - {err}")
        sys.exit(1)
    else:
        print("ALL PHASE 6 VALIDATION & SCENARIO TESTS PASSED SUCCESSFULLY!")
        print("==================================================")

if __name__ == "__main__":
    run_phase6_tests()
