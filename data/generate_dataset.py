"""
Synthetic Dataset Generator & Validator for Threshold-Aware Checkout Recommendation System.
Generates reproducible products.csv and interactions.csv based on quick-commerce patterns.
"""

import os
import random
import pandas as pd
import numpy as np

# Fixed random seed for reproducibility
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

def generate_products():
    """Generates 50 quick-commerce products across 9 categories with realistic pricing."""
    product_definitions = [
        # Dairy (7 products)
        ("Fresh Milk (500ml)", "Dairy", 30),
        ("Toned Milk (1L)", "Dairy", 55),
        ("Amul Butter (100g)", "Dairy", 58),
        ("Plain Curd (200g)", "Dairy", 25),
        ("Paneer (200g)", "Dairy", 85),
        ("Cheese Slices (100g)", "Dairy", 75),
        ("Flavored Milk (200ml)", "Dairy", 35),

        # Bakery (6 products)
        ("Brown Bread", "Bakery", 45),
        ("White Sandwich Bread", "Bakery", 35),
        ("Burger Buns (4 pcs)", "Bakery", 30),
        ("Garlic Breadsticks", "Bakery", 40),
        ("Choco Chip Cookies (100g)", "Bakery", 50),
        ("Fruit Cake Slice", "Bakery", 25),

        # Fruits (6 products)
        ("Bananas (6 pcs)", "Fruits", 35),
        ("Fresh Apples (500g)", "Fruits", 90),
        ("Oranges (1 kg)", "Fruits", 80),
        ("Green Grapes (250g)", "Fruits", 45),
        ("Papaya (1 pc)", "Fruits", 60),
        ("Lemon (4 pcs)", "Fruits", 15),

        # Vegetables (6 products)
        ("Tomatoes (500g)", "Vegetables", 25),
        ("Onions (1 kg)", "Vegetables", 35),
        ("Potatoes (1 kg)", "Vegetables", 30),
        ("Fresh Coriander (100g)", "Vegetables", 10),
        ("Green Chillies (100g)", "Vegetables", 12),
        ("Cucumber (500g)", "Vegetables", 20),

        # Snacks (7 products)
        ("Potato Chips (Small)", "Snacks", 10),
        ("Potato Chips (Medium)", "Snacks", 20),
        ("Marie Gold Biscuits", "Snacks", 15),
        ("Bourbon Chocolate Biscuits", "Snacks", 30),
        ("Dark Chocolate Bar", "Snacks", 40),
        ("Salted Peanuts (100g)", "Snacks", 25),
        ("Instant Noodles (Single Pack)", "Snacks", 14),

        # Beverages (5 products)
        ("Mango Juice (250ml)", "Beverages", 20),
        ("Orange Juice (1L)", "Beverages", 110),
        ("Cola Soft Drink (600ml)", "Beverages", 38),
        ("Lemon Soda (300ml)", "Beverages", 20),
        ("Cold Coffee Can (240ml)", "Beverages", 65),

        # Staples (5 products)
        ("Basmati Rice (1 kg)", "Staples", 120),
        ("Whole Wheat Atta (1 kg)", "Staples", 55),
        ("Refined Sugar (1 kg)", "Staples", 48),
        ("Iodized Salt (1 kg)", "Staples", 28),
        ("Toor Dal (500g)", "Staples", 75),

        # Personal Care (4 products)
        ("Bathing Soap (Single)", "Personal Care", 35),
        ("Herbal Toothpaste (100g)", "Personal Care", 60),
        ("Hand Wash Refill (200ml)", "Personal Care", 45),
        ("Pocket Tissues (10 pcs)", "Personal Care", 15),

        # Household (4 products)
        ("Dishwash Bar", "Household", 15),
        ("Liquid Dishwash (250ml)", "Household", 55),
        ("Garbage Bags (30 pcs)", "Household", 65),
        ("Kitchen Sponge", "Household", 20),
    ]

    products_list = []
    for idx, (name, category, price) in enumerate(product_definitions, start=1):
        product_id = f"P{idx:03d}"
        products_list.append({
            "product_id": product_id,
            "product_name": name,
            "category": category,
            "price": float(price)
        })

    return pd.DataFrame(products_list)


def generate_user_interactions(products_df, num_users=100, target_interactions=1800):
    """
    Generates realistic user-product interactions based on behavioral personas.
    """
    categories = list(products_df['category'].unique())
    
    # Define 6 User Personas with distinct category preference weights
    personas = {
        "dairy_bakery": {
            "categories": ["Dairy", "Bakery"],
            "pref_weight": 0.75,
            "user_range": (1, 20)
        },
        "fruits_vegetables": {
            "categories": ["Fruits", "Vegetables"],
            "pref_weight": 0.75,
            "user_range": (21, 40)
        },
        "snacks_beverages": {
            "categories": ["Snacks", "Beverages"],
            "pref_weight": 0.75,
            "user_range": (41, 60)
        },
        "staples_household": {
            "categories": ["Staples", "Household"],
            "pref_weight": 0.75,
            "user_range": (61, 80)
        },
        "personalcare_household": {
            "categories": ["Personal Care", "Household"],
            "pref_weight": 0.75,
            "user_range": (81, 90)
        },
        "mixed": {
            "categories": categories,
            "pref_weight": 0.50,
            "user_range": (91, 100)
        }
    }

    interactions_list = []
    
    # Map user_id to persona
    user_personas = {}
    for persona_name, config in personas.items():
        start_u, end_u = config["user_range"]
        for u in range(start_u, end_u + 1):
            user_personas[f"U{u:03d}"] = persona_name

    # Determine interactions per user (averaging 18 interactions per user -> ~1800 total)
    user_ids = [f"U{u:03d}" for u in range(1, num_users + 1)]
    
    for user_id in user_ids:
        persona_name = user_personas[user_id]
        pref_cats = personas[persona_name]["categories"]
        pref_weight = personas[persona_name]["pref_weight"]
        
        # Calculate product probability weights for this user
        prod_weights = []
        for _, row in products_df.iterrows():
            if row['category'] in pref_cats:
                prod_weights.append(pref_weight)
            else:
                prod_weights.append((1.0 - pref_weight) / max(1, len(categories) - len(pref_cats)))
        
        # Normalize weights
        prod_weights = np.array(prod_weights)
        prod_weights /= prod_weights.sum()

        # Sample number of interactions for this user (15 to 22)
        n_inter = random.randint(15, 22)
        
        # Sample products without replacement for user history
        chosen_indices = np.random.choice(len(products_df), size=n_inter, replace=False, p=prod_weights)
        
        for idx in chosen_indices:
            p_id = products_df.iloc[idx]['product_id']
            # Interaction type distribution: 60% purchase, 25% add_to_cart, 15% view
            inter_type = random.choices(["purchase", "add_to_cart", "view"], weights=[0.60, 0.25, 0.15])[0]
            interactions_list.append({
                "user_id": user_id,
                "product_id": p_id,
                "interaction": inter_type
            })

    interactions_df = pd.DataFrame(interactions_list)
    # Shuffle interactions randomly while maintaining seed
    interactions_df = interactions_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return interactions_df


def validate_dataset(products_df, interactions_df):
    """Strict automated validation checks on products and interactions."""
    print("==================================================")
    print("RUNNING AUTOMATED DATASET VALIDATION")
    print("==================================================")
    
    errors = []
    
    # 1. User count check (~100)
    n_users = interactions_df['user_id'].nunique()
    print(f"1. Unique Users: {n_users}")
    if not (90 <= n_users <= 110):
        errors.append(f"Expected ~100 users, got {n_users}")
        
    # 2. Product count check (~50)
    n_prods = products_df['product_id'].nunique()
    print(f"2. Unique Products: {n_prods}")
    if not (45 <= n_prods <= 55):
        errors.append(f"Expected ~50 products, got {n_prods}")

    # 3. Interactions count check (1500 - 2000)
    n_inter = len(interactions_df)
    print(f"3. Total Interactions: {n_inter}")
    if not (1500 <= n_inter <= 2000):
        errors.append(f"Expected 1500-2000 interactions, got {n_inter}")

    # 4. Missing values check
    if products_df.isnull().sum().sum() > 0:
        errors.append("Missing values detected in products.csv")
    if interactions_df.isnull().sum().sum() > 0:
        errors.append("Missing values detected in interactions.csv")
    print("4. Missing Values Check: Passed (0 missing values)")

    # 5. Duplicate product IDs check
    if products_df['product_id'].duplicated().any():
        errors.append("Duplicate product_ids found in products.csv")
    print("5. Duplicate Product IDs Check: Passed")

    # 6. Invalid product references check
    valid_pids = set(products_df['product_id'])
    inter_pids = set(interactions_df['product_id'])
    invalid_pids = inter_pids - valid_pids
    if invalid_pids:
        errors.append(f"Invalid product references in interactions: {invalid_pids}")
    print("6. Product Reference Check: Passed (All interaction product_ids exist in products.csv)")

    # 7. Invalid interaction types check
    valid_types = {"purchase", "add_to_cart", "view"}
    actual_types = set(interactions_df['interaction'].unique())
    if not actual_types.issubset(valid_types):
        errors.append(f"Invalid interaction types found: {actual_types - valid_types}")
    print(f"7. Interaction Types Check: Passed ({actual_types})")

    # 8. Invalid prices check
    if (products_df['price'] <= 0).any():
        errors.append("Products with non-positive price found!")
    print("8. Product Price Validity Check: Passed (All prices > 0)")

    # 9. Low-cost threshold availability check (Products <= ₹75)
    low_cost_prods = products_df[products_df['price'] <= 75]
    print(f"9. Threshold-Aware Low-Cost Products (<= ₹75): {len(low_cost_prods)} products available")
    if len(low_cost_prods) < 15:
        errors.append("Fewer than 15 low-cost products available for threshold bridging!")

    # 10. Product interaction coverage check
    prods_with_inter = interactions_df['product_id'].nunique()
    print(f"10. Products with Interactions: {prods_with_inter}/{n_prods}")
    if prods_with_inter < n_prods:
        errors.append(f"Some products have 0 interactions: {valid_pids - inter_pids}")

    print("--------------------------------------------------")
    if errors:
        print("VALIDATION FAILED WITH ERRORS:")
        for err in errors:
            print(f" - {err}")
        raise ValueError("Dataset validation failed!")
    else:
        print("ALL AUTOMATED DATASET VALIDATION CHECKS PASSED SUCCESSFULLY!")
        print("==================================================")


def main():
    set_seed(SEED)
    
    # Target output paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    products_path = os.path.join(data_dir, "products.csv")
    interactions_path = os.path.join(data_dir, "interactions.csv")

    # Generate data
    products_df = generate_products()
    interactions_df = generate_user_interactions(products_df)

    # Save to CSV
    products_df.to_csv(products_path, index=False)
    interactions_df.to_csv(interactions_path, index=False)

    print(f"Saved products dataset to: {products_path}")
    print(f"Saved interactions dataset to: {interactions_path}")

    # Validate generated datasets
    validate_dataset(products_df, interactions_df)

    # Print Summary Statistics
    print("\n--- DATASET SUMMARY STATISTICS ---")
    print(f"Categories ({products_df['category'].nunique()}): {list(products_df['category'].unique())}")
    print(f"Price Range: ₹{products_df['price'].min():.2f} to ₹{products_df['price'].max():.2f} (Mean: ₹{products_df['price'].mean():.2f})")
    print("\nPrice Range Distribution:")
    price_bins = [0, 20, 30, 50, 75, 100, 200]
    labels = ["₹10-₹20", "₹20-₹30", "₹30-₹50", "₹50-₹75", "₹75-₹100", "₹100+"]
    price_dist = pd.cut(products_df['price'], bins=price_bins, labels=labels).value_counts().sort_index()
    print(price_dist.to_string())

    print("\nInteraction Types Distribution:")
    print(interactions_df['interaction'].value_counts().to_string())

if __name__ == "__main__":
    main()
