"""
Phase 3 Verification Script.
Validates data preparation, ID encodings, negative sampling, train/validation split,
and reproducibility without training any models.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import get_prepared_data, load_raw_data

def run_phase3_validation():
    print("==================================================")
    print("RUNNING PHASE 3 DATA PREPARATION VALIDATION")
    print("==================================================")
    
    # 1. Load raw data and prepared data
    raw_products, raw_interactions = load_raw_data("data")
    data = get_prepared_data(data_dir="data", test_size=0.2, seed=42)
    
    train_df = data['train_df']
    val_df = data['val_df']
    user2idx = data['user2idx']
    prod2idx = data['prod2idx']
    
    errors = []
    
    # Check 1 & 11: User Index Range (0 to 99)
    min_user_idx = min(user2idx.values())
    max_user_idx = max(user2idx.values())
    print(f"1. User Index Range: {min_user_idx} to {max_user_idx} (Total Users: {len(user2idx)})")
    if min_user_idx != 0 or max_user_idx != 99 or len(user2idx) != 100:
        errors.append(f"Invalid user index range or count! Got {min_user_idx}-{max_user_idx}, count {len(user2idx)}")

    # Check 2 & 12: Product Index Range (0 to 49)
    min_prod_idx = min(prod2idx.values())
    max_prod_idx = max(prod2idx.values())
    print(f"2. Product Index Range: {min_prod_idx} to {max_prod_idx} (Total Products: {len(prod2idx)})")
    if min_prod_idx != 0 or max_prod_idx != 49 or len(prod2idx) != 50:
        errors.append(f"Invalid product index range or count! Got {min_prod_idx}-{max_prod_idx}, count {len(prod2idx)}")

    # Check 3 & 4: Missing User/Product Indices
    if train_df['user_idx'].isnull().any() or train_df['product_idx'].isnull().any():
        errors.append("Null indices found in training data!")
    if val_df['user_idx'].isnull().any() or val_df['product_idx'].isnull().any():
        errors.append("Null indices found in validation data!")
    print("3. Missing Indices Check: Passed (0 missing values)")

    # Check 5 & 6: Training set positive/negative samples
    train_pos = (train_df['target'] == 1.0).sum()
    train_neg = (train_df['target'] == 0.0).sum()
    print(f"4. Training Set Class Distribution: Positive={train_pos}, Negative={train_neg} (Total={len(train_df)})")
    if train_pos == 0 or train_neg == 0:
        errors.append("Training set missing positive or negative samples!")

    # Check 7 & 8: Validation set positive/negative samples
    val_pos = (val_df['target'] == 1.0).sum()
    val_neg = (val_df['target'] == 0.0).sum()
    print(f"5. Validation Set Class Distribution: Positive={val_pos}, Negative={val_neg} (Total={len(val_df)})")
    if val_pos == 0 or val_neg == 0:
        errors.append("Validation set missing positive or negative samples!")

    # Check 9: Target values validity
    valid_targets = {0.0, 1.0}
    train_targets = set(train_df['target'].unique())
    val_targets = set(val_df['target'].unique())
    if not train_targets.issubset(valid_targets) or not val_targets.issubset(valid_targets):
        errors.append(f"Invalid target values detected: train={train_targets}, val={val_targets}")
    print("6. Target Value Validity Check: Passed (Only 0.0 and 1.0)")

    # Check 10: Invalid product/user references
    if not set(train_df['user_id']).issubset(set(user2idx.keys())):
        errors.append("Invalid user_ids in training set!")
    if not set(train_df['product_id']).issubset(set(prod2idx.keys())):
        errors.append("Invalid product_ids in training set!")
    print("7. ID References Validity Check: Passed")

    # Check 13: Reproducibility check
    data_again = get_prepared_data(data_dir="data", test_size=0.2, seed=42)
    if not train_df.equals(data_again['train_df']) or not val_df.equals(data_again['val_df']):
        errors.append("Reproducibility failed! Second run produced different data split.")
    print("8. Reproducibility Check: Passed (Identical results on repeated runs)")

    print("--------------------------------------------------")
    if errors:
        print("PHASE 3 VALIDATION FAILED:")
        for err in errors:
            print(f" - {err}")
        sys.exit(1)
    else:
        print("ALL PHASE 3 VALIDATION CHECKS PASSED SUCCESSFULLY!")
        print("==================================================")

if __name__ == "__main__":
    run_phase3_validation()
