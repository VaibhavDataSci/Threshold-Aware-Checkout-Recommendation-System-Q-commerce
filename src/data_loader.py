"""
Data Loader & Preprocessing Module.
Handles loading raw CSV files, encoding user/product IDs into integer indices,
generating reproducible negative samples, splitting into train/validation sets,
and serving PyTorch-compatible datasets while preserving product metadata.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

SEED = 42

def set_seed(seed=SEED):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_raw_data(data_dir="data"):
    """
    Loads products.csv and interactions.csv from data_dir.
    Returns products_df, interactions_df.
    """
    products_path = os.path.join(data_dir, "products.csv")
    interactions_path = os.path.join(data_dir, "interactions.csv")
    
    if not os.path.exists(products_path):
        raise FileNotFoundError(f"Products file not found at {products_path}")
    if not os.path.exists(interactions_path):
        raise FileNotFoundError(f"Interactions file not found at {interactions_path}")
        
    products_df = pd.read_csv(products_path)
    interactions_df = pd.read_csv(interactions_path)
    
    return products_df, interactions_df

def build_id_mappings(products_df, interactions_df):
    """
    Builds bidirectional integer index mappings for users and products.
    Sorting ensures strict mapping reproducibility.
    """
    unique_users = sorted(interactions_df['user_id'].unique())
    unique_prods = sorted(products_df['product_id'].unique())
    
    user2idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    idx2user = {idx: user_id for user_id, idx in user2idx.items()}
    
    prod2idx = {prod_id: idx for idx, prod_id in enumerate(unique_prods)}
    idx2prod = {idx: prod_id for prod_id, idx in prod2idx.items()}
    
    return user2idx, idx2user, prod2idx, idx2prod

def create_recommendation_samples(interactions_df, products_df, user2idx, prod2idx, negative_ratio=1.0, seed=SEED):
    """
    Creates positive and negative recommendation samples.
    - Positive sample (target = 1): user purchased product.
    - Negative sample (target = 0): user did NOT purchase product (sampled 1:1).
    """
    set_seed(seed)
    
    # 1. Extract positive purchase interactions
    purchases = interactions_df[interactions_df['interaction'] == 'purchase'].copy()
    positive_pairs = set(zip(purchases['user_id'], purchases['product_id']))
    
    samples = []
    
    # Add positive samples
    for user_id, prod_id in positive_pairs:
        if user_id in user2idx and prod_id in prod2idx:
            samples.append({
                "user_id": user_id,
                "product_id": prod_id,
                "user_idx": user2idx[user_id],
                "product_idx": prod2idx[prod_id],
                "target": 1.0
            })
            
    num_positives = len(samples)
    
    # 2. Negative sampling per user
    all_user_ids = sorted(user2idx.keys())
    all_product_ids = sorted(prod2idx.keys())
    
    user_purchased_dict = purchases.groupby('user_id')['product_id'].apply(set).to_dict()
    
    negative_samples = []
    for user_id in all_user_ids:
        purchased_prods = user_purchased_dict.get(user_id, set())
        unpurchased_prods = list(set(all_product_ids) - purchased_prods)
        
        # Calculate number of negatives to sample for this user
        user_pos_count = len(purchased_prods)
        num_neg_to_sample = int(round(user_pos_count * negative_ratio))
        num_neg_to_sample = max(1, min(num_neg_to_sample, len(unpurchased_prods)))
        
        sampled_negs = random.sample(unpurchased_prods, num_neg_to_sample)
        for neg_pid in sampled_negs:
            negative_samples.append({
                "user_id": user_id,
                "product_id": neg_pid,
                "user_idx": user2idx[user_id],
                "product_idx": prod2idx[neg_pid],
                "target": 0.0
            })
            
    # Combine positive and negative samples
    all_samples = samples + negative_samples
    samples_df = pd.DataFrame(all_samples)
    
    # Shuffle dataset reproducibly
    samples_df = samples_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return samples_df

def split_train_val(samples_df, test_size=0.2, seed=SEED):
    """
    Splits samples into train and validation sets with target stratification.
    """
    train_df, val_df = train_test_split(
        samples_df,
        test_size=test_size,
        random_state=seed,
        stratify=samples_df['target']
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

class RecommendationDataset(Dataset):
    """PyTorch Dataset wrapper for recommendation samples."""
    def __init__(self, samples_df):
        self.user_tensor = torch.tensor(samples_df['user_idx'].values, dtype=torch.long)
        self.product_tensor = torch.tensor(samples_df['product_idx'].values, dtype=torch.long)
        self.target_tensor = torch.tensor(samples_df['target'].values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.user_tensor)
        
    def __getitem__(self, idx):
        return {
            "user_idx": self.user_tensor[idx],
            "product_idx": self.product_tensor[idx],
            "target": self.target_tensor[idx]
        }

def get_prepared_data(data_dir="data", test_size=0.2, negative_ratio=1.0, seed=SEED):
    """
    Master data loading and preparation function.
    Returns dictionary containing:
    - train_df, val_df
    - train_dataset, val_dataset
    - user2idx, idx2user
    - prod2idx, idx2prod
    - products_df (metadata preserved)
    """
    set_seed(seed)
    products_df, interactions_df = load_raw_data(data_dir=data_dir)
    user2idx, idx2user, prod2idx, idx2prod = build_id_mappings(products_df, interactions_df)
    
    samples_df = create_recommendation_samples(
        interactions_df, products_df, user2idx, prod2idx, negative_ratio=negative_ratio, seed=seed
    )
    
    train_df, val_df = split_train_val(samples_df, test_size=test_size, seed=seed)
    
    train_dataset = RecommendationDataset(train_df)
    val_dataset = RecommendationDataset(val_df)
    
    return {
        "train_df": train_df,
        "val_df": val_df,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "user2idx": user2idx,
        "idx2user": idx2user,
        "prod2idx": prod2idx,
        "idx2prod": idx2prod,
        "products_df": products_df,
        "num_users": len(user2idx),
        "num_products": len(prod2idx)
    }

if __name__ == "__main__":
    # Test script execution
    prepared = get_prepared_data()
    print("Prepared Data Summary:")
    print(f"Num Users: {prepared['num_users']}")
    print(f"Num Products: {prepared['num_products']}")
    print(f"Train samples: {len(prepared['train_df'])}")
    print(f"Val samples: {len(prepared['val_df'])}")
