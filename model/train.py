"""
Model Training & Evaluation Script.
Loads preprocessed recommendation data, initializes EmbeddingRecommendationModel,
trains neural network using PyTorch, evaluates on validation set, and saves model weights.
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import get_prepared_data
from model.recommendation_model import EmbeddingRecommendationModel

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model(
    data_dir="data",
    model_save_path="model/saved_model.pt",
    epochs=20,
    batch_size=32,
    lr=0.005,
    embedding_dim=16,
    hidden_dim=32,
    seed=SEED
):
    set_seed(seed)
    print("==================================================")
    print("STARTING DEEP LEARNING MODEL TRAINING (PHASE 4)")
    print("==================================================")

    # 1. Load prepared data
    data = get_prepared_data(data_dir=data_dir, test_size=0.2, seed=seed)
    train_dataset = data['train_dataset']
    val_dataset = data['val_dataset']

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 2. Instantiate Neural Network Model
    model = EmbeddingRecommendationModel(
        num_users=data['num_users'],
        num_products=data['num_products'],
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim
    )

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    history = {
        "train_loss": [],
        "val_loss": []
    }

    print(f"Model Architecture: {model}")
    print(f"Total Users: {data['num_users']}, Total Products: {data['num_products']}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Training for {epochs} epochs...")
    print("--------------------------------------------------")

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_train_loss = 0.0
        for batch in train_loader:
            user_idx = batch['user_idx']
            product_idx = batch['product_idx']
            targets = batch['target']

            optimizer.zero_grad()
            preds = model(user_idx, product_idx)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * len(targets)

        epoch_train_loss = running_train_loss / len(train_dataset)

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                user_idx = batch['user_idx']
                product_idx = batch['product_idx']
                targets = batch['target']

                preds = model(user_idx, product_idx)
                loss = criterion(preds, targets)
                running_val_loss += loss.item() * len(targets)

        epoch_val_loss = running_val_loss / len(val_dataset)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)

        if epoch % 2 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

    print("--------------------------------------------------")
    print("TRAINING COMPLETED SUCCESSFULLY!")

    # 3. Save Model & Metadata Checkpoint
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_users": data['num_users'],
        "num_products": data['num_products'],
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "user2idx": data['user2idx'],
        "idx2user": data['idx2user'],
        "prod2idx": data['prod2idx'],
        "idx2prod": data['idx2prod'],
        "history": history
    }

    torch.save(checkpoint, model_save_path)
    print(f"Saved trained model checkpoint to: {model_save_path}")
    print("==================================================")

    return model, checkpoint, history

if __name__ == "__main__":
    train_model()
