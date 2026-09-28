"""
Neural Recommendation Model Architecture.
Implements an embedding-based deep learning recommendation network using PyTorch.
Architecture:
User ID -> User Embedding
Product ID -> Product Embedding
Concat -> Dense(hidden_dim) -> ReLU -> Dense(1) -> Sigmoid -> Relevance Score [0, 1]
"""

import torch
import torch.nn as nn

class EmbeddingRecommendationModel(nn.Module):
    """
    Simplified Embedding-based Neural Recommendation Model for Quick-Commerce.
    Learns user and product vector representations and predicts relevance score in [0, 1].
    """
    def __init__(self, num_users, num_products, embedding_dim=16, hidden_dim=32):
        super(EmbeddingRecommendationModel, self).__init__()
        self.num_users = num_users
        self.num_products = num_products
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Embedding layers
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        self.product_embedding = nn.Embedding(num_embeddings=num_products, embedding_dim=embedding_dim)

        # Fully connected dense layers
        self.fc1 = nn.Linear(in_features=embedding_dim * 2, out_features=hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_idx, product_idx):
        """
        Forward pass.
        :param user_idx: Tensor of user integer indices [batch_size]
        :param product_idx: Tensor of product integer indices [batch_size]
        :return: Tensor of predicted relevance scores [batch_size] in range [0, 1]
        """
        user_emb = self.user_embedding(user_idx)
        prod_emb = self.product_embedding(product_idx)

        # Concatenate user and product embeddings along feature dimension
        x = torch.cat([user_emb, prod_emb], dim=-1)

        # Dense layer 1 + ReLU activation
        x = self.fc1(x)
        x = self.relu(x)

        # Dense layer 2 + Sigmoid activation
        x = self.fc2(x)
        scores = self.sigmoid(x)

        return scores.squeeze(-1)

    def predict_pair(self, user_idx_int, product_idx_int):
        """Helper method to predict relevance for a single user-product pair."""
        self.eval()
        with torch.no_grad():
            u_tensor = torch.tensor([user_idx_int], dtype=torch.long)
            p_tensor = torch.tensor([product_idx_int], dtype=torch.long)
            score = self.forward(u_tensor, p_tensor).item()
        return score
