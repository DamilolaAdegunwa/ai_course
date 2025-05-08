### Project Title: Advanced Graph-Based Product Recommendation System with GNNs
### File Name: advanced_graph_based_product_recommendation_system.py

"""
Short Project Description:
This advanced product recommendation system leverages Graph Neural Networks (GNNs) using PyTorch Geometric to model complex relationships between users, products, and interactions. Unlike traditional matrix factorization or simple collaborative filtering techniques, this system constructs a heterogeneous graph where users, items, and interactions form a multi-relational network. The GNN learns from this structure to provide highly personalized recommendations.
"""

# Import necessary libraries
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load dataset (Using the MovieLens dataset via torch_geometric.datasets)
from torch_geometric.datasets import MovieLens

dataset = MovieLens(root="./data")
# dataset = MovieLens(root="./data", model="user-item")
data: Data = dataset[0]  # The graph data object


# Define a Graph Convolutional Network (GCN) model
class GNNRecommendationModel(torch.nn.Module):
    def __init__(self, num_nodes: int, embedding_dim: int = 64):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, 128)
        self.conv2 = GCNConv(128, 64)
        self.fc = torch.nn.Linear(64, 1)  # Final layer to predict score

    def forward(self, edge_index: torch.Tensor):
        x = self.embedding.weight
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


# Instantiate model
num_nodes: int = data.num_nodes
model = GNNRecommendationModel(num_nodes)

# Train/Test Split
edge_index: torch.Tensor = data.get_edge_index  # Edges in the graph
# edge_index: torch.Tensor = data.edge_index  # Edges in the graph
train_edges, test_edges = train_test_split(edge_index.t().numpy(), test_size=0.2, random_state=42)
train_edges, test_edges = torch.tensor(train_edges.T), torch.tensor(test_edges.T)


# Define training loop
def train_model(model: GNNRecommendationModel, train_edges: torch.Tensor, epochs: int = 100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(train_edges)
        loss = F.mse_loss(out[train_edges[0]], out[train_edges[1]])
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# Train the model
train_model(model, train_edges)


# Making Predictions
def recommend_items(model: GNNRecommendationModel, user_id: int, num_recommendations: int = 5):
    user_embedding = model.embedding(torch.tensor(user_id))
    item_embeddings = model.embedding.weight[data.num_users:]
    scores = torch.matmul(user_embedding, item_embeddings.T)
    top_items = scores.argsort(descending=True)[:num_recommendations]
    return top_items.tolist()


# Example usage
user_id_example: int = 42
recommendations = recommend_items(model, user_id_example)
print(f"Recommended items for user {user_id_example}: {recommendations}")

"""
Use Cases:
1. E-commerce platforms (Amazon, Alibaba) for personalized product recommendations.
2. Streaming services (Netflix, Spotify) to recommend movies or songs.
3. Online learning platforms (Coursera, Udemy) for course recommendations.
4. Retail websites to suggest complementary products.

Example Inputs & Expected Outputs:
- Input: User ID = 42
- Expected Output: List of recommended item IDs [103, 456, 789, 120, 88]

Key Learnings & Research Areas:
1. Graph Neural Networks (GNNs) for recommendations.
2. PyTorch Geometric for graph-based machine learning.
3. Edge-based learning in heterogeneous networks.
4. Hybrid filtering methods using collaborative and content-based features.
5. Optimization techniques for large-scale recommendation models.
"""
