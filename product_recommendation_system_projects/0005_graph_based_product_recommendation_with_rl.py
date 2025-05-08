import inputs
import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE
import pandas as pd
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
import gym
env = gym.make("CartPole-v1")

# Load dataset from a remote source
df: pd.DataFrame = pd.read_csv("https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv")

# Encoding users and products
user_encoder: LabelEncoder = LabelEncoder()
product_encoder: LabelEncoder = LabelEncoder()
df["user_id"] = user_encoder.fit_transform(df["user_id"])
df["book_id"] = product_encoder.fit_transform(df["book_id"])

# start the rl
ray.shutdown()  # Ensure any previous Ray instance is stopped
ray.init(ignore_reinit_error=True)  # Start a new Ray instance

# Creating graph edges
edge_index: torch.Tensor = torch.tensor(df[["user_id", "book_id"]].values.T, dtype=torch.long)

# Node Features (Random embeddings)
num_nodes: int = max(df["user_id"].max(), df["book_id"].max()) + 1
print(f"number of nodes: {num_nodes}")
print(f"user_id max + 1: {df["user_id"].max() + 1}")
print(f"book_id max + 1: {df["book_id"].max() + 1}")
node_features: torch.Tensor = torch.rand((num_nodes, 128), dtype=torch.float32)

# Graph Data
graph_data: Data = Data(x=node_features, edge_index=edge_index, num_nodes=num_nodes)


# Graph Neural Network Model
class GNNRecommendationModel(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super(GNNRecommendationModel, self).__init__()
        self.conv1 = GraphSAGE(in_channels, hidden_channels, num_layers=2)
        self.conv2 = GraphSAGE(hidden_channels, out_channels, num_layers=2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# Model Training
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model: GNNRecommendationModel = GNNRecommendationModel(128, 64, 32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

device = torch.device("cpu")
model.to(device)
# inputs, targets = inputs.to(device), targets.to(device)


# Training Loop
def train_gnn(epochs: int = 10):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out: torch.Tensor = model(graph_data.x.to(device), graph_data.edge_index.to(device))
        # target = torch.randint(0, out.shape[0], (out.shape[0],)).to(device)
        print(f"out.shape[0], {out.shape[0]}")
        print(f"out.shape[1], {out.shape[1]}")
        print(f"out.shape, {out.shape}")

        target = torch.randint(0, 4, (53424,)).to(device)
        # target = torch.randint(0, 53424, (53424,)).to(device)
        print(f"target (in the train_gnn), {target}")
        print(f"target (size), {target.size()}")
        print(f"out (in the train_gnn), {out}")
        print(f"out (size), {out.size()}")

        loss = F.nll_loss(out, target)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


# Reinforcement Learning with PPO

# old config
_old_config = '''
config = (
    PPO.get_default_config()
    .framework("torch")
    .training(lr=0.001, train_batch_size=2000)
    .environment("CartPole-v1")
    .experimental(_validate_config=False)
)
trainer = PPO(config=config)
'''
config = (
    PPOConfig()
    .framework("torch")
    # .rollouts(num_rollout_workers=1)  # Ensure at least 1 rollout worker
    .env_runners(num_env_runners=1)  # Use this instead
    .training(lr=0.001, train_batch_size=2000)
    .environment(env="CartPole-v1")  # Ensure environment is correctly defined
)
trainer = config.build()


def train_rl(epochs: int = 10):
    for i in range(epochs):
        result = trainer.train()
        print(f"RL Iteration {i}, Reward: {result}")


# Generate Recommendations
def get_recommendations(user_id: int) -> List[str]:
    user_idx: int = user_encoder.transform([user_id])[0]
    user_embedding: torch.Tensor = model(graph_data.x.to(device), graph_data.edge_index.to(device))[user_idx]
    scores: torch.Tensor = torch.matmul(graph_data.x, user_embedding)
    top_items: torch.Tensor = torch.topk(scores, 3).indices.cpu().numpy()
    return product_encoder.inverse_transform(top_items)


# Example Usage
train_gnn(epochs=5)
train_rl(epochs=5)
user_id: int = 1025
print(f"Recommended Products for User {user_id}: {get_recommendations(user_id)}")
ray.shutdown()
