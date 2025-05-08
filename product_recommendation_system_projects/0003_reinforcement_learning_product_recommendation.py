import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

# Load Dataset (MovieLens dataset)
reader: Reader = Reader(line_format="user item rating timestamp", sep="\t")
data_url: str = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"  # download it and save as a file
data_file: str = r"C:\Users\damil\PycharmProjects\ai_course\data\files_grouplens_org_datasets_movielens_ml_100k_u.data.txt"
data = Dataset.load_from_file(data_file, reader=reader)
trainset, testset = train_test_split(data, test_size=0.2)


# Define Deep Q-Network
class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Define RL Agent
class RecommendationAgent:
    def __init__(self, state_size: int, action_size: int):
        self.state_size: int = state_size
        self.action_size: int = action_size
        self.memory: deque = deque(maxlen=2000)
        self.gamma: float = 0.95
        self.epsilon: float = 1.0
        self.epsilon_decay: float = 0.995
        self.epsilon_min: float = 0.01
        self.learning_rate: float = 0.001
        self.model: DQN = DQN(state_size, action_size)
        self.optimizer: optim.Adam = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion: nn.MSELoss = nn.MSELoss()

    def remember(self, state: np.array, action: int, reward: float, next_state: np.array, done: bool):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.array) -> int:
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor: torch.Tensor = torch.FloatTensor(state).unsqueeze(0)
        action_values: torch.Tensor = self.model(state_tensor)
        return torch.argmax(action_values).item()

    def replay(self, batch_size: int = 32):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target: torch.Tensor = self.model(torch.FloatTensor(state))
            target[action] = reward if done else reward + self.gamma * torch.max(self.model(torch.FloatTensor(next_state)))
            output: torch.Tensor = self.model(torch.FloatTensor(state))
            loss: torch.Tensor = self.criterion(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Train the RL model
state_size: int = 10  # Example feature vector size
action_size: int = 20  # Number of unique products
agent = RecommendationAgent(state_size, action_size)

for episode in range(1000):
    state = np.random.rand(state_size)  # Simulated user interaction state
    for time_step in range(10):
        action = agent.act(state)
        reward = np.random.choice([1, -1], p=[0.8, 0.2])  # Simulating user engagement
        next_state = np.random.rand(state_size)  # Simulated next state
        done = time_step == 9
        agent.remember(state, action, reward, next_state, done)
        state = next_state
    agent.replay()

# Making a recommendation
test_user_state: np.array = np.random.rand(state_size)
recommended_product: int = agent.act(test_user_state)
print(f"Recommended Product ID: {recommended_product}")
