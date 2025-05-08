import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import pandas as pd
from surprise import SVD, Reader
from surprise import Dataset
from surprise.model_selection import cross_validate

# Load MovieLens dataset (User ID, Movie ID, Rating)
data_url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
data = pd.read_csv(data_url, sep='\t', names=["user", "item", "stars", "timestamp"])

# Drop unnecessary column
data = data.drop(columns=["timestamp"])


def preprocess_data(the_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess dataset by extracting relevant features."""
    dframe = pd.DataFrame(the_data)
    dframe.dropna(inplace=True)
    return dframe


df: pd.DataFrame = preprocess_data(data)
user_map: dict = {id: i for i, id in enumerate(df['user'].unique())}
item_map: dict = {id: i for i, id in enumerate(df['item'].unique())}


def create_graph(df: pd.DataFrame) -> (int, Data):
    """Create a Graph Data object for PyTorch Geometric."""
    edge_index = torch.tensor(df[['user', 'item']].values.T, dtype=torch.long)
    _num_nodes: int = len(df)
    _graph_data: Data = Data(edge_index=edge_index, num_nodes=_num_nodes)
    return _num_nodes, _graph_data  # Explicitly set num_nodes


num_nodes: int
graph_data: Data
num_nodes, graph_data = create_graph(df)
print(f"graph_data : {graph_data}")
print(f"num_nodes : {num_nodes}")


# Define a Graph Neural Network model
class GCN(torch.nn.Module):
    def __init__(self, num_nodes: int, embedding_dim: int = 64):
        super(GCN, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, embedding_dim)
        self.conv1 = GCNConv(embedding_dim, 128)
        self.conv2 = GCNConv(128, 64)
        self.fc = torch.nn.Linear(64, 1)

    def forward(self, data: Data):
        x = self.embedding(torch.arange(data.num_nodes))
        x = F.relu(self.conv1(x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        x = self.fc(x)
        return x


model = GCN(num_nodes)


def train_model(the_model: GCN, the_graph_data: Data, epochs: int = 20):
    """Train the Graph Neural Network."""
    optimizer = torch.optim.Adam(the_model.parameters(), lr=0.01)
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = the_model(the_graph_data)
        loss = F.mse_loss(out.squeeze(), torch.tensor(df['stars'].values, dtype=torch.float))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")


print(f"model: {model}")
train_model(model, graph_data)

# Collaborative Filtering with Surprise
reader = Dataset.load_from_df(df[['user', 'item', 'stars']], reader=Reader(rating_scale=(1, 5)))
svd = SVD()
cross_validate(svd, reader, cv=5, verbose=True)


# Example prediction
def recommend_product(user_id: int) -> list:
    """Recommend top 5 products for a given user."""
    user_encoded = user_map.get(user_id, None)
    if user_encoded is None:
        return []
    predictions = [(item, svd.predict(user_encoded, item).est) for item in item_map.values()]
    prod_recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]
    return prod_recommendations


example_user: int = list(user_map.keys())[0]
recommendations: list = recommend_product(example_user)
print(f"example_user: {example_user}, Recommended Products: {recommendations}")
