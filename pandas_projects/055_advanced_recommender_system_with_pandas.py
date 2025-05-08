# cddml-H7fJ3kLwQpR
# File Name: advanced_recommender_system_with_pandas.py

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import networkx as nx  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore
from surprise import Dataset, Reader, SVD  # type: ignore
from surprise.model_selection import train_test_split as surprise_train_test_split  # type: ignore
import warnings  # type: ignore
from typing import Tuple, List, Dict, Any

warnings.filterwarnings("ignore")


# ---------------------------
# Data Simulation and Loading
# ---------------------------
def simulate_ratings_data(num_users: int = 100, num_items: int = 50, sparsity: float = 0.7) -> pd.DataFrame:
    np.random.seed(42)
    user_ids: np.ndarray = np.arange(1, num_users + 1)
    item_ids: np.ndarray = np.arange(1, num_items + 1)
    ratings: List[Dict[str, Any]] = []
    for user in user_ids:
        for item in item_ids:
            if np.random.rand() > sparsity:
                rating: float = np.random.uniform(1, 5)
                ratings.append({"user_id": int(user), "item_id": int(item), "rating": round(rating, 1)})
    df: pd.DataFrame = pd.DataFrame(ratings)
    return df


def simulate_item_metadata(num_items: int = 50) -> pd.DataFrame:
    np.random.seed(42)
    genres: List[str] = ["Action", "Comedy", "Drama", "Horror", "Sci-Fi"]
    items: List[Dict[str, Any]] = []
    for item in range(1, num_items + 1):
        genre: str = np.random.choice(genres)
        year: int = np.random.randint(1980, 2022)
        items.append({"item_id": int(item), "genre": genre, "year": year})
    df_items: pd.DataFrame = pd.DataFrame(items)
    return df_items


# ---------------------------
# Data Preprocessing
# ---------------------------
def preprocess_ratings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    le_user: LabelEncoder = LabelEncoder()
    le_item: LabelEncoder = LabelEncoder()
    df["user_id"] = le_user.fit_transform(df["user_id"])
    df["item_id"] = le_item.fit_transform(df["item_id"])
    return df


def create_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    matrix: pd.DataFrame = df.pivot(index="user_id", columns="item_id", values="rating")
    return matrix.fillna(0)


# ---------------------------
# Collaborative Filtering using Cosine Similarity
# ---------------------------
def compute_user_similarity(matrix: pd.DataFrame) -> np.ndarray:
    sim: np.ndarray = cosine_similarity(matrix)
    return sim


def recommend_by_user_similarity(matrix: pd.DataFrame, user_sim: np.ndarray, user_index: int, top_n: int = 5) -> List[
    int]:
    user_ratings: np.ndarray = matrix.iloc[user_index].values
    scores: np.ndarray = np.dot(user_sim[user_index], matrix)
    # Mask items already rated by user
    scores[user_ratings > 0] = 0
    recommended_indices: List[int] = list(np.argsort(scores)[::-1][:top_n])
    return recommended_indices


# ---------------------------
# Graph-Based Recommender using NetworkX
# ---------------------------
def build_item_similarity_graph(matrix: pd.DataFrame) -> nx.Graph:
    item_matrix: pd.DataFrame = matrix.transpose()
    sim_matrix: np.ndarray = cosine_similarity(item_matrix)
    G: nx.Graph = nx.Graph()
    num_items: int = sim_matrix.shape[0]
    for i in range(num_items):
        for j in range(i + 1, num_items):
            if sim_matrix[i, j] > 0.5:
                G.add_edge(item_matrix.index[i], item_matrix.index[j], weight=sim_matrix[i, j])
    return G


def recommend_by_graph(G: nx.Graph, item_id: int, top_n: int = 5) -> List[int]:
    neighbors: Dict[int, float] = dict(G[item_id])
    sorted_neighbors: List[int] = sorted(neighbors, key=neighbors.get, reverse=True)[:top_n]
    return sorted_neighbors


# ---------------------------
# Matrix Factorization using Surprise SVD
# ---------------------------
def train_surprise_svd(df: pd.DataFrame) -> Any:
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["user_id", "item_id", "rating"]], reader)
    trainset, testset = surprise_train_test_split(data, test_size=0.2, random_state=42)
    svd: SVD = SVD(n_factors=50, random_state=42)
    svd.fit(trainset)
    return svd, testset


def predict_surprise(svd: Any, testset: List[Tuple[int, int, float]]) -> List[float]:
    predictions: List[float] = [pred.est for pred in svd.test(testset)]
    return predictions


# ---------------------------
# Hybrid Recommender: Combine Graph & Matrix Factorization
# ---------------------------
def hybrid_recommendation(user_matrix: pd.DataFrame, svd: Any, G: nx.Graph, user_index: int, top_n: int = 5) -> List[
    int]:
    rec_sim: List[int] = recommend_by_user_similarity(user_matrix, compute_user_similarity(user_matrix), user_index,
                                                      top_n)
    # Get items from graph-based recommendation for each item in rec_sim
    hybrid_scores: Dict[int, float] = {}
    for item in rec_sim:
        graph_recs: List[int] = recommend_by_graph(G, item, top_n=top_n)
        for rec in graph_recs:
            hybrid_scores[rec] = hybrid_scores.get(rec, 0) + 1
    sorted_hybrid: List[int] = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)[:top_n]
    return sorted_hybrid


# ---------------------------
# Model Evaluation Metrics
# ---------------------------
def evaluate_predictions(y_true: List[float], y_pred: List[float]) -> None:
    rmse: float = np.sqrt(mean_squared_error(y_true, y_pred))
    print("RMSE:", rmse)


# ---------------------------
# Visualization Functions
# ---------------------------
def plot_rating_distribution(matrix: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    sns.heatmap(matrix, cmap="YlGnBu")
    plt.title("User-Item Rating Matrix Heatmap")
    plt.xlabel("Item ID")
    plt.ylabel("User ID")
    plt.show()


def plot_recommendations(recommendations: List[int]) -> None:
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(recommendations)), recommendations, color='skyblue')
    plt.title("Top Recommended Items")
    plt.xlabel("Rank")
    plt.ylabel("Item ID")
    plt.show()


# ---------------------------
# Main Pipeline Execution
# ---------------------------
if __name__ == "__main__":
    # Timestamp and project info
    current_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Timestamp: {current_time}")

    # Simulate dataset
    ratings_df: pd.DataFrame = simulate_ratings_data(num_users=100, num_items=50, sparsity=0.8)
    items_df: pd.DataFrame = simulate_item_metadata(num_items=50)
    ratings_df = preprocess_ratings(ratings_df)
    full_df: pd.DataFrame = pd.merge(ratings_df, items_df, on="item_id", how="left")

    # Create user-item matrix
    user_item_matrix: pd.DataFrame = create_user_item_matrix(full_df)
    plot_rating_distribution(user_item_matrix)

    # Collaborative Filtering via Cosine Similarity
    user_similarity: np.ndarray = compute_user_similarity(user_item_matrix)
    recommended_items: List[int] = recommend_by_user_similarity(user_item_matrix, user_similarity, user_index=0,
                                                                top_n=5)
    print("User 0 Recommended Items (Collaborative Filtering):", recommended_items)

    # Graph-based Recommender
    item_graph: nx.Graph = build_item_similarity_graph(user_item_matrix)
    graph_recs: List[int] = recommend_by_graph(item_graph, item_id=user_item_matrix.columns[0], top_n=5)
    print("Graph-based Recommendations for Item", user_item_matrix.columns[0], ":", graph_recs)

    # Matrix Factorization using Surprise SVD
    svd_model, testset = train_surprise_svd(full_df)
    svd_predictions: List[float] = predict_surprise(svd_model, testset)
    print("Surprise SVD Predictions (first 5):", svd_predictions[:5])

    # Hybrid Recommendation
    hybrid_recs: List[int] = hybrid_recommendation(user_item_matrix, svd_model, item_graph, user_index=0, top_n=5)
    print("Hybrid Recommendations for User 0:", hybrid_recs)
    plot_recommendations(hybrid_recs)

    # Evaluate SVD model using RMSE (simulate true ratings from testset)
    true_ratings: List[float] = [rating for (_, _, rating) in testset]
    evaluate_predictions(true_ratings, svd_predictions)

    # OPTIONAL: Dimensionality Reduction and Visualization
    pca: PCA = PCA(n_components=2)
    user_item_pca: np.ndarray = pca.fit_transform(user_item_matrix)
    tsne: TSNE = TSNE(n_components=2, random_state=42)
    user_item_tsne: np.ndarray = tsne.fit_transform(user_item_matrix)
    plt.figure(figsize=(10, 6))
    plt.scatter(user_item_tsne[:, 0], user_item_tsne[:, 1], c='blue', alpha=0.6)
    plt.title("t-SNE Visualization of User-Item Matrix")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()

    # End of pipeline for advanced multimodal recommender system using Pandas and other ML modules.
