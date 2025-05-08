# Project Title: cddml-CR8yXwLmFz
# File Name: advanced_multimodal_consumer_behavior_recommender.py

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import datetime  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.model_selection import train_test_split, StratifiedKFold  # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder  # type: ignore
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.cluster import KMeans  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.ensemble import RandomForestClassifier, StackingClassifier  # type: ignore
from surprise import Dataset, Reader, SVD  # type: ignore
from surprise.model_selection import train_test_split as surprise_train_test_split  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from joblib import dump, load  # type: ignore
import optuna  # type: ignore
from typing import List, Tuple, Dict, Any

# --- Data Simulation Functions ---
def simulate_customer_data(num_customers: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    customer_ids: np.ndarray = np.arange(1, num_customers + 1)
    ages: np.ndarray = np.random.randint(18, 70, num_customers)
    incomes: np.ndarray = np.random.normal(50, 15, num_customers)  # in k$
    segments: List[str] = np.random.choice(["low", "medium", "high"], num_customers).tolist()
    df_customers: pd.DataFrame = pd.DataFrame({
        "customer_id": customer_ids,
        "age": ages,
        "income": incomes,
        "segment": segments
    })
    return df_customers

def simulate_product_data(num_products: int = 100) -> pd.DataFrame:
    np.random.seed(99)
    product_ids: np.ndarray = np.arange(1, num_products + 1)
    categories: List[str] = np.random.choice(["electronics", "clothing", "home", "beauty"], num_products).tolist()
    prices: np.ndarray = np.random.uniform(5, 500, num_products)
    df_products: pd.DataFrame = pd.DataFrame({
        "product_id": product_ids,
        "category": categories,
        "price": prices
    })
    return df_products

def simulate_transaction_data(num_transactions: int = 5000) -> pd.DataFrame:
    np.random.seed(123)
    transaction_ids: np.ndarray = np.arange(1, num_transactions + 1)
    customer_ids: np.ndarray = np.random.randint(1, 501, num_transactions)
    product_ids: np.ndarray = np.random.randint(1, 101, num_transactions)
    ratings: np.ndarray = np.random.randint(1, 6, num_transactions)
    dates: pd.DatetimeIndex = pd.date_range(start="2023-01-01", periods=num_transactions, freq="H")
    df_transactions: pd.DataFrame = pd.DataFrame({
        "transaction_id": transaction_ids,
        "customer_id": customer_ids,
        "product_id": product_ids,
        "rating": ratings,
        "date": dates
    })
    return df_transactions

# --- Data Ingestion and Fusion ---
def fuse_datasets(customers: pd.DataFrame, products: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    df: pd.DataFrame = pd.merge(transactions, customers, on="customer_id", how="left")
    df = pd.merge(df, products, on="product_id", how="left")
    return df

# --- Data Preprocessing ---
def preprocess_fused_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["segment", "category"]:
        le: LabelEncoder = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    scaler: StandardScaler = StandardScaler()
    num_cols: List[str] = ["age", "income", "price", "rating"]
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

# --- Feature Engineering ---
def create_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    agg_df: pd.DataFrame = df.groupby("customer_id").agg({
        "rating": ["mean", "std"],
        "income": "first",
        "age": "first",
        "segment": "first"
    })
    agg_df.columns = ["_".join(col) for col in agg_df.columns]
    agg_df.reset_index(inplace=True)
    return agg_df

def add_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df["year"] = df["date"].dt.year.astype(int)
    df["month"] = df["date"].dt.month.astype(int)
    df["day"] = df["date"].dt.day.astype(int)
    return df

def create_text_features(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    # Simulate product reviews for content-based filtering
    reviews: List[str] = ["This product is excellent" if i % 2 == 0 else "Not good quality" for i in range(len(df))]
    vectorizer: TfidfVectorizer = TfidfVectorizer(stop_words="english", max_features=50)
    tfidf_matrix: np.ndarray = vectorizer.fit_transform(reviews).toarray()
    feature_names: List[str] = vectorizer.get_feature_names_out().tolist()
    return tfidf_matrix, feature_names

# --- Recommender System ---
def create_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    matrix: pd.DataFrame = df.pivot_table(index="customer_id", columns="product_id", values="rating", aggfunc="mean")
    return matrix.fillna(0)

def collaborative_filtering_recommendations(matrix: pd.DataFrame, user_idx: int, top_n: int = 5) -> List[int]:
    from sklearn.metrics.pairwise import cosine_similarity
    sim: np.ndarray = cosine_similarity(matrix)
    scores: np.ndarray = np.dot(sim[user_idx], matrix)
    user_ratings: np.ndarray = matrix.iloc[user_idx].values
    scores[user_ratings > 0] = 0
    recommended_indices: List[int] = list(np.argsort(scores)[::-1][:top_n])
    return matrix.columns[recommended_indices].tolist()

def build_surprise_model(df: pd.DataFrame) -> Tuple[Any, List[Tuple[int, int, float]]]:
    from surprise import Reader, Dataset, SVD
    reader: Reader = Reader(rating_scale=(1, 5))
    data: Dataset = Dataset.load_from_df(df[["customer_id", "product_id", "rating"]], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    svd: SVD = SVD(n_factors=50, random_state=42)
    svd.fit(trainset)
    return svd, testset

def predict_surprise(svd: Any, testset: List[Tuple[int, int, float]]) -> List[float]:
    predictions: List[float] = [pred.est for pred in svd.test(testset)]
    return predictions

# --- Deep Learning for Recommendation (Neural Collaborative Filtering) ---
def build_ncf_model(input_dim: int) -> tf.keras.Model:
    model: tf.keras.Model = Sequential([
        Dense(256, activation="relu", input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_ncf_model(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> tf.keras.Model:
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[early_stop], verbose=0)
    return model

# --- Key Learnings / Research Areas ---
key_learnings: Dict[str, List[str]] = {
    "Features": ["Data Fusion", "Multi-Modal Integration", "Feature Engineering", "Collaborative Filtering", "Content-Based Filtering", "Neural Collaborative Filtering", "Model Explainability"],
    "Components": ["Pandas", "Dask", "Modin", "Scikit-Learn", "TensorFlow", "Surprise", "Optuna", "SHAP", "LIME"],
    "Keywords": ["Consumer Behavior", "Recommender Systems", "Multi-Modal Data", "Ensemble Learning", "Deep Learning", "Predictive Analytics"],
    "Research Areas": ["Recommender Systems", "Collaborative Filtering", "Content-Based Recommendation", "Neural Collaborative Filtering", "Multi-Modal Data Fusion"],
    "Hashtags": ["#RecommenderSystems", "#MultiModalML", "#PredictiveAnalytics", "#DeepLearning", "#DataFusion"]
}

# Main Pipeline Execution
if __name__ == "__main__":
    current_timestamp: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Current Timestamp:", current_timestamp)

    # Simulate data
    customers: pd.DataFrame = simulate_customer_data(num_customers=500)
    products: pd.DataFrame = simulate_product_data(num_products=100)
    transactions: pd.DataFrame = simulate_transaction_data(num_transactions=5000)

    # Fuse datasets
    fused_df: pd.DataFrame = fuse_datasets(customers, products, transactions)
    fused_df = add_transaction_features(fused_df)
    fused_df = preprocess_fused_data(fused_df)

    # Create aggregated features for customer segmentation
    agg_df: pd.DataFrame = create_aggregate_features(fused_df)

    # Split dataset for classification (predicting high spender: spending_score > median)
    median_spending: float = agg_df["rating_mean"].median()
    agg_df["high_spender"] = (agg_df["rating_mean"] > median_spending).astype(int)
    X_train_rec, X_test_rec, y_train_rec, y_test_rec = train_test_split(agg_df.drop(columns=["customer_id", "high_spender"]), agg_df["high_spender"], test_size=0.2, random_state=42, stratify=agg_df["high_spender"])

    # Build stacking ensemble model for customer segmentation
    stacking_model: StackingClassifier = build_stacking_model(X_train_rec, y_train_rec)
    stack_preds: np.ndarray = stacking_model.predict(X_test_rec)
    print("Stacking Ensemble Model Accuracy:", accuracy_score(y_test_rec, stack_preds))
    evaluate_model(y_test_rec, stack_preds)

    # Build and evaluate Surprise SVD for collaborative filtering recommendations
    svd_model, testset = build_surprise_model(fused_df)
    svd_preds: List[float] = predict_surprise(svd_model, testset)
    print("First 5 Surprise SVD Predictions:", svd_preds[:5])

    # Create user-item matrix for collaborative filtering recommendations
    user_item_matrix: pd.DataFrame = create_user_item_matrix(fused_df)
    rec_items: List[int] = collaborative_filtering_recommendations(user_item_matrix, user_idx=0, top_n=5)
    print("Collaborative Filtering Recommendations for User 0:", rec_items)

    # Build knowledge graph from product co-purchase (simulate with transactions)
    product_graph: nx.Graph = nx.Graph()
    for _, row in transactions.iterrows():
        product_graph.add_edge(row["product_id"], row["customer_id"], weight=row["rating"])
    # Query graph for a product (e.g., product_id = 10)
    neighbors: List[Any] = list(product_graph.neighbors(10))
    print("Neighbors of Product 10 in Knowledge Graph:", neighbors)

    # Train deep learning model for neural collaborative filtering
    X_matrix: pd.DataFrame = create_user_item_matrix(fused_df)
    X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(X_matrix, (X_matrix.sum(axis=1) > X_matrix.sum(axis=1).median()).astype(int), test_size=0.2, random_state=42)
    scaler_dl: StandardScaler = StandardScaler()
    X_train_dl_scaled: np.ndarray = scaler_dl.fit_transform(X_train_dl)
    X_test_dl_scaled: np.ndarray = scaler_dl.transform(X_test_dl)
    ncf_model: tf.keras.Model = build_ncf_model(input_dim=X_train_dl_scaled.shape[1])
    ncf_model = train_ncf_model(ncf_model, X_train_dl_scaled, y_train_dl.to_numpy(), X_test_dl_scaled, y_test_dl.to_numpy())
    ncf_preds: np.ndarray = (ncf_model.predict(X_test_dl_scaled) > 0.5).astype(int).flatten()
    print("Neural Collaborative Filtering Model Accuracy:", accuracy_score(y_test_dl.to_numpy(), ncf_preds))

    # Hyperparameter tuning with Optuna for RandomForest on aggregated features
    X_train_np: np.ndarray = X_train_rec.values
    y_train_np: np.ndarray = y_train_rec.values
    study: optuna.study.Study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: optuna_objective(trial, X_train_np, y_train_np), n_trials=30)
    print("Optuna Best Params for RandomForest:", study.best_params)

    # SHAP explanation for stacking model
    shap_explain_model(stacking_model, pd.DataFrame(X_train_rec.values, columns=X_train_rec.columns))

    # LIME explanation for stacking model (using first instance of X_test_rec)
    lime_explain_model(stacking_model, X_train_rec.values, X_train_rec.columns.tolist(), X_test_rec.values)

    # Save final models
    dump(stacking_model, "final_stacking_recommender.joblib")
    dump(ncf_model, "final_ncf_model.joblib")

    # Key Learnings / Research Areas
    key_learnings: Dict[str, List[str]] = {
        "Features": ["Data Fusion", "Multi-Modal Integration", "Collaborative Filtering", "Content-Based Filtering", "Neural Collaborative Filtering", "Knowledge Graph Construction", "Ensemble Learning", "Explainable AI"],
        "Components": ["Pandas", "Dask", "Modin", "Scikit-Learn", "TensorFlow", "Surprise", "NetworkX", "Optuna", "SHAP", "LIME"],
        "Keywords": ["Consumer Behavior", "Recommender Systems", "Multi-Modal Data", "Ensemble Learning", "Predictive Analytics", "Deep Learning"],
        "Research Areas": ["Recommender Systems", "Data Integration", "AutoML", "Model Explainability", "Graph Analytics"],
        "Hashtags": ["#RecommenderSystems", "#MultiModalML", "#PredictiveAnalytics", "#DeepLearning", "#DataFusion"]
    }
    print("Key Learnings:\n", key_learnings)

    # Timestamp, Timeframe, and Final Product Description
    project_timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Project Timestamp:", project_timestamp)
    estimated_timeframe: str = "4-6 months"
    print("Estimated Completion Timeframe:", estimated_timeframe)

    # Final product: A scalable, modular recommendation engine that integrates customer, product, and transaction data,
    # providing collaborative and content-based recommendations, ensemble models, and interactive knowledge graphs.

    # Sister Projects: Deep Learning Recommenders, Customer Segmentation, Fraud Detection Systems.
    # Integrated Applications: BI Dashboards, E-Commerce Platforms, CRM Systems.

    # Where to go from here: Explore cloud integration, real-time data streaming with Kafka, and advanced graph neural networks.
