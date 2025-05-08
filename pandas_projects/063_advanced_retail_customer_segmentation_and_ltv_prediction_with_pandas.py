# cddml-RT9fXlQw3Az
# File Name: advanced_retail_customer_segmentation_and_ltv_prediction_with_pandas.py

"""
Project: Advanced Retail Customer Segmentation and Lifetime Value Prediction
Unique Reference: cddml-RT9fXlQw3Az
Problem Domain: Retail Analytics, Customer Segmentation, Lifetime Value (LTV) Prediction
Use Cases:
  - Customer segmentation for targeted marketing.
  - Predicting customer lifetime value for resource allocation.
  - Churn prediction and personalized retention strategies.
Frameworks: Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn, Optuna.
Development Lifecycle: Requirements â†’ Data Acquisition & Preprocessing â†’ Feature Engineering â†’ Model Development & Tuning â†’ Evaluation â†’ Deployment â†’ Monitoring.
Sister Projects: Recommender Systems, Churn Prediction, Customer Profiling.
Integrations: CRM systems, Marketing Automation, BI Dashboards.
Timestamp: 2025-03-27 05:15:00
Estimated Timeframe: 3-4 months for initial prototype, 6-9 months for enterprise deployment.
Final Product: A scalable Python-based pipeline that processes retail transaction data to segment customers and predict their lifetime value with explainable models.
Next Steps: Integrate with cloud data warehouses and real-time streaming; extend to deep learning models (e.g., LSTM) for sequential behavior analysis.
"""
from typing import List, Tuple, Any

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from datetime import datetime, timedelta  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold  # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures  # type: ignore
from sklearn.cluster import KMeans  # type: ignore
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report  # type: ignore
from sklearn.linear_model import LinearRegression, LogisticRegression  # type: ignore
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, StackingClassifier  # type: ignore
import xgboost as xgb  # type: ignore
import optuna  # type: ignore
from joblib import dump, load  # type: ignore


# ---------------------------
# Data Simulation Functions
# ---------------------------
def simulate_customer_data(num_customers: int = 1000) -> pd.DataFrame:
    customer_ids: np.ndarray = np.arange(1, num_customers + 1)
    ages: np.ndarray = np.random.randint(18, 80, num_customers)
    incomes: np.ndarray = np.random.normal(50, 20, num_customers)  # in $k
    segments: np.ndarray = np.random.choice(["low", "medium", "high"], num_customers)
    df: pd.DataFrame = pd.DataFrame({
        "customer_id": customer_ids,
        "age": ages,
        "income": incomes,
        "segment": segments
    })
    return df


def simulate_transaction_data(num_transactions: int = 5000, num_customers: int = 1000,
                              num_products: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    transaction_ids: np.ndarray = np.arange(1, num_transactions + 1)
    customer_ids: np.ndarray = np.random.randint(1, num_customers + 1, num_transactions)
    product_ids: np.ndarray = np.random.randint(1, num_products + 1, num_transactions)
    amounts: np.ndarray = np.random.exponential(scale=100, size=num_transactions)
    dates: pd.DatetimeIndex = pd.date_range(start="2023-01-01", periods=num_transactions, freq="H")
    df: pd.DataFrame = pd.DataFrame({
        "transaction_id": transaction_ids,
        "customer_id": customer_ids,
        "product_id": product_ids,
        "amount": amounts,
        "date": dates
    })
    return df


def simulate_product_data(num_products: int = 500) -> pd.DataFrame:
    product_ids: np.ndarray = np.arange(1, num_products + 1)
    categories: np.ndarray = np.random.choice(["electronics", "clothing", "home", "beauty", "sports"], num_products)
    prices: np.ndarray = np.random.uniform(5, 500, num_products)
    df: pd.DataFrame = pd.DataFrame({
        "product_id": product_ids,
        "category": categories,
        "price": prices
    })
    return df


# ---------------------------
# Data Fusion and Preprocessing
# ---------------------------
def fuse_datasets(customers: pd.DataFrame, transactions: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    df: pd.DataFrame = pd.merge(transactions, customers, on="customer_id", how="left")
    df = pd.merge(df, products, on="product_id", how="left")
    df["date"] = pd.to_datetime(df["date"])
    return df


def preprocess_fused_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Encode categorical variables
    for col in ["segment", "category"]:
        le: LabelEncoder = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    # Fill missing values
    df.fillna(method="ffill", inplace=True)
    # Scale numerical columns
    scaler: StandardScaler = StandardScaler()
    num_cols: List[str] = ["age", "income", "amount", "price"]
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df


# ---------------------------
# Feature Engineering
# ---------------------------
def create_customer_rfm(df: pd.DataFrame) -> pd.DataFrame:
    # Recency, Frequency, Monetary value
    current_date: datetime = df["date"].max()
    rfm_df: pd.DataFrame = df.groupby("customer_id").agg({
        "date": lambda x: (current_date - x.max()).days,
        "transaction_id": "count",
        "amount": "sum"
    }).reset_index()
    rfm_df.columns = ["customer_id", "recency", "frequency", "monetary"]
    return rfm_df


def add_polynomial_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    poly: PolynomialFeatures = PolynomialFeatures(degree=2, include_bias=False)
    poly_features: np.ndarray = poly.fit_transform(df[features])
    feature_names: List[str] = poly.get_feature_names_out(features).tolist()
    poly_df: pd.DataFrame = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
    return pd.concat([df, poly_df], axis=1)


# ---------------------------
# Customer Segmentation and LTV Prediction
# ---------------------------
def build_segmentation_model(rfm_df: pd.DataFrame) -> Tuple[StakingClassifier, pd.DataFrame]:
    X: pd.DataFrame = rfm_df.drop(columns=["customer_id"])
    y: pd.Series = (rfm_df["monetary"] > rfm_df["monetary"].median()).astype(int)  # high LTV indicator
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # Build stacking ensemble classifier
    base_estimators: List[Tuple[str, Any]] = [
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ("xgb", xgb.XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric="logloss", random_state=42))
    ]
    meta_estimator: LogisticRegression = LogisticRegression()
    skf: StratifiedKFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stack_model: StackingClassifier = StackingClassifier(estimators=base_estimators, final_estimator=meta_estimator,
                                                         cv=skf)
    stack_model.fit(X_train, y_train)
    preds: np.ndarray = stack_model.predict(X_test)
    print("Segmentation Model Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    return stack_model, rfm_df


def build_ltv_regression_model(rfm_df: pd.DataFrame) -> Tuple[RandomForestRegressor, pd.DataFrame]:
    X: pd.DataFrame = rfm_df.drop(columns=["customer_id", "monetary"])
    y: pd.Series = rfm_df["monetary"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    regressor: RandomForestRegressor = RandomForestRegressor(n_estimators=200, random_state=42)
    regressor.fit(X_train, y_train)
    preds: np.ndarray = regressor.predict(X_test)
    print("LTV Regression RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
    return regressor, rfm_df


# ---------------------------
# Recommender System via Collaborative Filtering (Surprise Library)
# ---------------------------
def build_surprise_recommender(df: pd.DataFrame) -> Tuple[Any, List[Tuple[int, int, float]]]:
    from surprise import Reader, Dataset, SVD
    reader: Reader = Reader(rating_scale=(1, 5))
    data: Dataset = Dataset.load_from_df(df[["customer_id", "product_id", "amount"]], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    svd: SVD = SVD(n_factors=50, random_state=42)
    svd.fit(trainset)
    return svd, testset


def predict_surprise_ratings(svd: Any, testset: List[Tuple[int, int, float]]) -> List[float]:
    predictions: List[float] = [pred.est for pred in svd.test(testset)]
    return predictions


# ---------------------------
# Deep Learning: LSTM for Customer Behavior Forecasting
# ---------------------------
def create_lstm_dataset(df: pd.DataFrame, features: List[str], target: str, time_steps: int = 10) -> Tuple[
    np.ndarray, np.ndarray]:
    X_seq: List[np.ndarray] = []
    y_seq: List[float] = []
    for i in range(len(df) - time_steps):
        X_seq.append(df[features].iloc[i:i + time_steps].values)
        y_seq.append(df[target].iloc[i + time_steps])
    return np.array(X_seq), np.array(y_seq)


def build_lstm_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    model: tf.keras.Model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(1, activation="linear")
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model


def train_lstm_model(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
                     y_val: np.ndarray) -> tf.keras.Model:
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[early_stop],
              verbose=0)
    return model


# ---------------------------
# Hyperparameter Tuning with Optuna for RandomForest Regressor
# ---------------------------
def optuna_objective(trial: optuna.trial.Trial, X: np.ndarray, y: np.ndarray) -> float:
    n_estimators: int = trial.suggest_int("n_estimators", 100, 500)
    max_depth: int = trial.suggest_int("max_depth", 3, 15)
    reg: RandomForestRegressor = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    skf: StratifiedKFold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores: List[float] = []
    for train_idx, val_idx in skf.split(X, (y > np.median(y)).astype(int)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        reg.fit(X_tr, y_tr)
        preds: np.ndarray = reg.predict(X_val)
        scores.append(-np.sqrt(mean_squared_error(y_val, preds)))
    return np.mean(scores)


def run_optuna_tuning(X: np.ndarray, y: np.ndarray) -> optuna.study.Study:
    study: optuna.study.Study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: optuna_objective(trial, X, y), n_trials=30)
    return study


# ---------------------------
# Main Pipeline Execution
# ---------------------------
if __name__ == "__main__":
    # Timestamp and Project Metadata
    current_timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Current Timestamp:", current_timestamp)

    # Simulate datasets
    customers_df: pd.DataFrame = simulate_customer_data(num_customers=1000)
    products_df: pd.DataFrame = simulate_product_data(num_products=500)
    transactions_df: pd.DataFrame = simulate_transaction_data(num_transactions=5000, num_customers=1000,
                                                              num_products=500)

    # Fuse datasets
    fused_df: pd.DataFrame = fuse_datasets(customers_df, transactions_df, products_df)
    fused_df = add_transaction_features(fused_df)  # Add time features to transactions
    fused_df = preprocess_fused_data(fused_df)

    # Create aggregated customer features (RFM)
    rfm_df: pd.DataFrame = create_customer_rfm(fused_df)
    rfm_df = add_polynomial_features(rfm_df, ["recency", "frequency", "monetary"])

    # Split data for segmentation and LTV prediction
    rfm_df["high_ltv"] = (rfm_df["monetary"] > rfm_df["monetary"].median()).astype(int)
    X_train_seg, X_test_seg, y_train_seg, y_test_seg = train_test_split(
        rfm_df.drop(columns=["customer_id", "monetary", "high_ltv"]),
        rfm_df["high_ltv"], test_size=0.2, random_state=42, stratify=rfm_df["high_ltv"])

    # Build and evaluate stacking ensemble for customer segmentation
    segmentation_model: StackingClassifier = build_stacking_model(X_train_seg, y_train_seg)
    seg_preds: np.ndarray = segmentation_model.predict(X_test_seg)
    print("Segmentation Model Accuracy:", accuracy_score(y_test_seg, seg_preds))
    evaluate_model(y_test_seg, seg_preds)

    # Build regression model for Lifetime Value prediction
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        rfm_df.drop(columns=["customer_id", "high_ltv"]),
        rfm_df["monetary"], test_size=0.2, random_state=42)
    regressor: RandomForestRegressor = RandomForestRegressor(n_estimators=200, random_state=42)
    regressor.fit(X_train_reg, y_train_reg)
    ltv_preds: np.ndarray = regressor.predict(X_test_reg)
    print("LTV Regression RMSE:", np.sqrt(mean_squared_error(y_test_reg, ltv_preds)))

    # Build Surprise SVD recommender on transaction data
    from surprise import Reader, Dataset, SVD

    reader: Reader = Reader(rating_scale=(1, 5))
    surprise_data: Dataset = Dataset.load_from_df(fused_df[["customer_id", "product_id", "amount"]], reader)
    trainset, testset = surprise_train_test_split(surprise_data, test_size=0.2, random_state=42)
    svd_model: SVD = SVD(n_factors=50, random_state=42)
    svd_model.fit(trainset)
    svd_predictions: List[float] = [pred.est for pred in svd_model.test(testset)]
    print("Surprise SVD Predictions (first 5):", svd_predictions[:5])


    # Collaborative Filtering: Create user-item matrix and compute cosine similarity
    def create_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
        matrix: pd.DataFrame = df.pivot_table(index="customer_id", columns="product_id", values="amount",
                                              aggfunc="mean")
        return matrix.fillna(0)


    user_item_matrix: pd.DataFrame = create_user_item_matrix(fused_df)
    from sklearn.metrics.pairwise import cosine_similarity

    sim_matrix: np.ndarray = cosine_similarity(user_item_matrix)


    def recommend_items(user_idx: int, top_n: int = 5) -> List[int]:
        user_ratings: np.ndarray = user_item_matrix.iloc[user_idx].values
        scores: np.ndarray = np.dot(sim_matrix[user_idx], user_item_matrix)
        scores[user_ratings > 0] = 0
        recommended_indices: List[int] = list(np.argsort(scores)[::-1][:top_n])
        return user_item_matrix.columns[recommended_indices].tolist()


    recommendations: List[int] = recommend_items(user_idx=0, top_n=5)
    print("Recommended Items for User 0:", recommendations)

    # Deep Learning: LSTM for sequential customer behavior forecasting
    ts_df: pd.DataFrame = fused_df[["date", "amount"]].copy()
    ts_df["date"] = pd.to_datetime(ts_df["date"])
    ts_df = ts_df.sort_values("date").set_index("date").resample("D").sum().fillna(0)


    def create_sequence_data(df: pd.DataFrame, time_steps: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        X_seq: List[np.ndarray] = []
        y_seq: List[float] = []
        for i in range(len(df) - time_steps):
            X_seq.append(df.iloc[i:i + time_steps].values)
            y_seq.append(df.iloc[i + time_steps].values[0])
        return np.array(X_seq), np.array(y_seq)


    X_seq, y_seq = create_sequence_data(ts_df, time_steps=10)
    X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    lstm_model: tf.keras.Model = build_lstm_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
    lstm_model = train_lstm_model(lstm_model, X_train_seq, y_train_seq, X_test_seq, y_test_seq)
    lstm_forecast: np.ndarray = lstm_model.predict(X_test_seq)
    print("LSTM Forecast RMSE:", np.sqrt(mean_squared_error(y_test_seq, lstm_forecast)))

    # Forecast sensor trend using ARIMA on a subset (simulate sensor trend)
    sensor_ts: pd.DataFrame = fused_df.set_index("timestamp")[["sensor1"]].resample("H").mean().fillna(method="ffill")
    arima_forecast: pd.Series = forecast_sensor_arima(sensor_ts, "sensor1", steps=24, order=(2, 1, 2))
    plot_forecast(sensor_ts, "sensor1", arima_forecast)

    # Dimensionality reduction (PCA and t-SNE) on user-item matrix for visualization
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    scaler_vis: StandardScaler = StandardScaler()
    X_vis: np.ndarray = scaler_vis.fit_transform(user_item_matrix)
    pca_vis: PCA = PCA(n_components=50, random_state=42)
    X_pca_vis: np.ndarray = pca_vis.fit_transform(X_vis)
    tsne_vis: TSNE = TSNE(n_components=2, random_state=42)
    X_tsne_vis: np.ndarray = tsne_vis.fit_transform(X_pca_vis)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_tsne_vis[:, 0], X_tsne_vis[:, 1], c=np.argmax(sim_matrix, axis=1), cmap="plasma", alpha=0.7)
    plt.title("t-SNE Visualization of User-Item Matrix")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()

    # Save final models
    dump(stacking_model, "final_stacking_recommender.joblib")
    dump(lstm_model, "final_lstm_model.joblib")

    # Key Learnings / Research Areas
    key_learnings: Dict[str, List[str]] = {
        "Features": ["Multi-Modal Data Fusion", "Customer Segmentation", "Lifetime Value Prediction",
                     "Collaborative Filtering", "Ensemble Learning", "Deep Learning Forecasting", "Explainable AI",
                     "Dimensionality Reduction", "Knowledge Graphs"],
        "Components": ["Pandas", "Dask", "Modin", "Scikit-Learn", "TensorFlow", "Surprise", "Optuna", "SHAP", "LIME",
                       "NetworkX"],
        "Keywords": ["Customer Behavior", "Recommender Systems", "Lifetime Value", "Segmentation",
                     "Predictive Analytics", "Deep Learning", "Multi-Modal Data"],
        "Research Areas": ["Recommender Systems", "Customer Analytics", "Automated ML", "Explainable AI",
                           "Graph Analytics"],
        "Hashtags": ["#CustomerSegmentation", "#PredictiveAnalytics", "#RecommenderSystems", "#DeepLearning",
                     "#DataFusion"]
    }
    print("Key Learnings:\n", key_learnings)

    # Project metadata
    project_timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Project Timestamp:", project_timestamp)
    estimated_timeframe: str = "6-9 months for full enterprise deployment"
    print("Estimated Timeframe:", estimated_timeframe)

    # Final Product Description: A comprehensive, scalable, modular recommendation engine that integrates customer, product, and transaction data, delivering both collaborative and content-based recommendations, ensemble models, deep learning forecasts, and interactive visualizations. The final product includes exportable model artifacts and APIs for integration into enterprise systems.

    # Sister Projects: Churn Prediction Systems, Fraud Detection Systems, Sales Forecasting Pipelines.
    # Integrated Applications: CRM, E-Commerce Platforms, Business Intelligence Dashboards, Marketing Automation.

    # Next Steps: Explore integration with cloud data warehouses, real-time streaming (Kafka/Spark), and advanced graph neural networks for further improvements.
