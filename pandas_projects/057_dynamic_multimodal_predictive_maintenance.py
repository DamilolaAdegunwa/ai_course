# cddml-QpX7vRfLzNk
# File Name: dynamic_multimodal_predictive_maintenance.py

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
import dask.dataframe as dd  # type: ignore
import modin.pandas as mpd  # type: ignore
from sklearn.model_selection import train_test_split, StratifiedKFold  # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder  # type: ignore
from sklearn.ensemble import IsolationForest, RandomForestClassifier  # type: ignore
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
import tensorflow as tf  # type: ignore
import optuna  # type: ignore
from joblib import dump, load  # type: ignore
from datetime import datetime, timedelta  # type: ignore
from sklearn.cluster import DBSCAN  # type: ignore
import shap  # type: ignore
import lime.lime_tabular  # type: ignore


# ---------------------------
# Data Simulation Functions
# ---------------------------
def simulate_sensor_data(num_records: int = 10000) -> pd.DataFrame:
    np.random.seed(42)
    timestamps: pd.DatetimeIndex = pd.date_range(start="2023-01-01", periods=num_records, freq="H")
    sensor_1: np.ndarray = np.random.normal(loc=50, scale=5, size=num_records)
    sensor_2: np.ndarray = np.random.normal(loc=75, scale=7, size=num_records)
    sensor_3: np.ndarray = np.random.normal(loc=100, scale=10, size=num_records)
    df: pd.DataFrame = pd.DataFrame({
        "timestamp": timestamps,
        "sensor_1": sensor_1,
        "sensor_2": sensor_2,
        "sensor_3": sensor_3
    })
    df.loc[np.random.choice(df.index, size=50, replace=False), "sensor_1"] += np.random.uniform(20, 40, 50)
    return df


def simulate_maintenance_log(num_records: int = 200) -> pd.DataFrame:
    np.random.seed(24)
    timestamps: pd.DatetimeIndex = pd.date_range(start="2023-01-01", periods=num_records, freq="7D")
    failure: np.ndarray = np.random.choice([0, 1], size=num_records, p=[0.8, 0.2])
    df: pd.DataFrame = pd.DataFrame({
        "timestamp": timestamps,
        "equipment_failure": failure
    })
    return df


def simulate_environmental_data(num_records: int = 10000) -> pd.DataFrame:
    np.random.seed(99)
    timestamps: pd.DatetimeIndex = pd.date_range(start="2023-01-01", periods=num_records, freq="H")
    temperature: np.ndarray = np.random.normal(loc=25, scale=3, size=num_records)
    humidity: np.ndarray = np.random.normal(loc=50, scale=10, size=num_records)
    df: pd.DataFrame = pd.DataFrame({
        "timestamp": timestamps,
        "temperature": temperature,
        "humidity": humidity
    })
    return df


# ---------------------------
# Data Ingestion and Fusion
# ---------------------------
def load_and_fuse_data() -> pd.DataFrame:
    sensor_df: pd.DataFrame = simulate_sensor_data()
    maintenance_df: pd.DataFrame = simulate_maintenance_log()
    env_df: pd.DataFrame = simulate_environmental_data()
    sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])
    env_df["timestamp"] = pd.to_datetime(env_df["timestamp"])
    fused_df: pd.DataFrame = pd.merge_asof(sensor_df.sort_values("timestamp"), env_df.sort_values("timestamp"),
                                           on="timestamp", direction="nearest")
    maintenance_df["timestamp"] = pd.to_datetime(maintenance_df["timestamp"])
    fused_df["date"] = fused_df["timestamp"].dt.floor("D")
    maintenance_df["date"] = maintenance_df["timestamp"].dt.floor("D")
    maintenance_daily: pd.DataFrame = maintenance_df.groupby("date")["equipment_failure"].max().reset_index()
    fused_df = pd.merge(fused_df, maintenance_daily, on="date", how="left")
    fused_df["equipment_failure"].fillna(0, inplace=True)
    return fused_df


# ---------------------------
# Data Preprocessing
# ---------------------------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    scaler: StandardScaler = StandardScaler()
    numerical_cols: List[str] = ["sensor_1", "sensor_2", "sensor_3", "temperature", "humidity"]
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    return df


def add_lag_features(df: pd.DataFrame, col: str, lags: List[int]) -> pd.DataFrame:
    for lag in lags:
        df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, col: str, windows: List[int]) -> pd.DataFrame:
    for window in windows:
        df[f"{col}_roll_mean_{window}"] = df[col].rolling(window=window).mean()
        df[f"{col}_roll_std_{window}"] = df[col].rolling(window=window).std()
    return df


# ---------------------------
# Feature Engineering and Labeling
# ---------------------------
def label_failure(df: pd.DataFrame, sensor_col: str, threshold: float) -> pd.DataFrame:
    df["failure_label"] = (df[sensor_col] > threshold).astype(int)
    return df


# ---------------------------
# Anomaly Detection using Isolation Forest
# ---------------------------
def detect_anomalies(df: pd.DataFrame, features: List[str], contamination: float = 0.01) -> pd.DataFrame:
    iso: IsolationForest = IsolationForest(contamination=contamination, random_state=42)
    df["anomaly"] = iso.fit_predict(df[features])
    return df


# ---------------------------
# LSTM Model for Predictive Maintenance
# ---------------------------
def create_lstm_dataset(df: pd.DataFrame, feature_cols: List[str], target_col: str, time_steps: int = 10) -> Tuple[
    np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(df) - time_steps):
        X.append(df[feature_cols].iloc[i:i + time_steps].values)
        y.append(df[target_col].iloc[i + time_steps])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    model: tf.keras.Model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_lstm_model(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
                     y_val: np.ndarray) -> tf.keras.Model:
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[early_stop],
              verbose=0)
    return model


# ---------------------------
# Ensemble Model using Stacking
# ---------------------------
def build_stacking_model(X_train: pd.DataFrame, y_train: pd.Series) -> object:
    from sklearn.ensemble import StackingClassifier
    base_estimators = [
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ("xgb", XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric="logloss", random_state=42))
    ]
    final_estimator = LogisticRegression()
    stacking = StackingClassifier(estimators=base_estimators, final_estimator=final_estimator,
                                  cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
    stacking.fit(X_train, y_train)
    return stacking


# ---------------------------
# Evaluation Metrics and Visualization
# ---------------------------
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    acc = accuracy_score(y_true, y_pred)
    print("Accuracy:", acc)
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def plot_time_series_forecast(actual: pd.Series, forecast: pd.Series) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual, label="Actual")
    forecast_index = pd.date_range(start=actual.index[-1] + timedelta(hours=1), periods=len(forecast), freq="H")
    plt.plot(forecast_index, forecast, label="Forecast", linestyle="--", color="red")
    plt.title("Time Series Forecast")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


# ---------------------------
# Main Pipeline Execution
# ---------------------------
if __name__ == "__main__":
    current_time: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Current Timestamp:", current_time)

    # Data Fusion: Load and merge sensor, maintenance, and environmental data
    sensor_df: pd.DataFrame = simulate_sensor_data(num_records=5000)
    maintenance_df: pd.DataFrame = simulate_maintenance_log(num_records=200)
    env_df: pd.DataFrame = simulate_environmental_data(num_records=5000)

    sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])
    env_df["timestamp"] = pd.to_datetime(env_df["timestamp"])
    fused_df: pd.DataFrame = pd.merge_asof(sensor_df.sort_values("timestamp"), env_df.sort_values("timestamp"),
                                           on="timestamp", direction="nearest")
    maintenance_df["timestamp"] = pd.to_datetime(maintenance_df["timestamp"])
    fused_df["date"] = fused_df["timestamp"].dt.floor("D")
    maintenance_daily: pd.DataFrame = maintenance_df.groupby("date")["equipment_failure"].max().reset_index()
    fused_df = pd.merge(fused_df, maintenance_daily, on="date", how="left")
    fused_df["equipment_failure"].fillna(0, inplace=True)

    # Preprocess fused data
    fused_df = preprocess_data(fused_df)
    fused_df = add_time_features(fused_df)
    fused_df = add_lag_features(fused_df, "sensor_1", [1, 2, 3])
    fused_df = add_rolling_features(fused_df, "sensor_1", [5, 10])

    # Label data for predictive maintenance (failure label based on sensor_1 threshold)
    fused_df = label_failure(fused_df, "sensor_1", threshold=1.0)

    # Anomaly detection on sensor data
    fused_df = detect_anomalies(fused_df, ["sensor_1", "sensor_2", "sensor_3"], contamination=0.02)

    # Split data for predictive maintenance model (LSTM)
    maintenance_data: pd.DataFrame = fused_df[
        ["timestamp", "sensor_1", "temperature", "humidity", "equipment_failure"]].copy()
    maintenance_data.sort_values("timestamp", inplace=True)
    maintenance_data.reset_index(drop=True, inplace=True)
    # Use last column as target (binary failure label)
    maintenance_data["failure_label"] = maintenance_data["equipment_failure"].astype(int)

    feature_cols: List[str] = ["sensor_1", "temperature", "humidity"]
    ml_df: pd.DataFrame = maintenance_data[feature_cols + ["failure_label"]].dropna()

    X: pd.DataFrame = ml_df[feature_cols]
    y: pd.Series = ml_df["failure_label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler: StandardScaler = StandardScaler()
    X_train_scaled: np.ndarray = scaler.fit_transform(X_train)
    X_test_scaled: np.ndarray = scaler.transform(X_test)

    # Build and train ensemble stacking model
    stacking_model = build_stacking_model(X_train, y_train)
    stack_preds: np.ndarray = stacking_model.predict(X_test)
    print("Stacking Model Accuracy:", accuracy_score(y_test, stack_preds))
    evaluate_model(y_test.to_numpy(), stack_preds)

    # Prepare dataset for LSTM (predict maintenance failure)
    lstm_features: List[str] = ["sensor_1", "temperature", "humidity"]
    lstm_df: pd.DataFrame = maintenance_data[lstm_features + ["failure_label"]].copy()
    lstm_df.dropna(inplace=True)
    time_steps: int = 10
    X_lstm, y_lstm = create_lstm_dataset(lstm_df, lstm_features, "failure_label", time_steps=time_steps)
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2,
                                                                            random_state=42, stratify=y_lstm)

    # Build and train LSTM model
    lstm_model: tf.keras.Model = build_lstm_model(input_dim=X_train_lstm.shape[1])
    lstm_model = train_lstm_model(lstm_model, X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm)
    lstm_preds: np.ndarray = (lstm_model.predict(X_test_lstm) > 0.5).astype(int).flatten()
    print("LSTM Model Accuracy:", accuracy_score(y_test_lstm, lstm_preds))

    # Forecast sensor_1 trend using ARIMA on fused data
    ts_df: pd.DataFrame = fused_df.set_index("timestamp")[["sensor_1"]].resample("H").mean().fillna(method="ffill")
    arima_forecast: pd.Series = forecast_arima(ts_df, "sensor_1", steps=24, order=(2, 1, 2))
    plot_time_series_forecast(ts_df["sensor_1"], arima_forecast)

    # Dimensionality reduction for visualization of fused features
    pca_model: PCA = PCA(n_components=2)
    pca_data: np.ndarray = pca_model.fit_transform(X_train_scaled)
    tsne_model = TSNE(n_components=2, random_state=42)
    tsne_data: np.ndarray = tsne_model.fit_transform(X_train_scaled)
    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=y_train.to_numpy(), cmap="coolwarm", alpha=0.7)
    plt.title("t-SNE Visualization of Maintenance Data")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()

    # Hyperparameter tuning with Optuna for RandomForest on scaled data
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: optuna_objective(trial, X_train_scaled, y_train.to_numpy()), n_trials=30)
    print("Optuna Best Params:", study.best_params)

    # SHAP explanation for stacking model
    shap_explain(stacking_model, pd.DataFrame(X_train_scaled, columns=X_train.columns))

    # LIME explanation for stacking model
    lime_explain(stacking_model, X_train_scaled, X_train.columns.tolist(), X_test_scaled)

    # Save models
    dump(stacking_model, "stacking_model_maintenance.joblib")
    dump(lstm_model, "lstm_model_maintenance.joblib")

    # End of advanced multimodal predictive maintenance pipeline.
