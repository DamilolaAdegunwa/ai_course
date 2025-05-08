# Project Title: cddml-AoP3zLw9Xf
# File Name: advanced_multisource_iot_predictive_maintenance.py

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import dask.dataframe as dd  # type: ignore
import modin.pandas as mpd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from datetime import datetime, timedelta  # type: ignore
from sklearn.model_selection import train_test_split, StratifiedKFold  # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder  # type: ignore
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
import tensorflow as tf  # type: ignore
import optuna  # type: ignore
from joblib import dump, load  # type: ignore
import shap  # type: ignore
import lime.lime_tabular  # type: ignore
from statsmodels.tsa.arima.model import ARIMA  # type: ignore
from prophet import Prophet  # type: ignore
import networkx as nx  # type: ignore


# Data Simulation for IoT Sensors
def simulate_iot_sensor_data(num_records: int = 10000) -> pd.DataFrame:
    np.random.seed(42)
    timestamps: pd.DatetimeIndex = pd.date_range(start="2023-01-01", periods=num_records, freq="H")
    sensor1: np.ndarray = np.random.normal(loc=50, scale=5, size=num_records)
    sensor2: np.ndarray = np.random.normal(loc=75, scale=7, size=num_records)
    sensor3: np.ndarray = np.random.normal(loc=100, scale=10, size=num_records)
    df: pd.DataFrame = pd.DataFrame({
        "timestamp": timestamps,
        "sensor1": sensor1,
        "sensor2": sensor2,
        "sensor3": sensor3
    })
    # Inject anomalies
    anomaly_indices: np.ndarray = np.random.choice(num_records, size=int(0.02 * num_records), replace=False)
    df.loc[anomaly_indices, "sensor1"] += np.random.uniform(20, 40, size=len(anomaly_indices))
    return df


# Simulate Maintenance Log Data
def simulate_maintenance_log(num_records: int = 500) -> pd.DataFrame:
    np.random.seed(24)
    timestamps: pd.DatetimeIndex = pd.date_range(start="2023-01-01", periods=num_records, freq="D")
    failure: np.ndarray = np.random.choice([0, 1], size=num_records, p=[0.9, 0.1])
    df: pd.DataFrame = pd.DataFrame({
        "timestamp": timestamps,
        "maintenance_required": failure
    })
    return df


# Simulate Environmental Data (Temperature, Humidity)
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


# Data Fusion: Merge IoT sensor, maintenance, and environmental data
def fuse_data(sensor_df: pd.DataFrame, maintenance_df: pd.DataFrame, env_df: pd.DataFrame) -> pd.DataFrame:
    sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])
    env_df["timestamp"] = pd.to_datetime(env_df["timestamp"])
    fused_df: pd.DataFrame = pd.merge_asof(sensor_df.sort_values("timestamp"), env_df.sort_values("timestamp"),
                                           on="timestamp", direction="nearest")
    maintenance_df["timestamp"] = pd.to_datetime(maintenance_df["timestamp"])
    fused_df["date"] = fused_df["timestamp"].dt.floor("D")
    maintenance_daily: pd.DataFrame = maintenance_df.groupby("date")["maintenance_required"].max().reset_index()
    fused_df = pd.merge(fused_df, maintenance_daily, on="date", how="left")
    fused_df["maintenance_required"].fillna(0, inplace=True)
    return fused_df


# Preprocess Data: Cleaning and scaling
def preprocess_fused_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    scaler: StandardScaler = StandardScaler()
    numeric_cols: List[str] = ["sensor1", "sensor2", "sensor3", "temperature", "humidity"]
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


# Add time-based features
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["timestamp"].dt.hour.astype(int)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype(int)
    df["month"] = df["timestamp"].dt.month.astype(int)
    return df


# Create lag features
def create_lag_features(df: pd.DataFrame, col: str, lags: List[int]) -> pd.DataFrame:
    for lag in lags:
        df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    df.dropna(inplace=True)
    return df


# Build predictive maintenance label based on sensor threshold
def label_maintenance(df: pd.DataFrame, col: str, threshold: float) -> pd.DataFrame:
    df["failure_label"] = (df[col] > threshold).astype(int)
    return df


# Anomaly detection using Isolation Forest
def detect_anomalies(df: pd.DataFrame, features: List[str], contamination: float = 0.02) -> pd.DataFrame:
    from sklearn.ensemble import IsolationForest
    iso: IsolationForest = IsolationForest(contamination=contamination, random_state=42)
    df["anomaly"] = iso.fit_predict(df[features])
    return df


# LSTM Model: Prepare data for sequential model
def create_lstm_dataset(df: pd.DataFrame, features: List[str], target: str, time_steps: int = 10) -> Tuple[
    np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(df) - time_steps):
        X.append(df[features].iloc[i:i + time_steps].values)
        y.append(df[target].iloc[i + time_steps])
    return np.array(X), np.array(y)


# Build LSTM model for predictive maintenance
def build_lstm_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    model: tf.keras.Model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model


# Train LSTM model
def train_lstm(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
               y_val: np.ndarray) -> tf.keras.Model:
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[early_stop],
              verbose=0)
    return model


# Build ensemble stacking classifier using RandomForest and XGBoost
def build_stacking_model(X_train: pd.DataFrame, y_train: pd.Series) -> object:
    from sklearn.ensemble import StackingClassifier
    base_estimators = [
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ("xgb", XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric="logloss", random_state=42))
    ]
    final_estimator = LogisticRegression()
    stack: StackingClassifier = StackingClassifier(estimators=base_estimators, final_estimator=final_estimator,
                                                   cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
    stack.fit(X_train, y_train)
    return stack


# Forecast sensor trend using ARIMA
def forecast_sensor_arima(df: pd.DataFrame, col: str, steps: int = 24, order: tuple = (2, 1, 2)) -> pd.Series:
    model: ARIMA = ARIMA(df[col], order=order)
    model_fit = model.fit()
    forecast: pd.Series = model_fit.forecast(steps=steps)
    return forecast


# Visualize forecast results
def plot_forecast(df: pd.DataFrame, col: str, forecast: pd.Series) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[col], label="Historical")
    forecast_index = pd.date_range(start=df.index[-1] + timedelta(hours=1), periods=len(forecast), freq="H")
    plt.plot(forecast_index, forecast, label="Forecast", linestyle="--", color="red")
    plt.xlabel("Timestamp")
    plt.ylabel(col)
    plt.title(f"ARIMA Forecast for {col}")
    plt.legend()
    plt.show()


# Dimensionality reduction for visualization using PCA and t-SNE
def reduce_dimensions(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    pca: PCA = PCA(n_components=50, random_state=42)
    X_pca: np.ndarray = pca.fit_transform(X)
    tsne: TSNE = TSNE(n_components=2, random_state=42)
    X_tsne: np.ndarray = tsne.fit_transform(X_pca)
    return X_pca, X_tsne


def plot_tsne(X_tsne: np.ndarray, labels: np.ndarray) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap="coolwarm", alpha=0.7)
    plt.title("t-SNE Visualization")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()


# Main Pipeline Execution
if __name__ == "__main__":
    current_timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Current Timestamp:", current_timestamp)

    # Simulate and fuse data
    sensor_data: pd.DataFrame = simulate_iot_sensor_data(num_records=5000)
    maintenance_log: pd.DataFrame = simulate_maintenance_log(num_records=200)
    env_data: pd.DataFrame = simulate_environmental_data(num_records=5000)
    fused_df: pd.DataFrame = fuse_data(sensor_data, maintenance_log, env_data)

    # Preprocess fused data and add time features
    fused_df = preprocess_fused_data(fused_df)
    fused_df = add_time_features(fused_df)
    fused_df = create_lag_features(fused_df, "sensor1", [1, 2, 3])
    fused_df = add_rolling_features(fused_df, "sensor1", [5, 10])
    fused_df = label_maintenance(fused_df, "sensor1", threshold=1.0)

    # Anomaly Detection on fused sensor data
    fused_df = detect_anomalies(fused_df, ["sensor1", "sensor2", "sensor3"], contamination=0.02)

    # Split data for predictive maintenance modeling
    maintenance_df: pd.DataFrame = fused_df[
        ["timestamp", "sensor1", "temperature", "humidity", "maintenance_required"]].copy()
    maintenance_df.sort_values("timestamp", inplace=True)
    maintenance_df.reset_index(drop=True, inplace=True)
    maintenance_df["failure_label"] = maintenance_df["maintenance_required"].astype(int)
    ml_df: pd.DataFrame = maintenance_df[["sensor1", "temperature", "humidity", "failure_label"]].dropna()
    X: pd.DataFrame = ml_df[["sensor1", "temperature", "humidity"]]
    y: pd.Series = ml_df["failure_label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler: StandardScaler = StandardScaler()
    X_train_scaled: np.ndarray = scaler.fit_transform(X_train)
    X_test_scaled: np.ndarray = scaler.transform(X_test)

    # Build and train stacking ensemble for maintenance prediction
    ensemble_model = build_stacking_model(X_train, y_train)
    ens_preds: np.ndarray = ensemble_model.predict(X_test)
    print("Ensemble Model Accuracy:", accuracy_score(y_test, ens_preds))

    # Prepare data for LSTM model
    lstm_df: pd.DataFrame = maintenance_df[["sensor1", "temperature", "humidity", "failure_label"]].dropna()
    time_steps: int = 10
    X_lstm, y_lstm = create_lstm_dataset(lstm_df, ["sensor1", "temperature", "humidity"], "failure_label",
                                         time_steps=time_steps)
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2,
                                                                            random_state=42, stratify=y_lstm)
    lstm_model: tf.keras.Model = build_lstm_model(input_shape=(time_steps, X_train_lstm.shape[2]))
    lstm_model = train_lstm_model(lstm_model, X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm)
    lstm_preds: np.ndarray = (lstm_model.predict(X_test_lstm) > 0.5).astype(int).flatten()
    print("LSTM Model Accuracy:", accuracy_score(y_test_lstm, lstm_preds))

    # Forecast sensor1 trend using ARIMA
    ts_df: pd.DataFrame = fused_df.set_index("timestamp")[["sensor1"]].resample("H").mean().fillna(method="ffill")
    sensor_forecast: pd.Series = forecast_sensor_arima(ts_df, "sensor1", steps=24, order=(2, 1, 2))
    plot_forecast(ts_df, "sensor1", sensor_forecast)

    # Dimensionality reduction for visualization (PCA + t-SNE)
    X_pca, X_tsne = reduce_dimensions(X_train_scaled)
    plot_tsne(X_tsne, labels=y_train.to_numpy())


    # Hyperparameter tuning with Optuna for RandomForest on sensor data
    def optuna_objective(trial, X, y):
        n_estimators: int = trial.suggest_int("n_estimators", 100, 500)
        max_depth: int = trial.suggest_int("max_depth", 3, 15)
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_val)
            scores.append(accuracy_score(y_val, preds))
        return np.mean(scores)


    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: optuna_objective(trial, X_train_scaled, y_train.to_numpy()), n_trials=30)
    print("Optuna Best Params:", study.best_params)

    # SHAP and LIME explainability for ensemble model
    shap_explain(ensemble_model, pd.DataFrame(X_train_scaled, columns=X_train.columns))
    lime_explain(ensemble_model, X_train_scaled, X_train.columns.tolist(), X_test_scaled)

    # Save models
    dump(ensemble_model, "ensemble_model_maintenance.joblib")
    dump(lstm_model, "lstm_model_maintenance.joblib")

    # End of advanced predictive maintenance pipeline.
