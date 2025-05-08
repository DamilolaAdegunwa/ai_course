# cddml-Ze9XcTqR5Lp
# File Name: advanced_genomic_variant_classification_and_network_analysis_with_pandas.py

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore
from sklearn.model_selection import train_test_split, StratifiedKFold  # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.svm import SVC  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # type: ignore
from sklearn.ensemble import StackingClassifier  # type: ignore
from joblib import dump, load  # type: ignore
from typing import List, Tuple, Dict, Any
import datetime  # type: ignore
import optuna  # type: ignore


# Simulate genomic variant dataset
def simulate_variant_data(num_variants: int = 1000) -> pd.DataFrame:
    np.random.seed(42)
    variant_ids: np.ndarray = np.arange(1, num_variants + 1)
    genes: List[str] = [f"Gene_{i}" for i in np.random.randint(1, 101, num_variants)]
    allele_freqs: np.ndarray = np.random.uniform(0.001, 0.05, num_variants)
    consequences: List[str] = np.random.choice(["missense", "synonymous", "nonsense", "frameshift"],
                                               num_variants).tolist()
    impacts: List[str] = np.random.choice(["high", "moderate", "low"], num_variants).tolist()
    # Label: 1 if pathogenic, 0 if benign, simulate with some randomness
    labels: np.ndarray = np.random.choice([0, 1], num_variants, p=[0.8, 0.2])
    df: pd.DataFrame = pd.DataFrame({
        "variant_id": variant_ids,
        "gene": genes,
        "allele_frequency": allele_freqs,
        "consequence": consequences,
        "impact": impacts,
        "label": labels
    })
    return df


# Preprocess variant data: encode categorical, scale numerical
def preprocess_variant_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    le_consequence: LabelEncoder = LabelEncoder()
    df["consequence_encoded"] = le_consequence.fit_transform(df["consequence"])
    le_impact: LabelEncoder = LabelEncoder()
    df["impact_encoded"] = le_impact.fit_transform(df["impact"])
    scaler: StandardScaler = StandardScaler()
    df["allele_frequency_scaled"] = scaler.fit_transform(df[["allele_frequency"]])
    df["combined_feature"] = df["allele_frequency_scaled"] * df["impact_encoded"].astype(float)
    return df


# Split data for modeling
def split_variant_data(df: pd.DataFrame, target: str = "label") -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X: pd.DataFrame = df[["allele_frequency_scaled", "consequence_encoded", "impact_encoded", "combined_feature"]]
    y: pd.Series = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Build stacking ensemble classifier for variant classification
def build_variant_stacking_model(X_train: pd.DataFrame, y_train: pd.Series) -> StackingClassifier:
    estimators: List[Tuple[str, Any]] = [
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ("svm", SVC(probability=True, random_state=42))
    ]
    meta_estimator: LogisticRegression = LogisticRegression()
    stack: StackingClassifier = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_estimator,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    )
    stack.fit(X_train, y_train)
    return stack


# Evaluate model performance
def evaluate_model(y_true: pd.Series, y_pred: np.ndarray) -> None:
    acc: float = accuracy_score(y_true, y_pred)
    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_true, y_pred))
    cm: np.ndarray = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


# Build knowledge graph from variant data (group by gene)
def build_gene_variant_graph(df: pd.DataFrame) -> nx.Graph:
    G: nx.Graph = nx.Graph()
    for gene in df["gene"].unique():
        variants: pd.DataFrame = df[df["gene"] == gene]
        variant_ids: List[int] = variants["variant_id"].tolist()
        for i in range(len(variant_ids)):
            for j in range(i + 1, len(variant_ids)):
                G.add_edge(variant_ids[i], variant_ids[j], gene=gene)
    return G


# Query the knowledge graph for a given gene's variants
def query_gene_graph(G: nx.Graph, gene: str, df: pd.DataFrame) -> List[int]:
    variant_ids: List[int] = df[df["gene"] == gene]["variant_id"].tolist()
    subgraph: nx.Graph = G.subgraph(variant_ids)
    return list(subgraph.nodes())


# Hyperparameter tuning using Optuna for RandomForest
def optuna_objective(trial: optuna.trial.Trial, X: np.ndarray, y: np.ndarray) -> float:
    n_estimators: int = trial.suggest_int("n_estimators", 100, 500)
    max_depth: int = trial.suggest_int("max_depth", 3, 15)
    clf: RandomForestClassifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                         random_state=42)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores: List[float] = []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[test_idx]
        y_tr, y_val = y[train_idx], y[test_idx]
        clf.fit(X_tr, y_tr)
        preds: np.ndarray = clf.predict(X_val)
        scores.append(accuracy_score(y_val, preds))
    return np.mean(scores)


def run_optuna(X: np.ndarray, y: np.ndarray) -> optuna.study.Study:
    study: optuna.study.Study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: optuna_objective(trial, X, y), n_trials=30)
    return study


# Main Pipeline Execution
if __name__ == "__main__":
    # Timestamp and project metadata
    current_timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Current Timestamp:", current_timestamp)

    # Simulate and preprocess genomic variant dataset
    variant_df: pd.DataFrame = simulate_variant_data(num_variants=1000)
    variant_df: pd.DataFrame = preprocess_variant_data(variant_df)

    # Split data for classification
    X_train_df, X_test_df, y_train, y_test = split_variant_data(variant_df, target="label")

    # Build stacking ensemble classifier
    stacking_model = build_variant_stacking_model(X_train_df, y_train)
    stack_preds: np.ndarray = stacking_model.predict(X_test_df)
    print("Stacking Model Accuracy:", accuracy_score(y_test, stack_preds))
    evaluate_model(y_test, stack_preds)

    # Hyperparameter tuning with Optuna on RandomForest as a baseline
    X_train_np: np.ndarray = X_train_df.values
    y_train_np: np.ndarray = y_train.values
    study: optuna.study.Study = run_optuna(X_train_np, y_train_np)
    print("Optuna Best Params:", study.best_params)

    # Build knowledge graph from variant data
    gene_graph: nx.Graph = build_gene_variant_graph(variant_df)
    queried_variants: List[int] = query_gene_graph(gene_graph, gene="Gene_10", df=variant_df)
    print("Variants in Gene_10:", queried_variants)

    # LSTM for sequential prediction on a simulated time series (using variant combined feature as proxy)
    ts_df: pd.DataFrame = variant_df[["variant_id", "combined_feature"]].copy()
    ts_df.sort_values("variant_id", inplace=True)
    ts_df = ts_df.set_index("variant_id")
    ts_df = ts_df.rename(columns={"combined_feature": "value"})


    # Create time series dataset for LSTM prediction
    def create_sequence_data(df: pd.DataFrame, time_steps: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        X_seq, y_seq = [], []
        for i in range(len(df) - time_steps):
            X_seq.append(df.iloc[i:i + time_steps].values)
            y_seq.append(df.iloc[i + time_steps].values[0])
        return np.array(X_seq), np.array(y_seq)


    X_seq, y_seq = create_sequence_data(ts_df, time_steps=10)
    X_train_seq, X_test_seq, y_train_seq, y_test_seq = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    lstm_model: tf.keras.Model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
        BatchNormalization(),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dense(1, activation="linear")
    ])
    lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    lstm_model.fit(X_train_seq, y_train_seq, validation_data=(X_test_seq, y_test_seq), epochs=50, batch_size=32,
                   callbacks=[early_stop], verbose=0)
    lstm_preds: np.ndarray = lstm_model.predict(X_test_seq)
    mse_value: float = mean_squared_error(y_test_seq, lstm_preds)
    print("LSTM Model MSE on Time Series:", mse_value)

    # Forecast using Prophet on simulated sensor data trend
    sensor_ts_df: pd.DataFrame = simulate_iot_sensor_data(num_records=1000)[["timestamp", "sensor1"]].copy()
    sensor_ts_df["timestamp"] = pd.to_datetime(sensor_ts_df["timestamp"])
    sensor_ts_df = sensor_ts_df.sort_values("timestamp").set_index("timestamp")
    sensor_ts_df = sensor_ts_df.resample("H").mean().fillna(method="ffill")
    prophet_df: pd.DataFrame = sensor_ts_df.reset_index().rename(columns={"timestamp": "ds", "sensor1": "y"})
    prophet_model: Prophet = Prophet()
    prophet_model.fit(prophet_df)
    future_df: pd.DataFrame = prophet_model.make_future_dataframe(periods=48, freq="H")
    forecast: pd.DataFrame = prophet_model.predict(future_df)
    fig = prophet_model.plot(forecast)
    plt.title("Prophet Forecast for Sensor1")
    plt.show()

    # SHAP analysis for stacking model
    shap_explain(stacking_model, pd.DataFrame(X_train_df.values, columns=X_train_df.columns))

    # LIME explanation for stacking model on first test instance
    lime_explain(stacking_model, X_train_df.values, X_train_df.columns.tolist(), X_test_df.values)

    # Save models
    dump(stacking_model, "stacking_variant_model.joblib")
    dump(lstm_model, "lstm_variant_model.joblib")

    # End of pipeline for advanced multimodal predictive maintenance and genomic variant time series analysis.
