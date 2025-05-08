# cddml-GH4kL2NmPq
# File Name: advanced_multimodal_healthcare_outcome_prediction.py

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import datetime  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.model_selection import train_test_split, StratifiedKFold  # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder  # type: ignore
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
import shap  # type: ignore
import lime.lime_tabular  # type: ignore
import optuna  # type: ignore
from joblib import dump, load  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from typing import List, Tuple, Dict, Any


def simulate_clinical_data(num_patients: int = 1000) -> pd.DataFrame:
    np.random.seed(42)
    patient_ids: np.ndarray = np.arange(1, num_patients + 1)
    age: np.ndarray = np.random.randint(20, 90, num_patients)
    bmi: np.ndarray = np.random.normal(25, 4, num_patients)
    blood_pressure: np.ndarray = np.random.normal(120, 15, num_patients)
    outcome: np.ndarray = np.random.choice([0, 1], num_patients, p=[0.7, 0.3])
    df: pd.DataFrame = pd.DataFrame({
        "patient_id": patient_ids,
        "age": age,
        "bmi": bmi,
        "blood_pressure": blood_pressure,
        "clinical_outcome": outcome
    })
    return df


def simulate_genomic_data(num_patients: int = 1000) -> pd.DataFrame:
    np.random.seed(24)
    patient_ids: np.ndarray = np.arange(1, num_patients + 1)
    variant_score: np.ndarray = np.random.uniform(0, 1, num_patients)
    gene_expression: np.ndarray = np.random.normal(0, 1, num_patients)
    df: pd.DataFrame = pd.DataFrame({
        "patient_id": patient_ids,
        "variant_score": variant_score,
        "gene_expression": gene_expression
    })
    return df


def simulate_imaging_data(num_patients: int = 1000) -> pd.DataFrame:
    np.random.seed(99)
    patient_ids: np.ndarray = np.arange(1, num_patients + 1)
    imaging_feature1: np.ndarray = np.random.normal(0, 1, num_patients)
    imaging_feature2: np.ndarray = np.random.normal(0, 1, num_patients)
    df: pd.DataFrame = pd.DataFrame({
        "patient_id": patient_ids,
        "imaging_feature1": imaging_feature1,
        "imaging_feature2": imaging_feature2
    })
    return df


def merge_multimodal_data(clinical_df: pd.DataFrame, genomic_df: pd.DataFrame,
                          imaging_df: pd.DataFrame) -> pd.DataFrame:
    df: pd.DataFrame = pd.merge(clinical_df, genomic_df, on="patient_id", how="inner")
    df = pd.merge(df, imaging_df, on="patient_id", how="inner")
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        le: LabelEncoder = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    scaler: StandardScaler = StandardScaler()
    num_cols: List[str] = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    num_cols.remove("clinical_outcome")
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df["age_bmi_interaction"] = df["age"] * df["bmi"]
    df["bp_variant_ratio"] = df["blood_pressure"] / (df["variant_score"] + 1e-5)
    poly_features: pd.DataFrame = pd.DataFrame(
        np.power(df[["gene_expression", "imaging_feature1", "imaging_feature2"]], 2),
        columns=["gene_expression_sq", "imaging_feature1_sq", "imaging_feature2_sq"])
    df = pd.concat([df, poly_features], axis=1)
    return df


def split_dataset(df: pd.DataFrame, target: str = "clinical_outcome") -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X: pd.DataFrame = df.drop(columns=[target, "patient_id"])
    y: pd.Series = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def build_stacking_model(X_train: pd.DataFrame, y_train: pd.Series) -> StackingClassifier:
    base_estimators: List[Tuple[str, Any]] = [
        ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
        ("gb", GradientBoostingClassifier(n_estimators=200, random_state=42)),
        ("xgb", XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric="logloss", random_state=42))
    ]
    final_estimator: LogisticRegression = LogisticRegression()
    skf: StratifiedKFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stack: StackingClassifier = StackingClassifier(estimators=base_estimators, final_estimator=final_estimator, cv=skf)
    stack.fit(X_train, y_train)
    return stack


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


def shap_explain_model(model: Any, X_sample: pd.DataFrame) -> None:
    explainer = shap.Explainer(model.predict, X_sample)
    shap_values = explainer(X_sample)
    shap.summary_plot(shap_values, X_sample)


def lime_explain_model(model: Any, X_train: np.ndarray, feature_names: List[str], X_test: np.ndarray) -> None:
    import lime.lime_tabular as lime_tabular
    explainer = lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names,
                                                  class_names=["benign", "pathogenic"], discretize_continuous=True)
    exp = explainer.explain_instance(X_test[0], model.predict_proba, num_features=10)
    exp.show_in_notebook()


def build_deep_learning_model(input_dim: int) -> tf.keras.Model:
    model: tf.keras.Model = Sequential([
        Dense(256, activation="relu", input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_deep_model(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
                     y_val: np.ndarray) -> tf.keras.Model:
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[early_stop],
              verbose=0)
    return model


def optuna_objective(trial: optuna.trial.Trial, X: np.ndarray, y: np.ndarray) -> float:
    n_estimators: int = trial.suggest_int("n_estimators", 100, 500)
    max_depth: int = trial.suggest_int("max_depth", 3, 15)
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores: List[float] = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        clf.fit(X_tr, y_tr)
        preds: np.ndarray = clf.predict(X_val)
        scores.append(accuracy_score(y_val, preds))
    return np.mean(scores)


def run_optuna_tuning(X: np.ndarray, y: np.ndarray) -> optuna.study.Study:
    study: optuna.study.Study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: optuna_objective(trial, X, y), n_trials=30)
    return study


# KEY LEARNINGS / RESEARCH AREAS
key_learnings: Dict[str, List[str]] = {
    "Features": ["Data Fusion", "Multi-Modal Integration", "Feature Engineering", "Ensemble Learning", "Deep Learning",
                 "Explainable AI"],
    "Components": ["Pandas", "Dask", "Modin", "Scikit-Learn", "TensorFlow", "Optuna", "SHAP", "LIME", "NetworkX"],
    "Keywords": ["Healthcare Outcome Prediction", "Multi-Modal Data", "Genomic Data", "Clinical Data", "Imaging Data",
                 "Predictive Analytics"],
    "Research Areas": ["Precision Medicine", "Data Integration", "Ensemble Learning", "Model Explainability",
                       "Automated Machine Learning"],
    "Hashtags": ["#DataFusion", "#MultiModalML", "#PredictiveAnalytics", "#HealthcareAI", "#ExplainableAI"]
}

# Main Pipeline Execution
if __name__ == "__main__":
    current_timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Current Timestamp:", current_timestamp)

    # Simulate Data from Different Modalities
    clinical_df: pd.DataFrame = simulate_clinical_data(num_patients=1000)
    genomic_df: pd.DataFrame = simulate_genomic_data(num_patients=1000)
    imaging_df: pd.DataFrame = simulate_imaging_data(num_patients=1000)

    # Merge datasets
    merged_df: pd.DataFrame = merge_multimodal_data(clinical_df, genomic_df, imaging_df)
    print("Merged Data Shape:", merged_df.shape)

    # Preprocess and feature engineer
    processed_df: pd.DataFrame = preprocess_data(merged_df)
    processed_df: pd.DataFrame = feature_engineering(processed_df)

    # Split data for classification (predict clinical outcome)
    X_train_df, X_test_df, y_train, y_test = split_dataset(processed_df, target="clinical_outcome")
    X_train_scaled: np.ndarray;
    X_test_scaled: np.ndarray;
    scaler: StandardScaler = scale_features(X_train_df, X_test_df)

    # Build and evaluate stacking ensemble classifier
    stacking_model = build_stacking_model(X_train_df, y_train)
    stack_preds: np.ndarray = stacking_model.predict(X_test_df)
    print("Stacking Ensemble Accuracy:", accuracy_score(y_test, stack_preds))
    evaluate_model(y_test, stack_preds)

    # Hyperparameter tuning with Optuna
    study = run_optuna_tuning(X_train_scaled, y_train.to_numpy())
    print("Optuna Best Parameters:", study.best_params)

    # SHAP explanation for stacking model
    shap_explain_model(stacking_model, pd.DataFrame(X_train_scaled, columns=X_train_df.columns))

    # LIME explanation for stacking model on one test instance
    lime_explain_model(stacking_model, X_train_scaled, X_train_df.columns.tolist(), X_test_scaled)

    # Build and train a deep learning model
    X_train_dl, X_val_dl, y_train_dl, y_val_dl = train_test_split(X_train_scaled, y_train.to_numpy(), test_size=0.2,
                                                                  random_state=42, stratify=y_train)
    dl_model: tf.keras.Model = build_deep_learning_model(input_dim=X_train_scaled.shape[1])
    dl_model = train_deep_model(dl_model, X_train_dl, y_train_dl, X_val_dl, y_val_dl)
    dl_preds: np.ndarray = (dl_model.predict(X_test_scaled) > 0.5).astype(int).flatten()
    print("Deep Learning Model Accuracy:", accuracy_score(y_test.to_numpy(), dl_preds))

    # Dimensionality reduction for visualization
    pca_model: PCA = PCA(n_components=2, random_state=42)
    X_pca: np.ndarray = pca_model.fit_transform(X_train_scaled)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_train, palette="viridis", alpha=0.7)
    plt.title("PCA Visualization of Training Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()

    # Build knowledge graph from feature correlations
    corr_matrix: pd.DataFrame = X_train_df.corr()
    G: nx.Graph = nx.Graph()
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                G.add_edge(corr_matrix.columns[i], corr_matrix.columns[j], weight=corr_matrix.iloc[i, j])
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=800, node_color="lightblue", edge_color="gray")
    plt.title("Knowledge Graph of Feature Correlations")
    plt.show()

    # Save final models
    dump(stacking_model, "final_stacking_model.joblib")
    dump(dl_model, "final_deep_model.joblib")

    # Final key learnings as a Python dictionary
    key_learnings: Dict[str, List[str]] = {
        "Features": ["Data Fusion", "Multi-Modal Integration", "Feature Engineering", "Ensemble Learning",
                     "Deep Learning", "Explainable AI", "Knowledge Graph Construction"],
        "Components": ["Pandas", "Dask", "Modin", "Scikit-Learn", "TensorFlow", "Optuna", "SHAP", "LIME", "NetworkX"],
        "Keywords": ["Healthcare Outcome Prediction", "Multi-Modal Data", "Genomic Data", "Clinical Data",
                     "Imaging Data", "Predictive Analytics", "Time Series", "Ensemble Learning"],
        "Research Areas": ["Precision Medicine", "Data Integration", "Automated Machine Learning (AutoML)",
                           "Explainable AI", "Graph Analytics"],
        "Hashtags": ["#DataFusion", "#MultiModalML", "#PredictiveAnalytics", "#HealthcareAI", "#ExplainableAI"]
    }
    print("Key Learnings:\n", key_learnings)

    # Timestamp and project metadata
    project_timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Project Timestamp:", project_timestamp)

    # End of advanced multimodal healthcare outcome prediction pipeline.
