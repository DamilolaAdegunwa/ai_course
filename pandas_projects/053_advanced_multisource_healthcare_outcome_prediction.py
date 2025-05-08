# cddml-9KzXGf3RmLt
# File Name: advanced_multisource_healthcare_outcome_prediction.py

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.datasets import load_diabetes  # type: ignore
from sklearn.model_selection import train_test_split, StratifiedKFold  # type: ignore
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from xgboost import XGBClassifier  # type: ignore
from lightgbm import LGBMClassifier  # type: ignore
from catboost import CatBoostClassifier  # type: ignore
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, \
    roc_curve  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
import optuna  # type: ignore
from optuna.integration import TFKerasPruningCallback  # type: ignore
import shap  # type: ignore
import lime
import lime.lime_tabular  # type: ignore
from joblib import dump, load  # type: ignore
import datetime  # type: ignore
import warnings  # type: ignore

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)


def load_healthcare_data() -> pd.DataFrame:
    data: dict = load_diabetes(return_X_y=False)
    df: pd.DataFrame = pd.DataFrame(data=data["data"], columns=data["feature_names"])
    df["target"] = data["target"]
    df["age_cat"] = pd.cut(df["age"], bins=[0, 0.5, 1.0], labels=["young", "old"])
    return df


def clean_and_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        le: LabelEncoder = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    scaler: StandardScaler = StandardScaler()
    num_cols: list = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    poly: PolynomialFeatures = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    num_cols: list = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    num_cols.remove("target")
    poly_features: np.ndarray = poly.fit_transform(df[num_cols])
    poly_feature_names: list = poly.get_feature_names_out(num_cols).tolist()
    poly_df: pd.DataFrame = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
    df = pd.concat([df, poly_df], axis=1)
    return df


def split_dataset(df: pd.DataFrame, target: str) -> tuple:
    X: pd.DataFrame = df.drop(columns=[target])
    y: pd.Series = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def build_deep_learning_model(input_dim: int) -> tf.keras.Model:
    model: tf.keras.Model = Sequential([
        Dense(256, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_deep_model(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
                     y_val: np.ndarray) -> tf.keras.Model:
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[early_stop],
              verbose=0)
    return model


def build_ensemble_model(X_train: pd.DataFrame, y_train: pd.Series) -> object:
    from sklearn.ensemble import StackingClassifier
    estimators: list = [
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=42)),
        ('lgb', LGBMClassifier(n_estimators=200, random_state=42)),
        ('cat', CatBoostClassifier(iterations=200, verbose=0, random_state=42))
    ]
    final_estimator: LogisticRegression = LogisticRegression()
    stack: StackingClassifier = StackingClassifier(estimators=estimators, final_estimator=final_estimator,
                                                   cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
    stack.fit(X_train, y_train)
    return stack


def optuna_objective(trial: optuna.trial.Trial, X: np.ndarray, y: np.ndarray) -> float:
    n_estimators: int = trial.suggest_int("n_estimators", 100, 500)
    max_depth: int = trial.suggest_int("max_depth", 3, 15)
    clf: RandomForestClassifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                         random_state=42)
    skf: StratifiedKFold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores: list = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        clf.fit(X_tr, y_tr)
        preds: np.ndarray = clf.predict(X_val)
        scores.append(accuracy_score(y_val, preds))
    return np.mean(scores)


def run_optuna_optimization(X_train: np.ndarray, y_train: np.ndarray) -> optuna.study.Study:
    study: optuna.study.Study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: optuna_objective(trial, X_train, y_train), n_trials=30)
    return study


def shap_explain(model: object, X_sample: pd.DataFrame) -> None:
    explainer = shap.Explainer(model.predict, X_sample)
    shap_values = explainer(X_sample)
    shap.summary_plot(shap_values, X_sample)


def lime_explain(model: object, X_train: np.ndarray, feature_names: list, X_test: np.ndarray) -> None:
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, class_names=["0", "1"],
                                                       discretize_continuous=True)
    exp = explainer.explain_instance(X_test[0], model.predict_proba, num_features=10)
    exp.show_in_notebook()


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


def save_model(model: object, filename: str) -> None:
    dump(model, filename)


def load_model(filename: str) -> object:
    return load(filename)


if __name__ == "__main__":
    # Current timestamp and project info
    timestamp: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Timestamp: {timestamp}")

    # Load and preprocess dataset
    df: pd.DataFrame = load_dataset()
    df: pd.DataFrame = preprocess_dataset(df)
    df: pd.DataFrame = feature_engineering(df)
    X_train_df, X_test_df, y_train, y_test = split_dataset(df, target="class")

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_df, X_test_df)

    # Deep Learning Model Training
    X_train_dl, X_val_dl, y_train_dl, y_val_dl = train_test_split(X_train_scaled, y_train.to_numpy(), test_size=0.2,
                                                                  random_state=42, stratify=y_train)
    dl_model: tf.keras.Model = build_deep_learning_model(input_dim=X_train_dl.shape[1])
    dl_model = train_deep_model(dl_model, X_train_dl, y_train_dl, X_val_dl, y_val_dl)
    dl_preds: np.ndarray = (dl_model.predict(X_test_scaled) > 0.5).astype(int).flatten()
    print("Deep Learning Accuracy:", accuracy_score(y_test.to_numpy(), dl_preds))

    # Ensemble Model Training using Stacking
    ensemble_model = build_ensemble_model(X_train_df, y_train)
    ens_preds: np.ndarray = ensemble_model.predict(X_test_df)
    print("Ensemble Model Accuracy:", accuracy_score(y_test, ens_preds))

    # Hyperparameter tuning with Optuna
    study: optuna.study.Study = run_optuna_optimization(X_train_scaled, y_train.to_numpy())
    print("Best Optuna Params:", study.best_params)

    # SHAP explanation for Ensemble model
    shap_explain(ensemble_model, pd.DataFrame(X_train_scaled, columns=X_train_df.columns))

    # LIME explanation for Ensemble model (using first test instance)
    lime_explain(ensemble_model, X_train_scaled, X_train_df.columns.tolist(), X_test_scaled)

    # Stacking Ensemble with Deep Learning as Meta-learner
    from sklearn.ensemble import StackingClassifier

    base_estimators = [
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=42))
    ]
    meta_estimator = LogisticRegression()
    stacking_model = StackingClassifier(estimators=base_estimators, final_estimator=meta_estimator,
                                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
    stacking_model.fit(X_train_df, y_train)
    stack_preds: np.ndarray = stacking_model.predict(X_test_df)
    print("Stacking Model Accuracy:", accuracy_score(y_test, stack_preds))

    # Dimensionality Reduction for visualization
    pca = PCA(n_components=0.95)
    X_train_pca: np.ndarray = pca.fit_transform(X_train_scaled)
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, random_state=42)
    X_train_tsne: np.ndarray = tsne.fit_transform(X_train_pca)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=y_train.astype(int), cmap="viridis", alpha=0.7)
    plt.title("t-SNE Visualization of Training Data")
    plt.show()

    # Save and load models
    save_model(ensemble_model, "ensemble_model.joblib")
    loaded_ensemble = load_model("ensemble_model.joblib")
    loaded_preds = loaded_ensemble.predict(X_test_df)
    print("Loaded Ensemble Model Accuracy:", accuracy_score(y_test, loaded_preds))

    # Plot confusion matrix for stacking model
    plot_confusion(y_test.to_numpy(), stack_preds)

    # Save final stacking model
    dump(stacking_model, "stacking_model.joblib")

    # End of pipeline
