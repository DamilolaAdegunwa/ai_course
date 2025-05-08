# advanced_multimodal_ensemble_ml_pipeline.py

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.datasets import fetch_openml  # type: ignore
from sklearn.model_selection import train_test_split, StratifiedKFold  # type: ignore
from sklearn.preprocessing import StandardScaler, LabelEncoder  # type: ignore
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # type: ignore
from xgboost import XGBClassifier  # type: ignore
from lightgbm import LGBMClassifier  # type: ignore
from catboost import CatBoostClassifier  # type: ignore
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


def load_dataset() -> pd.DataFrame:
    df: pd.DataFrame = fetch_openml(name="adult", version=2, as_frame=True).frame
    return df


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df: pd.DataFrame = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df.replace("?", np.nan, inplace=True)
    df.dropna(inplace=True)
    for col in df.select_dtypes(include=["object"]).columns:
        le: LabelEncoder = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df


def split_dataset(df: pd.DataFrame, target: str) -> tuple:
    X: pd.DataFrame = df.drop(columns=[target])
    y: pd.Series = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    scaler: StandardScaler = StandardScaler()
    X_train_scaled: np.ndarray = scaler.fit_transform(X_train)
    X_test_scaled: np.ndarray = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def build_deep_learning_model(input_dim: int) -> tf.keras.Model:
    model: tf.keras.Model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_deep_learning_model(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
                              y_val: np.ndarray) -> tf.keras.Model:
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32,
                        callbacks=[early_stopping], verbose=0)
    return model


def build_ensemble_model(X_train: pd.DataFrame, y_train: pd.Series) -> StackingClassifier:
    estimators: list = [
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=200, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=42)),
        ('lgb', LGBMClassifier(n_estimators=200, random_state=42)),
        ('cat', CatBoostClassifier(iterations=200, verbose=0, random_state=42))
    ]
    meta_model: LogisticRegression = LogisticRegression()
    ensemble: StackingClassifier = StackingClassifier(estimators=estimators, final_estimator=meta_model,
                                                      cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
    ensemble.fit(X_train, y_train)
    return ensemble


def perform_optuna_optimization(X_train: np.ndarray, y_train: np.ndarray) -> optuna.study.Study:
    def objective(trial: optuna.trial.Trial) -> float:
        n_estimators: int = trial.suggest_int('n_estimators', 100, 500)
        max_depth: int = trial.suggest_int('max_depth', 3, 10)
        clf: RandomForestClassifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                             random_state=42)
        cv_scores: np.ndarray = []
        for train_index, valid_index in StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(X_train,
                                                                                                         y_train):
            X_tr, X_val = X_train[train_index], X_train[valid_index]
            y_tr, y_val = y_train[train_index], y_train[valid_index]
            clf.fit(X_tr, y_tr)
            preds: np.ndarray = clf.predict(X_val)
            score: float = accuracy_score(y_val, preds)
            cv_scores.append(score)
        return np.mean(cv_scores)

    study: optuna.study.Study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    return study


def explain_model_shap(model: Any, X_sample: pd.DataFrame) -> None:
    explainer = shap.Explainer(model.predict, X_sample)
    shap_values = explainer(X_sample)
    shap.summary_plot(shap_values, X_sample)


def explain_model_lime(model: Any, X_train: np.ndarray, feature_names: List[str], X_test: np.ndarray) -> None:
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, class_names=['0', '1'],
                                                       discretize_continuous=True)
    exp = explainer.explain_instance(X_test[0], model.predict_proba, num_features=10)
    exp.show_in_notebook()


def plot_model_performance(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
    print(classification_report(y_true, y_pred))


def save_model(model: Any, filename: str) -> None:
    dump(model, filename)


def load_model(filename: str) -> Any:
    return load(filename)


if __name__ == "__main__":
    # Timestamp and Setup
    current_time: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Current Timestamp: {current_time}")

    # Load and preprocess data
    df: pd.DataFrame = load_dataset()
    df: pd.DataFrame = preprocess_dataset(df)
    X_train_df: pd.DataFrame;
    X_test_df: pd.DataFrame;
    y_train: pd.Series;
    y_test: pd.Series = split_dataset(df, target="class")
    X_train_scaled: np.ndarray;
    X_test_scaled: np.ndarray;
    scaler: StandardScaler = scale_features(X_train_df, X_test_df)

    # Deep Learning Model Training
    X_train_dl, X_val_dl, y_train_dl, y_val_dl = train_test_split(X_train_scaled, y_train, test_size=0.2,
                                                                  random_state=42, stratify=y_train)
    dl_model: tf.keras.Model = build_deep_learning_model(input_dim=X_train_dl.shape[1])
    dl_model = train_deep_learning_model(dl_model, X_train_dl, y_train_dl, X_val_dl, y_val_dl)
    dl_preds: np.ndarray = (dl_model.predict(X_test_scaled) > 0.5).astype(int).flatten()
    print("Deep Learning Model Accuracy:", accuracy_score(y_test, dl_preds))

    # Ensemble Model Training
    ensemble_model: StackingClassifier = build_ensemble_model(X_train_df, y_train)
    ens_preds: np.ndarray = ensemble_model.predict(X_test_df)
    print("Ensemble Model Accuracy:", accuracy_score(y_test, ens_preds))

    # Hyperparameter Tuning with Optuna
    study: optuna.study.Study = perform_optuna_optimization(X_train_scaled, y_train.to_numpy())
    print("Optuna Best Params:", study.best_params)

    # SHAP Explanation for Ensemble Model
    explain_model_shap(ensemble_model, pd.DataFrame(X_train_scaled, columns=X_train_df.columns))

    # LIME Explanation for Ensemble Model
    explain_model_lime(ensemble_model, X_train_scaled, X_train_df.columns.tolist(), X_test_scaled)

    # Train a Stacking Ensemble with Deep Learning as Meta-learner
    base_estimators: list = [
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=200, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=42))
    ]
    meta_estimator: LogisticRegression = LogisticRegression()
    stacking_model: StackingClassifier = StackingClassifier(estimators=base_estimators, final_estimator=meta_estimator,
                                                            cv=StratifiedKFold(n_splits=5, shuffle=True,
                                                                               random_state=42))
    stacking_model.fit(X_train_df, y_train)
    stack_preds: np.ndarray = stacking_model.predict(X_test_df)
    print("Stacking Model Accuracy:", accuracy_score(y_test, stack_preds))

    # Dimensionality Reduction for Visualization (PCA & t-SNE)
    pca_model = PCA(n_components=0.95)
    X_train_pca: np.ndarray = pca_model.fit_transform(X_train_scaled)
    from sklearn.manifold import TSNE

    tsne_model = TSNE(n_components=2, random_state=42)
    X_train_tsne: np.ndarray = tsne_model.fit_transform(X_train_pca)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=y_train.astype(int), cmap="viridis", alpha=0.5)
    plt.title("t-SNE Visualization of Training Data")
    plt.show()

    # Save and Load Ensemble Model
    save_model(ensemble_model, "ensemble_model.joblib")
    loaded_model = load_model("ensemble_model.joblib")
    loaded_preds = loaded_model.predict(X_test_df)
    print("Loaded Model Accuracy:", accuracy_score(y_test, loaded_preds))

    # Plot performance of Ensemble Model
    plot_model_performance(y_test.to_numpy(), ens_preds)

    # Final Model Export
    dump(stacking_model, "stacking_model.joblib")

    # End of advanced ML pipeline using Pandas and complementary modules.
