# advanced_machine_learning_project.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from joblib import dump, load
import shap
import lime
import lime.lime_tabular
from sklearn.inspection import permutation_importance
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import optuna
from optuna.integration import SklearnPruningCallback
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
import logging
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# Load dataset
def load_data() -> pd.DataFrame:
    logger.info("Loading dataset...")
    data = fetch_openml(name='adult', version=2, as_frame=True)
    df = data.frame
    return df


# Preprocess dataset
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Preprocessing data...")
    # Handle missing values
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)

    # Encode categorical features
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    return df


# Split dataset
def split_data(df: pd.DataFrame):
    logger.info("Splitting data into train and test sets...")
    X = df.drop('class', axis=1)
    y = df['class']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# Balance dataset
def balance_data(X_train: pd.DataFrame, y_train: pd.Series):
    logger.info("Balancing data using SMOTE and RandomUnderSampler...")
    over = SMOTE(sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.5)
    steps = [('o', over), ('u', under)]
    pipeline = ImbPipeline(steps=steps)
    X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


# Scale features
def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# Dimensionality reduction
def reduce_dimensions(X_train: np.ndarray, X_test: np.ndarray):
    logger.info("Reducing dimensions using PCA and t-SNE...")
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    tsne = TSNE(n_components=2, random_state=42)
    X_train_tsne = tsne.fit_transform(X_train_pca)
    X_test_tsne = tsne.fit_transform(X_test_pca)

    return X_train_tsne, X_test_tsne


# Build models
def build_models():
    logger.info("Building models...")
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'LightGBM': LGBMClassifier(random_state=42),
        'CatBoost': CatBoostClassifier(verbose=0, random_state=42)
    }
    return models


# Hyperparameter tuning using Optuna
def tune_hyperparameters(model, X_train, y_train):
    logger.info(f"Tuning hyperparameters for {model.__class__.__name__} using Optuna...")

    def objective(trial):
        if isinstance(model, RandomForestClassifier):
            n_estimators = trial.suggest_int('n_estimators', 100, 1000)
            max_depth = trial.suggest_int('max_depth', 10, 100)
            clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        elif isinstance(model, GradientBoostingClassifier):
            n_estimators = trial.suggest_int('n_estimators', 100, 1000)
            learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.3)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                             max_depth=max_depth, random_state=42)
        elif isinstance(model, SVC):
            C = trial.suggest_loguniform('C', 0.1, 10)
            kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
            clf = SVC(C=C, kernel=kernel, probability=True, random_state=42)
        elif isinstance(model, KNeighborsClassifier):
            n_neighbors = trial.suggest_int('n_neighbors', 3, 15)
            clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        elif isinstance(model, XGBClassifier):
            n_estimators = trial.suggest_int('n_estimators', 100, 1000)
            learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.3)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            clf = XGBClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                                use_label_encoder=False, eval_metric='logloss', random_state=42)
        elif isinstance(model, LGBMClassifier):
            n_estimators = trial.suggest_int('n_estimators', 100, 1000)
            learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.3)
            num_leaves = trial.suggest_int('num_leaves', 31, 150)
