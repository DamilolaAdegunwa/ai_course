# cddml-VpZL3WqR9Yx
# File Name: real_time_multimodal_sentiment_analysis_and_trend_forecasting.py

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from datetime import datetime, timedelta  # type: ignore
from nltk.sentiment.vader import SentimentIntensityAnalyzer  # type: ignore
import nltk  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.cluster import KMeans  # type: ignore
from statsmodels.tsa.arima.model import ARIMA  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from sklearn.manifold import TSNE  # type: ignore
import optuna  # type: ignore
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
import shap  # type: ignore

nltk.download('vader_lexicon')


def load_tweet_data() -> pd.DataFrame:
    data: pd.DataFrame = pd.DataFrame({
        "timestamp": pd.date_range(start="2024-01-01", periods=100, freq="H"),
        "tweet": [
            "I love the new product! It's amazing." if i % 5 != 0 else "Terrible experience, very disappointed."
            for i in range(100)
        ]
    })
    return data


def preprocess_tweets(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def compute_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    sid: SentimentIntensityAnalyzer = SentimentIntensityAnalyzer()
    df["sentiment"] = df["tweet"].apply(lambda x: sid.polarity_scores(x)["compound"])
    return df


def aggregate_sentiment(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    df_agg: pd.DataFrame = df.set_index("timestamp").resample(freq)["sentiment"].mean().reset_index()
    return df_agg


def forecast_sentiment_arima(df: pd.DataFrame, column: str, steps: int = 5, order: tuple = (2, 1, 2)) -> pd.Series:
    ts: pd.Series = df.set_index("timestamp")[column]
    model: ARIMA = ARIMA(ts, order=order)
    model_fit = model.fit()
    forecast: pd.Series = model_fit.forecast(steps=steps)
    return forecast


def compute_tfidf(df: pd.DataFrame, text_column: str) -> Tuple[np.ndarray, list]:
    vectorizer: TfidfVectorizer = TfidfVectorizer(stop_words="english", max_features=100)
    tfidf_matrix: np.ndarray = vectorizer.fit_transform(df[text_column]).toarray()
    feature_names: list = vectorizer.get_feature_names_out().tolist()
    return tfidf_matrix, feature_names


def cluster_tweets(tfidf_matrix: np.ndarray, n_clusters: int = 3) -> np.ndarray:
    kmeans: KMeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters: np.ndarray = kmeans.fit_predict(tfidf_matrix)
    return clusters


def visualize_clusters(df: pd.DataFrame, tfidf_matrix: np.ndarray) -> None:
    pca: PCA = PCA(n_components=50, random_state=42)
    X_reduced: np.ndarray = pca.fit_transform(tfidf_matrix)
    tsne: TSNE = TSNE(n_components=2, random_state=42)
    X_tsne: np.ndarray = tsne.fit_transform(X_reduced)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=df["cluster"], palette="viridis")
    plt.title("t-SNE Visualization of Tweet Clusters")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.show()


def optuna_objective(trial: optuna.trial.Trial, X: np.ndarray, y: np.ndarray) -> float:
    p: int = trial.suggest_int("p", 1, 3)
    d: int = trial.suggest_int("d", 0, 2)
    q: int = trial.suggest_int("q", 1, 3)
    try:
        model: ARIMA = ARIMA(y, order=(p, d, q))
        model_fit = model.fit()
        score: float = -model_fit.aic
    except Exception:
        score = -1e6
    return score


def run_optuna(X: np.ndarray, y: np.ndarray) -> optuna.study.Study:
    study: optuna.study.Study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: optuna_objective(trial, X, y), n_trials=30)
    return study


def shap_explain(model: object, X_sample: pd.DataFrame) -> None:
    explainer = shap.Explainer(model.predict, X_sample)
    shap_values = explainer(X_sample)
    shap.summary_plot(shap_values, X_sample)


def lime_explain(model: object, X_train: np.ndarray, feature_names: list, X_test: np.ndarray) -> None:
    import lime.lime_tabular as lime_tabular
    explainer = lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, class_names=["neg", "pos"],
                                                  discretize_continuous=True)
    exp = explainer.explain_instance(X_test[0], model.predict_proba, num_features=10)
    exp.show_in_notebook()


def build_deep_learning_model(input_dim: int) -> tf.keras.Model:
    model: tf.keras.Model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_deep_model(model: tf.keras.Model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
                     y_val: np.ndarray) -> tf.keras.Model:
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[early_stop],
              verbose=0)
    return model


if __name__ == "__main__":
    # Load and preprocess tweet data
    tweets_df: pd.DataFrame = load_tweet_data()
    tweets_df = preprocess_tweets(tweets_df)
    tweets_df = compute_sentiment(tweets_df)

    # Aggregate sentiment daily
    sentiment_daily: pd.DataFrame = aggregate_sentiment(tweets_df, freq="D")

    # Forecast sentiment trend using ARIMA
    sentiment_forecast: pd.Series = forecast_sentiment_arima(sentiment_daily, "sentiment", steps=5, order=(2, 1, 2))
    print("Sentiment Forecast:\n", sentiment_forecast)

    # Compute TF-IDF and cluster tweets
    tfidf_matrix: np.ndarray;
    feature_names: list = compute_tfidf(tweets_df, "tweet")
    tweets_df["cluster"] = cluster_tweets(tfidf_matrix, n_clusters=3)
    visualize_clusters(tweets_df, tfidf_matrix)

    # Split dataset for further ML (simulate binary classification using sentiment)
    X: pd.DataFrame = tweets_df[["sentiment"]]
    y: pd.Series = (tweets_df["sentiment"] > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler: StandardScaler = StandardScaler()
    X_train_scaled: np.ndarray = scaler.fit_transform(X_train)
    X_test_scaled: np.ndarray = scaler.transform(X_test)

    # Hyperparameter tuning for ARIMA forecasting (using aggregated sentiment as time series)
    study: optuna.study.Study = run_optuna(sentiment_daily["sentiment"].values, sentiment_daily["sentiment"].values)
    print("Optuna Best Params for ARIMA Forecast:", study.best_params)

    # Build and train a deep learning model for sentiment classification
    X_train_dl, X_val_dl, y_train_dl, y_val_dl = train_test_split(X_train_scaled, y_train.to_numpy(), test_size=0.2,
                                                                  random_state=42, stratify=y_train)
    dl_model: tf.keras.Model = build_deep_learning_model(input_dim=X_train_scaled.shape[1])
    dl_model = train_deep_model(dl_model, X_train_dl, y_train_dl, X_val_dl, y_val_dl)
    dl_preds: np.ndarray = (dl_model.predict(X_test_scaled) > 0.5).astype(int).flatten()
    print("Deep Learning Sentiment Classification Accuracy:", accuracy_score(y_test.to_numpy(), dl_preds))

    # SHAP and LIME explainability for deep model on sentiment classification
    shap_explain(dl_model, pd.DataFrame(X_train_scaled, columns=X_train.columns))
    lime_explain(dl_model, X_train_scaled, X_train.columns.tolist(), X_test_scaled)

    # Build ensemble stacking classifier for sentiment prediction
    from sklearn.ensemble import StackingClassifier

    base_estimators: list = [
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss', random_state=42))
    ]
    meta_estimator: LogisticRegression = LogisticRegression()
    stacking_model: StackingClassifier = StackingClassifier(estimators=base_estimators, final_estimator=meta_estimator,
                                                            cv=StratifiedKFold(n_splits=5, shuffle=True,
                                                                               random_state=42))
    stacking_model.fit(X_train, y_train)
    stack_preds: np.ndarray = stacking_model.predict(X_test)
    print("Stacking Model Accuracy:", accuracy_score(y_test, stack_preds))

    # PCA and t-SNE for visualization of tweet features
    pca_model = PCA(n_components=0.95)
    X_pca: np.ndarray = pca_model.fit_transform(tfidf_matrix)
    from sklearn.manifold import TSNE

    tsne_model = TSNE(n_components=2, random_state=42)
    X_tsne: np.ndarray = tsne_model.fit_transform(X_pca)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=tweets_df["cluster"], cmap="viridis", alpha=0.7)
    plt.title("t-SNE Visualization of Tweet Clusters")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()

    # Save final deep learning and stacking models
    dump(dl_model, "deep_model_sentiment.joblib")
    dump(stacking_model, "stacking_model_sentiment.joblib")

    # End of pipeline
