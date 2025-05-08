# advanced_geospatial_crime_trend_forecasting_with_pandas.py

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import folium
from shapely.geometry import Point, Polygon
from sklearn.cluster import DBSCAN
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")


def simulate_crime_data(num_incidents: int = 1000) -> pd.DataFrame:
    np.random.seed(42)
    base_date: datetime = datetime(2023, 1, 1)
    data: dict = {
        "incident_id": np.arange(1, num_incidents + 1),
        "latitude": np.random.uniform(40.5, 40.9, num_incidents),
        "longitude": np.random.uniform(-74.0, -73.7, num_incidents),
        "date": [base_date + timedelta(days=int(x)) for x in np.random.uniform(0, 365, num_incidents)]
    }
    df: pd.DataFrame = pd.DataFrame(data)
    return df


def simulate_administrative_boundaries() -> gpd.GeoDataFrame:
    poly1: Polygon = Polygon([(-74.0, 40.5), (-73.9, 40.5), (-73.9, 40.7), (-74.0, 40.7)])
    poly2: Polygon = Polygon([(-73.9, 40.5), (-73.8, 40.5), (-73.8, 40.7), (-73.9, 40.7)])
    poly3: Polygon = Polygon([(-74.0, 40.7), (-73.9, 40.7), (-73.9, 40.9), (-74.0, 40.9)])
    poly4: Polygon = Polygon([(-73.9, 40.7), (-73.8, 40.7), (-73.8, 40.9), (-73.9, 40.9)])
    data: dict = {"area_id": [1, 2, 3, 4], "name": ["Area1", "Area2", "Area3", "Area4"],
                  "geometry": [poly1, poly2, poly3, poly4]}
    gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(data, crs="EPSG:4326")
    return gdf


def preprocess_crime_data(df: pd.DataFrame) -> gpd.GeoDataFrame:
    geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
    gdf: gpd.GeoDataFrame = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gdf


def perform_spatial_clustering(gdf: gpd.GeoDataFrame, eps: float = 0.01, min_samples: int = 5) -> gpd.GeoDataFrame:
    coords: np.ndarray = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))
    dbscan: DBSCAN = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    clusters: np.ndarray = dbscan.fit_predict(coords)
    gdf["cluster"] = clusters
    return gdf


def aggregate_crime_by_area(gdf: gpd.GeoDataFrame, boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
    join_gdf: gpd.GeoDataFrame = gpd.sjoin(gdf, boundaries, how="left", predicate="within")
    agg_df: pd.DataFrame = join_gdf.groupby(["area_id", "name", "date"]).size().reset_index(name="crime_count")
    return agg_df


def create_time_series(agg_df: pd.DataFrame, area_id: int) -> pd.DataFrame:
    area_df: pd.DataFrame = agg_df[agg_df["area_id"] == area_id]
    ts: pd.DataFrame = area_df.groupby("date")["crime_count"].sum().reset_index()
    ts["date"] = pd.to_datetime(ts["date"])
    ts = ts.set_index("date").asfreq("D").fillna(0)
    return ts


def forecast_crime_trend(ts: pd.DataFrame, steps: int = 30, order: tuple = (2, 1, 2)) -> pd.Series:
    model: ARIMA = ARIMA(ts["crime_count"], order=order)
    model_fit = model.fit()
    forecast: pd.Series = model_fit.forecast(steps=steps)
    return forecast


def plot_crime_trend(ts: pd.DataFrame, forecast: pd.Series) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(ts.index, ts["crime_count"], label="Historical Crime Count")
    forecast_index = pd.date_range(start=ts.index[-1] + timedelta(days=1), periods=len(forecast), freq="D")
    plt.plot(forecast_index, forecast, label="Forecast", linestyle="--", color="red")
    plt.xlabel("Date")
    plt.ylabel("Crime Count")
    plt.title("Crime Trend Forecast")
    plt.legend()
    plt.show()


def perform_pca_tsne(matrix: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    scaler: StandardScaler = StandardScaler()
    X_scaled: np.ndarray = scaler.fit_transform(matrix)
    from sklearn.decomposition import PCA
    pca: PCA = PCA(n_components=50, random_state=42)
    X_pca: np.ndarray = pca.fit_transform(X_scaled)
    from sklearn.manifold import TSNE
    tsne: TSNE = TSNE(n_components=2, random_state=42)
    X_tsne: np.ndarray = tsne.fit_transform(X_pca)
    return X_pca, X_tsne


def plot_tsne(X_tsne: np.ndarray, labels: np.ndarray) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap="viridis", alpha=0.7)
    plt.title("t-SNE Visualization")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()


if __name__ == "__main__":
    crime_df: pd.DataFrame = simulate_crime_data(num_incidents=2000)
    boundaries_gdf: gpd.GeoDataFrame = simulate_administrative_boundaries()
    crime_gdf: gpd.GeoDataFrame = preprocess_crime_data(crime_df)
    crime_gdf = perform_spatial_clustering(crime_gdf, eps=0.01, min_samples=5)
    agg_crime_df: pd.DataFrame = aggregate_crime_by_area(crime_gdf, boundaries_gdf)
    ts_area1: pd.DataFrame = create_time_series(agg_crime_df, area_id=1)
    crime_forecast: pd.Series = forecast_crime_trend(ts_area1, steps=30, order=(2, 1, 2))
    plot_crime_trend(ts_area1, crime_forecast)

    user_item_matrix: pd.DataFrame = create_user_item_matrix(pd.DataFrame({
        "user_id": np.random.randint(1, 101, 1000),
        "item_id": np.random.randint(1, 51, 1000),
        "rating": np.random.randint(1, 6, 1000)
    }))
    sim_matrix: np.ndarray = cosine_similarity(user_item_matrix)

    pca_data, tsne_data = perform_pca_tsne(user_item_matrix)
    plot_tsne(tsne_data, labels=np.argmax(sim_matrix, axis=1))

    # Example: Build a folium map for visualizing crime clusters
    m: folium.Map = folium.Map(location=[40.7, -73.9], zoom_start=11)
    for idx, row in crime_gdf.iterrows():
        color: str = "red" if row["cluster"] == -1 else "blue"
        folium.CircleMarker(location=[row.geometry.y, row.geometry.x], radius=2, color=color, fill=True).add_to(m)
    m.save("crime_map.html")

    # End of advanced pipeline for geospatial crime trend forecasting and visualization.
