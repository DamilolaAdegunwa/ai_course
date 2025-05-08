from surprise import Dataset, Reader
import pandas as pd

# Load MovieLens dataset (User ID, Movie ID, Rating)
data_url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
data = pd.read_csv(data_url, sep='\t', names=["user", "item", "stars", "timestamp"])

# Drop unnecessary column
data = data.drop(columns=["timestamp"])

# Ensure proper formatting for Surprise
reader = Reader(rating_scale=(1, 5))
data_from_df = Dataset.load_from_df(data[['user', 'item', 'stars']], reader)

# Print sample data
print(data.head())
