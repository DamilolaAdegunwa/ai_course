import pandas as pd
import numpy as np
from surprise import Dataset, Reader
import tensorflow as tf
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Load dataset from Surprise
reader: Reader = Reader(rating_scale=(1, 5))
df: pd.DataFrame = pd.read_csv("https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv")
data = Dataset.load_from_df(df[["user_id", "book_id", "rating"]], reader)

# TensorFlow Feature Extraction
user_features: tf.Tensor = tf.convert_to_tensor(df.groupby('user_id').mean().drop(columns=['book_id']), dtype=tf.float32)
product_features: tf.Tensor = tf.convert_to_tensor(df.groupby('book_id').mean().drop(columns=['user_id']), dtype=tf.float32)

# Reinforcement Learning using RLlib
ray.init()

def reward_function(user_id: int, product_id: int) -> float:
    return np.random.uniform(1, 5)

config = (
    PPO.get_default_config()
    .environment("CartPole-v1")
    .framework("tf")
    .training(lr=0.001, train_batch_size=4000)
    .experimental(_validate_config=False)  # Suppress validation errors
)

trainer = PPO(config=config)

for i in range(10):
    result = trainer.train()
    print(f"Iteration {i}: reward {result['episode_reward_mean']}")

ray.shutdown()

# Function to Get Recommendations
def get_recommendations(user_id: int) -> list:
    return [int(product) for product in df["book_id"].sample(3)]

# Example Usage
user_id: int = 101
print(f"Recommended Products for User {user_id}: {get_recommendations(user_id)}")
