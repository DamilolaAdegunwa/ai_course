import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize the Kaggle API
api: KaggleApi = KaggleApi()
api.authenticate()

# Download the Titanic dataset
api.dataset_download_files('heptapod/titanic', path='./data', unzip=True)

# Load the dataset into a pandas DataFrame
df: pd.DataFrame = pd.read_csv(r'C:\Users\damil\PycharmProjects\ai_course\ml_projects\data\train_and_test2.csv')

# Print the dataset
print(df)
comments2 = """
from kaggle.api.kaggle_api_extended import KaggleApi

api: KaggleApi = KaggleApi()
api.authenticate()
print("Kaggle API authentication successful!")
"""


comments = """

import numpy as np
# X = np.random.normal(0, 1, (100, 2))  # Normal data
import pandas as pd

# Sample DataFrame
data = pd.DataFrame({
    'duration': ['10', '20.5', 'abc', '30'],
    'src_bytes': ['100', 'xyz', '200', '300'],
    'dst_bytes': ['500', '600', '700', '800'],
    'labels': ['1', '2', 'three', '4']
})

# Columns to check
cols_to_check = ['duration', 'src_bytes', 'dst_bytes', 'labels']


# Function to check if a value is convertible to float
def is_float_convertible(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


# Remove rows where any column in cols_to_check contains a non-convertible value
filtered_data = data[data[cols_to_check].applymap(is_float_convertible).all(axis=1)]

# Display the cleaned DataFrame
print(filtered_data)
"""
# --------------
