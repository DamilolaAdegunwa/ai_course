# Import necessary libraries
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load dataset
# For this example, we'll use a hypothetical dataset
# Replace this with your actual data loading code
data: pd.DataFrame = pd.read_csv('time_series_data.csv')

# Display the first few rows of the dataset
print(data.head())

# Data preprocessing
# Convert the date column to datetime format
data['ds'] = pd.to_datetime(data['ds'])

# Handle missing values if any
data = data.fillna(method='ffill')

# Feature engineering
# Adding external regressors if available
# For example, adding a 'holiday' indicator
# Ensure 'holiday' is a binary indicator (0 or 1)
if 'holiday' in data.columns:
    data['holiday'] = data['holiday'].apply(lambda x: 1 if x else 0)

# Initialize the Prophet model
model = Prophet()

# Add external regressors to the model if they exist
if 'holiday' in data.columns:
    model.add_regressor('holiday')

# Fit the model to the data
model.fit(data)

# Create a dataframe for future dates
future: pd.DataFrame = model.make_future_dataframe(periods=365)

# Include external regressors in the future dataframe if they exist
if 'holiday' in data.columns:
    # Assuming future holidays are known and stored in a dataframe 'future_holidays'
    # with columns 'ds' and 'holiday'
    future = future.merge(future_holidays, on='ds', how='left')
    future['holiday'] = future['holiday'].fillna(0)

# Generate forecasts
forecast: pd.DataFrame = model.predict(future)

# Plot the forecast
fig1 = model.plot(forecast)
plt.title('Time Series Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# Plot the forecast components
fig2 = model.plot_components(forecast)
plt.show()

# Save the forecast to a CSV file
forecast.to_csv('forecasted_values.csv', index=False)
