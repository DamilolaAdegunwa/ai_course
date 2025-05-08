# advanced_financial_time_series_analysis.py

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
# import talib


# Fetch historical stock data
def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data


# Plot stock closing price
def plot_closing_price(stock_data: pd.DataFrame, ticker: str) -> None:
    plt.figure(figsize=(14, 7))
    plt.plot(stock_data['Close'], label=f'{ticker} Closing Price')
    plt.title(f'{ticker} Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
