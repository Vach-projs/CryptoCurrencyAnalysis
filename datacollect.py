# Install dependencies (if not already installed)
!pip install yfinance --quiet

import yfinance as yf
import requests
import pandas as pd
from datetime import datetime
import os

# Create directory to store data
os.makedirs('/content/crypto_forecasting/data/raw', exist_ok=True)

# 1. Fetch Historical Data from Yahoo Finance
def fetch_yfinance_data(ticker='BTC-USD', start='2020-01-01', end=None, interval='1d'):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')

    data = yf.download(ticker, start=start, end=end, interval=interval)

    data = data.copy()
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    data.reset_index(inplace=True)
    data.columns = [str(col).lower().replace(' ', '_') for col in data.columns]

    return data

btc_df = fetch_yfinance_data(ticker='BTC-USD', start='2025-01-01')
btc_df.to_csv('/content/btc_yfinancen.csv', index=False)
print("BTC historical data from Yahoo Finance saved.")

# 2. Fetch Real-Time Market Price from CoinGecko
def get_realtime_price(coin_id='bitcoin', vs_currency='usd'):
    url = 'https://api.coingecko.com/api/v3/simple/price'
    params = {
        'ids': coin_id,
        'vs_currencies': vs_currency,
        'include_last_updated_at': 'true'
    }
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, params=params, headers=headers)

    if response.status_code != 200:
        raise Exception(f"Error fetching real-time data: {response.status_code}")

    data = response.json()
    price = data[coin_id][vs_currency]
    timestamp = datetime.fromtimestamp(data[coin_id]['last_updated_at'])

    realtime_df = pd.DataFrame([{
        'timestamp': timestamp,
        'price': price
    }])
    return realtime_df

btc_realtime_df = get_realtime_price('bitcoin')
btc_realtime_df.to_csv('/content/btc_realtime_pricen.csv', index=False)
print("Real-time BTC price from CoinGecko saved.")
