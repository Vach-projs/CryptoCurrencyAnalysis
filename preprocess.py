import pandas as pd
import numpy as np

# 📥 Load the historical data from Yahoo
df = pd.read_csv('/content/btc_yfinancen.csv', parse_dates=['date'])

# ✅ 1. Sort by date and reset index
df.sort_values('date', inplace=True)
df.reset_index(drop=True, inplace=True)

# ✅ 2. Fill missing values in raw data only (open, high, low, close, volume)
df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].fillna(method='ffill')

# ✅ 3. Feature Engineering

# Daily returns
df['return'] = df['close'].pct_change()
df['log_return'] = np.log(df['close'] / df['close'].shift(1))

# Moving averages
df['sma_7'] = df['close'].rolling(window=7).mean()
df['sma_21'] = df['close'].rolling(window=21).mean()

# Bollinger Bands
df['stddev_21'] = df['close'].rolling(window=21).std()
df['upper_band'] = df['sma_21'] + (2 * df['stddev_21'])
df['lower_band'] = df['sma_21'] - (2 * df['stddev_21'])

# RSI
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['rsi_14'] = compute_rsi(df['close'], 14)

# ATR
df['hl'] = df['high'] - df['low']
df['hc'] = np.abs(df['high'] - df['close'].shift())
df['lc'] = np.abs(df['low'] - df['close'].shift())
df['tr'] = df[['hl', 'hc', 'lc']].max(axis=1)
df['atr_14'] = df['tr'].rolling(window=14).mean()

# Final cleanup
df.drop(['hl', 'hc', 'lc', 'tr'], axis=1, inplace=True)

# ✅ Don't drop any rows — keep early rows even if some feature columns have NaNs
# You can later fill or interpolate selectively if needed

# 💾 Save to file
df.to_csv('/content/btc_featuresn.csv', index=False)
print("✅ Feature engineered data saved with full date range from Jan 1st.")
