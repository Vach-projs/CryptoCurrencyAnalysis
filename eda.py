import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed data
df = pd.read_csv('/content/btc_features.csv', parse_dates=['date'])

# Set plotting style
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)

# 1. Closing Price with Moving Averages
plt.figure()
plt.plot(df['date'], df['close'], label='Close', alpha=0.8)
plt.plot(df['date'], df['sma_7'], label='SMA-7')
plt.plot(df['date'], df['sma_21'], label='SMA-21')
plt.title('BTC Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# 2. Daily Return Distribution
plt.figure()
sns.histplot(df['return'], bins=100, kde=True)
plt.title('Daily Returns Distribution')
plt.xlabel('Daily Return')
plt.ylabel('Frequency')
plt.show()

# 3. Volatility (Rolling Std Dev)
plt.figure()
plt.plot(df['date'], df['stddev_21'], label='21-day Rolling Std Dev', color='orange')
plt.title('BTC Rolling Volatility')
plt.xlabel('Date')
plt.ylabel('Standard Deviation')
plt.legend()
plt.show()

# 4. Bollinger Bands
plt.figure()
plt.plot(df['date'], df['close'], label='Close', alpha=0.7)
plt.plot(df['date'], df['upper_band'], label='Upper Band', linestyle='--')
plt.plot(df['date'], df['lower_band'], label='Lower Band', linestyle='--')
plt.fill_between(df['date'], df['upper_band'], df['lower_band'], color='gray', alpha=0.2)
plt.title('Bollinger Bands')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# 5. RSI Plot
plt.figure()
plt.plot(df['date'], df['rsi_14'], label='RSI (14)', color='green')
plt.axhline(70, color='red', linestyle='--', label='Overbought')
plt.axhline(30, color='blue', linestyle='--', label='Oversold')
plt.title('RSI Over Time')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend()
plt.show()

# Plot ATR
plt.figure(figsize=(14, 6))
plt.plot(df['date'], df['atr_14'], label='ATR (14-day)', color='darkorange')
plt.title('Average True Range (ATR) - BTC')
plt.xlabel('Date')
plt.ylabel('ATR')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
