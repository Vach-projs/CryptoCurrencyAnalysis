

!pip install prophet

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/content/btc_sentimentn.csv', parse_dates=['date'])

# Fill sentiment NAs (bfill just in case)
df['sentiment_score'] = df['sentiment_score'].ffill().bfill()

# Rename for Prophet
df.rename(columns={'date': 'ds', 'close': 'y'}, inplace=True)

# Select needed columns
cols = ['ds', 'y', 'sentiment_score', 'volume', 'log_return',
        'sma_7', 'sma_21', 'upper_band', 'lower_band', 'rsi_14', 'atr_14']
df = df[cols].dropna()

# Initialize Prophet with regressors
model = Prophet(daily_seasonality=True)

# Add regressors
regressors = cols[2:]  # Everything except ds, y
for reg in regressors:
    model.add_regressor(reg)

# Fit model
model.fit(df)

# Make future dataframe
future = model.make_future_dataframe(periods=30)

# Extend future DataFrame with regressor values
# Use the last known value to fill in future steps
for reg in regressors:
    future[reg] = list(df[reg]) + [df[reg].iloc[-1]] * 30

# Forecast
forecast = model.predict(future)

# Plot forecast
fig1 = model.plot(forecast)
plt.title("Prophet Long-Term Forecast (30 Days)")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.grid(True)
plt.show()

# plot components (trend, seasonality)
fig2 = model.plot_components(forecast)
plt.show()

forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('prophet_forecast.csv', index=False)
