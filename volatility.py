

import pandas as pd

df = pd.read_csv('/content/btc_featuresn.csv', parse_dates=['date'])
df.set_index('date', inplace=True)

print(df.columns)
df[['return', 'atr_14']].dropna().head()
print(df.tail())

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))

# Plot ATR (Absolute volatility)
plt.plot(df.index, df['atr_14'], label='ATR (14)', color='orange')

# Calculate 7-Day Rolling StdDev and scale it for visibility
rolling_std_7 = df['return'].rolling(window=7).std()
scaled_rolling_std_7 = rolling_std_7 * 10000  # Scale to match ATR range

# Plot Scaled Daily Return StdDev
plt.plot(df.index, scaled_rolling_std_7, label='7-Day Rolling StdDev of Returns (x10,000)', color='purple')

plt.title('Volatility Indicators: ATR & Scaled Rolling Return StdDev')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

!pip install arch --quiet

from arch import arch_model
import pandas as pd
import matplotlib.pyplot as plt

# Load your processed dataset (if not already)
df = pd.read_csv('/content/btc_featuresn.csv', parse_dates=['date'], index_col='date')

# Drop NaNs and scale returns
returns = df['return'].dropna() * 100  # Scaling for GARCH stability

# GARCH(1,1) with normal distribution
model = arch_model(returns, vol='GARCH', p=1, q=1, dist='normal')
res = model.fit(disp='off')

# Print summary
print(res.summary())

plt.figure(figsize=(14, 6))
plt.plot(res.conditional_volatility, color='darkred', label='GARCH(1,1) Volatility')
plt.title('GARCH(1,1) Conditional Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

from arch import arch_model
import matplotlib.pyplot as plt

# Drop NaNs from your return series just to be safe
returns = df['return'].dropna() * 100  # Scaling returns for better model fit

# Fit EGARCH(1,1)
egarch_model = arch_model(returns, vol='EGARCH', p=1, o=1, q=1, dist='normal')
egarch_result = egarch_model.fit(disp='off')

# Get conditional volatility
egarch_vol = egarch_result.conditional_volatility

# Plot EGARCH conditional volatility
plt.figure(figsize=(14, 6))
plt.plot(egarch_vol.index, egarch_vol, color='darkgreen', label='EGARCH(1,1) Volatility')
plt.title('EGARCH(1,1) Conditional Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import pandas as pd
from arch import arch_model

# Load your existing features
features_df = pd.read_csv('/content/btc_featuresn.csv', parse_dates=['date'])
features_df.set_index('date', inplace=True)

# Prepare returns for modeling (in percentage form)
returns = features_df['return'].dropna() * 100

# Basic GARCH(1,1)
garch_model = arch_model(returns, vol='GARCH', p=1, q=1)
garch_res = garch_model.fit(disp='off')
features_df['garch_vol'] = garch_res.conditional_volatility

# EGARCH(1,1)
egarch_model = arch_model(returns, vol='EGARCH', p=1, q=1)
egarch_res = egarch_model.fit(disp='off')
features_df['egarch_vol'] = egarch_res.conditional_volatility

# Save the final feature set
features_df.reset_index(inplace=True)
features_df.to_csv('/content/btc_vol.csv', index=False)
print("Both GARCH and EGARCH volatility scores saved.")

df=pd.read_csv('/content/btc_vol.csv')
df.info()
