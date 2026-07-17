import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Generate price data with outliers ---
#n = 400
#price = 100 + np.cumsum(np.random.randn(n) * 1.0)
def Generate_Prices(start_price, Volatility, Direction_Bias, num_days):
    import random
    prices = [start_price]
    changes = [0]
    for i in range(num_days-1):
        Var = random.uniform(-1, 1)*random.uniform(0, 1)*Volatility
        price = prices[-1] *(1 + Var + Direction_Bias)
        prices.append(price)
        changes.append(prices[-1] / prices[i-1] - 1)
    return prices


start_price, Volatility, Direction_Bias, num_days = 100, 0.05, -0.0001, 1000
price = Generate_Prices(start_price, Volatility, Direction_Bias, num_days)

window = 20      # typical Bollinger window
k = 2            # bandwidth multiplier

# --- Classic Bollinger Bands (mean & standard deviation) ---
rolling_mean = pd.Series(price).rolling(window=window).mean()
rolling_std = pd.Series(price).rolling(window=window).std(ddof=0)  # population std
upper_classic = rolling_mean + k * rolling_std
lower_classic = rolling_mean - k * rolling_std

# --- Robust Bollinger Bands (median & IQR) ---
rolling_median = pd.Series(price).rolling(window=window).median()

# Calculate rolling IQR
def rolling_iqr(series, win):
    return series.rolling(window=win).apply(
        lambda x: np.percentile(x, 75) - np.percentile(x, 25), raw=True
    )

iqr = rolling_iqr(pd.Series(price), window)
# Scale IQR to approximate standard deviation (for normal distribution)
robust_width = k * (iqr / 1.35)
upper_robust = rolling_median + robust_width
lower_robust = rolling_median - robust_width

# --- Plot ---
plt.figure(figsize=(14, 7))
plt.plot(price, label='Price', color='black', linewidth=1)

# Classic bands
plt.plot(rolling_mean, label='SMA (Mean)', color='blue', linestyle='--')
plt.plot(upper_classic, label='Upper Band (Classic)', color='cyan', alpha=0.6)
plt.plot(lower_classic, label='Lower Band (Classic)', color='cyan', alpha=0.6)
plt.fill_between(range(len(price)), lower_classic, upper_classic, color='cyan', alpha=0.1)

# Robust bands
plt.plot(rolling_median, label='Rolling Median (Robust)', color='red', linestyle='--')
plt.plot(upper_robust, label='Upper Band (Robust)', color='orange', alpha=0.8)
plt.plot(lower_robust, label='Lower Band (Robust)', color='orange', alpha=0.8)
plt.fill_between(range(len(price)), lower_robust, upper_robust, color='orange', alpha=0.05)

plt.title('Comparison of Classic Bollinger Bands (Mean & Std) and Robust (Median & IQR)')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()