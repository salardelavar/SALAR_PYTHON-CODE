import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate random price data with some outlier spikes
#n = 1000
#price = 100 + np.cumsum(np.random.randn(n) * 1.0)   # random trend

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



def traditional_rsi(prices, window):
    delta = np.diff(prices, prepend=prices[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.full_like(gain, np.nan, dtype=float)
    avg_loss = np.full_like(loss, np.nan, dtype=float)
    # initial simple average
    avg_gain[window] = np.mean(gain[1:window+1])
    avg_loss[window] = np.mean(loss[1:window+1])
    # Wilder smoothing (EMA with alpha = 1/window)
    alpha = 1.0 / window
    for i in range(window+1, len(prices)):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def robust_rsi_median(prices, window):
    delta = np.diff(prices, prepend=prices[0])
    rsi_rob = np.full(len(prices), np.nan)
    for i in range(window, len(prices)):
        # extract recent window
        window_delta = delta[i-window+1 : i+1]
        gains = window_delta[window_delta > 0]
        losses = -window_delta[window_delta < 0]
        median_gain = np.median(gains) if len(gains) > 0 else 0.0
        median_loss = np.median(losses) if len(losses) > 0 else 0.0
        if median_loss == 0:
            rs = 100.0 if median_gain > 0 else 1.0   # if no loss
        else:
            rs = median_gain / median_loss
        rsi_rob[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi_rob

# Calculation
start_price, Volatility, Direction_Bias, num_days = 100, 0.05, -0.0001, 1000
price = Generate_Prices(start_price, Volatility, Direction_Bias, num_days)

window = 50
rsi_trad = traditional_rsi(price, window)
rsi_med = robust_rsi_median(price, window)

# Display
plt.figure(figsize=(14, 8))
plt.subplot(2,1,1)
plt.plot(price, label='Price', color='black')
plt.title('Price with Outlier Fluctuations')
plt.semilogy()
plt.legend()

plt.subplot(2,1,2)
plt.plot(rsi_trad, label='Classic RSI (Mean)', color='blue')
plt.plot(rsi_med, label='Robust RSI (Median)', color='red', linestyle='--')
MINt = np.quantile(rsi_trad[window:], 0.05)
MAXt = np.quantile(rsi_trad[window:], 0.95)
MINr = np.quantile(rsi_med[window:], 0.05)
MAXr = np.quantile(rsi_med[window:], 0.95)
print(f"5th percentile threshold: {MINt:.2f}, 95th percentile threshold: {MAXt:.2f}")
print(f"5th percentile threshold: {MINr:.2f}, 95th percentile threshold: {MAXr:.2f}")
plt.axhline(MAXt, color='blue', linestyle=':', alpha=0.7)
plt.axhline(MINt, color='blue', linestyle=':', alpha=0.7)
plt.axhline(MAXr, color='red', linestyle=':', alpha=0.7)
plt.axhline(MINr, color='red', linestyle=':', alpha=0.7)
plt.title('Comparison of Traditional and Robust RSI')
plt.legend()
plt.tight_layout()
plt.show()