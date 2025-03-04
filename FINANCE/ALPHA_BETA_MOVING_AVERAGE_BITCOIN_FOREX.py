### BETA MOVING AVERAGE 
# Import libraries
#import numpy as np
#import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Define the list of symbols to get data for
#symbols = ["META", "GOOG", "NFLX", "AMZN", "WMT", "^GSPC", "GC=F"]  # GC=F = Gold futures

#symbols = ["BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "GC=F"]  # GC=F = Gold futures
symbols = ["GC=F", "BTC-USD"]  # GC=F = Gold futures

data = yf.download(symbols, start="2018-01-01", end="2025-02-10")

# Calculate the price change for each symbol
price_change = data["Close"].pct_change()


# Define the y data as the Gold futures price change
y_data = price_change["GC=F"]

# Define the x data as the other symbols price change
x_data = price_change.drop(columns=["GC=F"])


period = 50
fig, ax = plt.subplots(figsize=(12, 8))
for symbol in x_data.columns:
    beta_symbol = x_data[symbol].rolling(period).cov(y_data) / y_data.rolling(period).var()
    ax.axhline(beta_symbol.quantile(0.05), color="green", linestyle="--", label=f"Q 5: {beta_symbol.quantile(0.05):.4f}")
    ax.axhline(beta_symbol.quantile(0.5), color="orange", linestyle="--", label=f"Q 50: {beta_symbol.quantile(0.5):.4f}")
    ax.axhline(beta_symbol.quantile(0.95), color="red", linestyle="--", label=f"Q 95: {beta_symbol.quantile(0.95):.4f}")
    # Plot the beta moving average for each symbol
    ax.plot(beta_symbol, label=f"{symbol}: {beta_symbol[-1]:.3f}")
ax.set_title("Beta Moving Average for Symbols")
ax.set_xlabel("Date")
ax.set_ylabel("Beta")
ax.grid()
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(12, 8))
for symbol in x_data.columns:
    beta_symbol = x_data[symbol].rolling(period).cov(y_data) / y_data.rolling(period).var()
    alpha_symbol = y_data.rolling(period).mean() - beta_symbol * x_data[symbol].rolling(period).mean()
    Y = alpha_symbol + alpha_symbol * y_data
    MOV = Y.rolling(period).mean()
    MOV_STD_BOT = Y.rolling(period).mean() - 2 * Y.rolling(period).std()
    MOV_STD_TOP = Y.rolling(period).mean() + 2 * Y.rolling(period).std()
    ax.axhline(Y.quantile(0.05), color="green", linestyle="--", label=f"Q 5: {Y.quantile(0.05):.4f}")
    ax.axhline(Y.quantile(0.5), color="orange", linestyle="--", label=f"Q 50: {Y.quantile(0.5):.4f}")
    ax.axhline(Y.quantile(0.95), color="red", linestyle="--", label=f"Q 95: {Y.quantile(0.95):.4f}")
    print(f'{symbol}\t\t alpha + beta * X : {alpha_symbol[-1]:.4e} + {beta_symbol[-1]:.4e} * X')
    # Plot the beta moving average for each symbol
    ax.plot(Y, label=f"{symbol}:  {alpha_symbol[-1]:.4e} + {beta_symbol[-1]:.4e} * X = {Y[-1]:.4f}")
    ax.plot(MOV, color='red', label=f"MOV: {MOV[-1]:.4f}")
    ax.plot(MOV_STD_BOT, color='purple', label=f"MOV-2*STD: {MOV_STD_BOT[-1]:.4f}")
    ax.plot(MOV_STD_TOP, color='purple', label=f"MOV+2*STD: {MOV_STD_TOP[-1]:.4f}")
ax.set_title("(alpha + beta * X) Moving Average for Symbols")
ax.set_xlabel("Date")
ax.set_ylabel("alpha + beta * X")
ax.grid()
ax.legend()
plt.show()

    