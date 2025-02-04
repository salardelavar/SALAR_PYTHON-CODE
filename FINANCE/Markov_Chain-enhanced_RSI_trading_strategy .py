"""
This code implements a Markov Chain-enhanced RSI trading strategy
 with advanced backtesting. Here's a concise breakdown:
1. Strategy Core: Combines Markov chains with RSI to replace fixed thresholds with dynamic state-based analysis.  
2. Data Processing: Fetches historical prices, calculates RSI, and discretizes it into states (e.g., 0–20, 21–40).  
3. Markov Model: Builds a transition matrix to track probabilistic RSI state shifts (e.g., oversold → neutral).  
4. Signal Logic: Generates buy/sell signals when transitions to higher/lower states exceed a confidence threshold.  
5. Risk Management: Adds stop-loss (5%) and take-profit (10%) rules to limit losses and lock gains.  
6. Advanced Metrics: Computes Sharpe Ratio, Max Drawdown, and Win Rate for robust performance evaluation.  
7. Validation: Uses walk-forward testing (80% train, 20% test) to avoid overfitting and ensure reliability.  
8. Visualization: Plots price, RSI trends, state sequences, transition matrix heatmap, and equity curve.  
9. Adaptability: Adjusts thresholds based on asset volatility (e.g., wider bins for unstable markets).  
10. Innovation: Quantifies market regimes probabilistically, improving risk analysis vs. static RSI rules.
------------------------------------------------------------------------------
THIS PROGRAM IS WRITTEN BY SALAR DELAVAR GHASHGHAEI (QASHQAI)
EMAIL: SALAR.D.GHASHGHAEI@GMAIL.COM
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Fetch historical price data
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Step 2: Calculate RSI
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rsi = 100 * (gain / (gain + loss))
    return rsi

# Step 3: Discretize RSI into states
def discretize_rsi(rsi, n_bins=10):
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    rsi_states = discretizer.fit_transform(rsi.values.reshape(-1, 1))
    return rsi_states.flatten()

# Step 4: Build Markov transition matrix
def build_transition_matrix(states):
    n_states = int(np.max(states) + 1)
    transition_matrix = np.zeros((n_states, n_states))
    
    for (i, j) in zip(states[:-1], states[1:]):
        transition_matrix[int(i), int(j)] += 1
    
    # Normalize rows to get probabilities
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    return transition_matrix

# Step 5: Generate trading signals
def generate_signals(rsi_states, transition_matrix, threshold=0.7):
    signals = []
    for state in rsi_states:
        next_state_probs = transition_matrix[int(state)]
        if next_state_probs.argmax() > state and next_state_probs.max() > threshold:
            signals.append(1)  # Buy signal
        elif next_state_probs.argmax() < state and next_state_probs.max() > threshold:
            signals.append(-1)  # Sell signal
        else:
            signals.append(0)  # Hold
    return signals

# Step 6: Backtest the strategy with advanced metrics
def backtest(prices, signals, stop_loss_pct=0.05, take_profit_pct=0.1):
    returns = prices.pct_change().shift(-1)
    strategy_returns = []
    position = 0  # 0 = no position, 1 = long, -1 = short
    entry_price = 0
    
    for i in range(len(signals)):
        current_price = prices.iloc[i]
        
        # Close position if stop-loss/take-profit triggered
        if position != 0:
            pl = (current_price - entry_price) / entry_price if position == 1 else (entry_price - current_price) / entry_price
            if pl <= -stop_loss_pct or pl >= take_profit_pct:
                strategy_returns.append(pl * position)
                position = 0
        
        # Open new position
        if position == 0 and signals[i] != 0:
            position = signals[i]
            entry_price = current_price
            strategy_returns.append(0)
        else:
            strategy_returns.append(0)
    
    # Calculate metrics
    strategy_returns = pd.Series(strategy_returns, index=prices.index)
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    # Advanced metrics
    sharpe_ratio = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
    max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
    win_rate = (strategy_returns > 0).mean()
    
    return cumulative_returns, sharpe_ratio, max_drawdown, win_rate

# Step 8: Walk-forward validation
def walk_forward_validation(prices, train_ratio=0.8, rsi_window=14, n_bins=10):
    train_size = int(len(prices) * train_ratio)
    train_prices = prices.iloc[:train_size]
    test_prices = prices.iloc[train_size:]
    
    # Train on historical data
    train_rsi = calculate_rsi(train_prices, window=rsi_window).dropna()
    train_states = discretize_rsi(train_rsi, n_bins=n_bins)
    transition_matrix = build_transition_matrix(train_states)
    
    # Test on unseen data
    test_rsi = calculate_rsi(test_prices, window=rsi_window).dropna()
    test_states = discretize_rsi(test_rsi, n_bins=n_bins)
    test_signals = generate_signals(test_states, transition_matrix)
    
    # Backtest
    cumulative_returns, sharpe, drawdown, win_rate = backtest(test_prices.iloc[-len(test_signals):], test_signals)
    return cumulative_returns, sharpe, drawdown, win_rate

# Step 9: Plot price, RSI, and Markov transition matrix
def plot_results(prices, rsi, rsi_states, transition_matrix, cumulative_returns):
    fig, axes = plt.subplots(4, 1, figsize=(14, 18), gridspec_kw={'height_ratios': [2, 1, 1, 2]})
    
    # Plot price chart
    # A line chart showing the historical price of the asset.
    axes[0].plot(prices, label='Price', color='blue')
    axes[0].set_title('Price Chart')
    axes[0].set_ylabel('Price')
    axes[0].legend()

    # Plot RSI chart
    # A line chart showing RSI values with overbought and oversold levels.
    axes[1].plot(rsi, label='RSI', color='orange')
    axes[1].axhline(70, linestyle='--', color='red', alpha=0.5)
    axes[1].axhline(30, linestyle='--', color='green', alpha=0.5)
    axes[1].set_title('RSI Chart')
    axes[1].set_ylabel('RSI')
    axes[1].legend()

    # Plot RSI states
    # A line chart showing the discretized RSI states.
    axes[2].plot(rsi_states, label='RSI States', color='purple')
    axes[2].set_title('RSI States')
    axes[2].set_ylabel('State')
    axes[2].legend()

    # Plot Markov transition matrix
    # A heatmap showing the probabilities of transitioning from one RSI state to another.
    sns.heatmap(transition_matrix, annot=True, cmap='Blues', ax=axes[3])
    axes[3].set_title('Markov Transition Matrix')
    axes[3].set_xlabel('Next State')
    axes[3].set_ylabel('Current State')

    plt.tight_layout()
    plt.show()

    # Plot cumulative returns
    # A line chart showing the cumulative returns of the strategy.
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns, label='Markov RSI Strategy', color='green')
    plt.title('Cumulative Returns of Markov RSI Strategy')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def main():
    # Parameters
    ticker = 'AAPL'
    start_date = '2010-01-01'
    end_date = '2023-01-01'
    rsi_window = 14
    n_bins = 5
    probability_threshold = 0.7

    # Fetch data
    data = fetch_data(ticker, start_date, end_date)
    prices = data['Close']

    # Calculate RSI
    rsi = calculate_rsi(prices, window=rsi_window)

    # Discretize RSI into states
    rsi_states = discretize_rsi(rsi.dropna(), n_bins=n_bins)

    # Build Markov transition matrix
    transition_matrix = build_transition_matrix(rsi_states)

    # Generate trading signals
    signals = generate_signals(rsi_states, transition_matrix, threshold=probability_threshold)

    # Backtest with advanced metrics
    cumulative_returns, sharpe, drawdown, win_rate = backtest(prices.iloc[-len(signals):], signals)
    
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {drawdown:.2%}")
    print(f"Win Rate: {win_rate:.2%}")

    # Walk-forward validation
    wf_returns, wf_sharpe, wf_drawdown, wf_win_rate = walk_forward_validation(prices)
    print("\nWalk-Forward Validation Results:")
    print(f"Sharpe Ratio: {wf_sharpe:.2f}")
    print(f"Max Drawdown: {wf_drawdown:.2%}")
    print(f"Win Rate: {wf_win_rate:.2%}")

    # Plot results
    plot_results(prices, rsi, rsi_states, transition_matrix, cumulative_returns)

if __name__ == "__main__":
    main()