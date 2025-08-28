###########################################################################################################
#                                          IN THE NAME OF ALLAH                                           #
#      UTILIZING MARKOV-CHAIN & MONTE-CARLO FOR STOCK PRICE FORECASTING ACROSS MULTIPLE SCENARIOS         #
#---------------------------------------------------------------------------------------------------------#
#                          THIS PROGRAM WRITTEN BY SALAR DELAVAR GHASHGHAEI (QASHQAI)                     #
#                                   EMAIL: salar.d.ghashghaei@gmail.com                                   #
########################################################################################################### 
"""
1. The script imports essential libraries: 'NumPy, Pandas, Matplotlib, and Yahoo Finance (yfinance)'.
2. It downloads 'historical stock prices' for Apple (AAPL) from 2020 to 2025.
3. It computes 'log returns' from the daily closing prices.
4. A 'log-likelihood function' is defined under the assumption that returns follow a Normal distribution.
5. The 'Metropolis–Hastings MCMC algorithm' is implemented to estimate drift (μ) and volatility (σ).
6. Starting from initial guesses, the algorithm iteratively samples candidate parameters and accepts/rejects them based on likelihood ratios.
7. The MCMC chain is averaged to produce final estimates of μ and σ.
8. Using these estimated parameters, the stock price is simulated via 'Geometric Brownian Motion (GBM)'.
9. The model generates '50 simulated price paths' over a one-year horizon (252 trading days).
10. Finally, the results are visualized in a 'log-scaled plot' showing multiple future stock price scenarios.
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
#%%-------------------------------------------------------------------------------
# 1. Download historical stock data
ticker = "AAPL"  # Example: Apple stock
data = yf.download(ticker, start="2020-01-01", end="2025-01-01")
prices = data["Close"]
#%%-------------------------------------------------------------------------------
# 2. Compute log-returns
returns = np.log(prices / prices.shift(1)).dropna()
#%%-------------------------------------------------------------------------------
# 3. Define log-likelihood for Normal distribution of returns
def log_likelihood(params, data):
    mu, sigma = params
    if sigma <= 0:  # sigma must be positive
        return -np.inf
    n = len(data)
    return -0.5 * n * np.log(2 * np.pi * sigma**2) - ((data - mu) ** 2).sum() / (2 * sigma**2)
#%%-------------------------------------------------------------------------------
# 4. Metropolis–Hastings MCMC to estimate μ and σ
def metropolis_hastings(data, initial_params, iterations, proposal_width=0.01):
    params = np.zeros((iterations, 2))
    params[0] = initial_params
    current_ll = log_likelihood(initial_params, data)

    for i in range(1, iterations):
        # Propose new parameters
        proposal = params[i-1] + np.random.normal(0, proposal_width, 2)
        proposal_ll = log_likelihood(proposal, data)

        # Acceptance probability
        alpha = np.exp(proposal_ll - current_ll)
        if np.random.rand() < alpha:
            params[i] = proposal
            current_ll = proposal_ll
        else:
            params[i] = params[i-1]

    return params
#%%-------------------------------------------------------------------------------
# 5. Run MCMC
params = metropolis_hastings(returns.values, [0, 0.02], 5000)
mu_est, sigma_est = params.mean(axis=0)
print(f"Estimated μ (drift): {mu_est:.5f}, Estimated σ (volatility): {sigma_est:.5f}")
#%%-------------------------------------------------------------------------------
# 6. Simulate future stock price scenarios with GBM
S0 = prices.iloc[-1]  # Current price
T = 252      # Trading days in one year
Nsim = 50    # Number of scenarios

simulations = []
for _ in range(Nsim):
    dt = 1/T
    prices_sim = [S0]
    for _ in range(T):
        dS = mu_est * prices_sim[-1] * dt + sigma_est * prices_sim[-1] * np.sqrt(dt) * np.random.randn()
        prices_sim.append(prices_sim[-1] + dS)
    simulations.append(prices_sim)
#%%-------------------------------------------------------------------------------
# 7. Plot results
plt.figure(figsize=(10,6))
for sim in simulations:
    plt.plot(sim, lw=1)
plt.title(f"Stock Price Simulation for {ticker} using MCMC-estimated GBM")
plt.xlabel("Days")
plt.ylabel("Price")
plt.semilogy()
plt.show()
#%%-------------------------------------------------------------------------------