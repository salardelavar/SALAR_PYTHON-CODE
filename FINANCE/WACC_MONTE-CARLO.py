import numpy as np
import matplotlib.pyplot as plt

# Number of Monte Carlo samples
n_samples = 20_000

# Capital structure
debt_amount = 5_000_000_000
equity_amount = 5_000_000_000
total = equity_amount + debt_amount

# Fixed tax rate
tax_rate = 0.15

# ------------------------------
# Define Beta-distributed uncertain input variables
# Adjust the (a, b) shape parameters to control distribution shape

# Interest rate (loan rate) between 20% and 40%
interest_rate_MIN = 0.2
interest_rate_MAX = 0.4
interest_rate = np.random.beta(a=2, b=5, size=n_samples) * (interest_rate_MAX - interest_rate_MIN) + interest_rate_MIN

# Risk-free rate between 15% and 35%
risk_free_rate_MIN = 0.15
risk_free_rate_MAX = 0.35
risk_free_rate = np.random.beta(a=2, b=5, size=n_samples) * (risk_free_rate_MAX - risk_free_rate_MIN) + risk_free_rate_MIN

# Risk premium between 15% and 35%  -> FUTURE INFLATION
risk_premium_MIN = 0.35
risk_premium_MAX = 0.50
risk_premium = np.random.beta(a=2, b=5, size=n_samples) * (risk_premium_MAX - risk_premium_MIN) + risk_premium_MIN

# ------------------------------
# Calculate cost of debt (after tax) and cost of equity

cost_of_debt = interest_rate * (1 - tax_rate)
cost_of_equity = risk_free_rate + risk_premium

# ------------------------------
# Calculate WACC for each simulation

wacc = (equity_amount / total) * cost_of_equity + (debt_amount / total) * cost_of_debt
wacc_percent = wacc * 100  # Convert to percentage for plotting

# ------------------------------
# Compute statistics

mean_wacc = np.mean(wacc_percent)
median_wacc = np.median(wacc_percent)
std_wacc = np.std(wacc_percent)

# ------------------------------
# Plot histogram with statistics

plt.figure(figsize=(10, 6))
plt.hist(wacc_percent, bins=200, color='skyblue', edgecolor='black')

# Add lines for mean, median, and mean ± std
plt.axvline(mean_wacc, color='red', linestyle='--', label=f'Mean: {mean_wacc:.2f}%')
plt.axvline(median_wacc, color='green', linestyle='--', label=f'Median: {median_wacc:.2f}%')
plt.axvline(mean_wacc - std_wacc, color='brown', linestyle='--', label=f'Mean - Std: {mean_wacc - std_wacc:.2f}%')
plt.axvline(mean_wacc + std_wacc, color='purple', linestyle='--', label=f'Mean + Std: {mean_wacc + std_wacc:.2f}%')

plt.title(f'Monte Carlo Simulation of WACC ({n_samples} samples)')
plt.xlabel('WACC (%)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
