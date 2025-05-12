import numpy as np
import matplotlib.pyplot as plt

# Number of Monte Carlo samples
n_samples = 20_000

# Total building area: 6 floors × 500 m² per floor
total_area_m2 = 3000

# ---------------------------------------
# Simulate uncertain unit costs using Beta distributions

# Material cost per m² (in Toman): between 5M and 7.5M
materials_cost_min = 5_000_000
materials_cost_max = 7_500_000
materials_cost = np.random.beta(a=2, b=5, size=n_samples) * (materials_cost_max - materials_cost_min) + materials_cost_min

# Labor cost per m² (in Toman): between 1.5M and 3M
labor_cost_min = 1_500_000
labor_cost_max = 3_000_000
labor_cost = np.random.beta(a=2, b=5, size=n_samples) * (labor_cost_max - labor_cost_min) + labor_cost_min

# Overhead cost per m² (in Toman): between 800k and 1.5M
overhead_cost_min = 800_000
overhead_cost_max = 1_500_000
overhead_cost = np.random.beta(a=2, b=5, size=n_samples) * (overhead_cost_max - overhead_cost_min) + overhead_cost_min

# ---------------------------------------
# Calculate total construction cost
total_cost_per_m2 = materials_cost + labor_cost + overhead_cost
total_cost_project = total_cost_per_m2 * total_area_m2  # for entire project

# ---------------------------------------
# Capital structure and financial assumptions

debt_amount = 5_000_000_000
equity_amount = 5_000_000_000
total_capital = debt_amount + equity_amount
tax_rate = 0.15

# Uncertain interest rate (20% to 40%)
interest_rate_MIN = 0.2
interest_rate_MAX = 0.4
interest_rate = np.random.beta(a=2, b=5, size=n_samples) * (interest_rate_MAX - interest_rate_MIN) + interest_rate_MIN

# Risk-free rate (15% to 35%)
risk_free_rate_MIN = 0.15
risk_free_rate_MAX = 0.35
risk_free_rate = np.random.beta(a=2, b=5, size=n_samples) * (risk_free_rate_MAX - risk_free_rate_MIN) + risk_free_rate_MIN

# Risk premium (expected inflation or market risk): 35% to 50%
risk_premium_MIN = 0.35
risk_premium_MAX = 0.50
risk_premium = np.random.beta(a=2, b=5, size=n_samples) * (risk_premium_MAX - risk_premium_MIN) + risk_premium_MIN

# ---------------------------------------
# Compute cost of capital components
cost_of_debt = interest_rate * (1 - tax_rate)
cost_of_equity = risk_free_rate + risk_premium

# ---------------------------------------
# Calculate WACC for each simulation
wacc = (equity_amount / total_capital) * cost_of_equity + (debt_amount / total_capital) * cost_of_debt
wacc_percent = wacc * 100

# WACC Statistics
mean_wacc = np.mean(wacc_percent)
median_wacc = np.median(wacc_percent)
std_wacc = np.std(wacc_percent)

# ---------------------------------------
# Plot construction cost distribution
plt.figure(figsize=(12, 6))
plt.hist(total_cost_project / 1e9, bins=150, color='lightgreen', edgecolor='black')  # convert to Billion Toman
plt.title(f"Monte Carlo Simulation: 6-Story RC Building Cost ({n_samples} Samples)")
plt.xlabel("Total Project Cost (Billion Toman)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------
# Plot WACC distribution
plt.figure(figsize=(10, 6))
plt.hist(wacc_percent, bins=200, color='skyblue', edgecolor='black')

plt.axvline(mean_wacc, color='red', linestyle='--', label=f'Mean: {mean_wacc:.2f}%')
plt.axvline(median_wacc, color='green', linestyle='--', label=f'Median: {median_wacc:.2f}%')
plt.axvline(mean_wacc - std_wacc, color='orange', linestyle='--', label=f'Mean - Std: {mean_wacc - std_wacc:.2f}%')
plt.axvline(mean_wacc + std_wacc, color='purple', linestyle='--', label=f'Mean + Std: {mean_wacc + std_wacc:.2f}%')

plt.title(f"Monte Carlo Simulation: WACC for Construction Financing ({n_samples} Samples)")
plt.xlabel("WACC (%)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
