"""
This example demonstrates how to manage financial uncertainty in a construction project,
 focusing on concrete and rebar costs. We'll use a Monte Carlo simulation to account for
 price fluctuations of these key materials.
Hypothetical Project Scenario:
A building construction project requiring concrete for various elements (foundations, columns, slabs)
 and rebar for reinforcement.
Assumptions:
 + Consumption:
   - Concrete: 1500 cubic meters (m³)
   - Rebar: 500 tons
 + Base Price (Initial Estimate):
   - Concrete: 3,500,000 IRR per m³
   - Rebar: 20,000,000 IRR per ton
 + Uncertainty (Standard Deviation for Normal Distribution):
   - Concrete: 12% (i.e., 420,000 IRR per m³)
   - Rebar: 15% (i.e., 3,000,000 IRR per ton)
 + Number of Simulations: 10,000
------------------------------------------------
THIS PROGRAM WRITTEN BY SALAR DELAVAR GHASHGHAEI (QASHQAI)
EMAIL: salar.d.ghashghaei@gmail.com   
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%% --- Input Data ---
# Material Consumption
consumption_concrete = 1500  # cubic meters
consumption_rebar = 500      # tons

# Base Price (Expected Mean)
base_price_concrete = 3_500_000   # IRR per m3
base_price_rebar = 20_000_000     # IRR per ton

# Price Standard Deviation (Uncertainty) as a percentage of base price
std_dev_percent_concrete = 0.12  # 12%
std_dev_percent_rebar = 0.15     # 15%

# Calculate actual standard deviation
std_dev_concrete = base_price_concrete * std_dev_percent_concrete
std_dev_rebar = base_price_rebar * std_dev_percent_rebar

# Number of Simulations
num_simulations = 10000

#%% --- Monte Carlo Simulation ---
# Generate normal distribution for prices
# Ensure prices are not negative
simulated_prices_concrete = np.maximum(0, np.random.normal(base_price_concrete, std_dev_concrete, num_simulations))
simulated_prices_rebar = np.maximum(0, np.random.normal(base_price_rebar, std_dev_rebar, num_simulations))

# Calculate total material cost for each simulation
simulated_total_costs = (simulated_prices_concrete * consumption_concrete) + \
                        (simulated_prices_rebar * consumption_rebar)

# --- Analyze Results ---
# Mean predicted total cost
mean_simulated_cost = np.mean(simulated_total_costs)

# Minimum and maximum predicted cost
min_simulated_cost = np.min(simulated_total_costs)
max_simulated_cost = np.max(simulated_total_costs)

# Calculate percentiles (e.g., 25th, 50th, and 75th percentiles)
percentile_25 = np.percentile(simulated_total_costs, 25)
percentile_50 = np.percentile(simulated_total_costs, 50) # Median
percentile_75 = np.percentile(simulated_total_costs, 75)

#%% --- Display Results ---
print(f"Number of simulations: {num_simulations}")
print(f"\n--- Material Cost Simulation Results ---")
print(f"Mean predicted cost: {mean_simulated_cost:,.0f} IRR")
print(f"Minimum predicted cost: {min_simulated_cost:,.0f} IRR")
print(f"Maximum predicted cost: {max_simulated_cost:,.0f} IRR")
print(f"25th percentile (cost with 25% probability of being lower): {percentile_25:,.0f} IRR")
print(f"Median (50th percentile): {percentile_50:,.0f} IRR")
print(f"75th percentile (cost with 75% probability of being lower): {percentile_75:,.0f} IRR")

#%% --- Plotting the histogram of cost distribution ---
plt.figure(figsize=(12, 8))
sns.histplot(simulated_total_costs / 1_000_000_000, bins=50, kde=True, color='lightcoral') # Divide by one billion for better readability
plt.axvline(mean_simulated_cost / 1_000_000_000, color='blue', linestyle='dashed', linewidth=4, label=f'Mean: {mean_simulated_cost/1e9:,.2f} billion IRR')
plt.axvline(percentile_75 / 1_000_000_000, color='darkgreen', linestyle='dashed', linewidth=4, label=f'75th Percentile: {percentile_75/1e9:,.2f} billion IRR')
plt.title('Simulated Distribution of Total Project Material Costs (Concrete & Rebar)')
plt.xlabel('Total Material Cost (Billion IRR)')
plt.ylabel('Number of Simulations')
plt.legend()
plt.grid(axis='y', alpha=0.75)
plt.show()
