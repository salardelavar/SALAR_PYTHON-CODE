"""
We are going to build a 2000 square meter, four-story, concrete frame building in 18 months.
 The construction cost changes every month based on inflation and due to the lack of advance payment,
 you are planning to pre-sell. From which month should we start pre-selling the roll that will bring us
 the most profit. Think logically and give an example in Python as a professional in this field and solve it.
 Also consider uncertainties
"""
import numpy as np
import matplotlib.pyplot as plt

def BETA_PDF(MIN_X, MAX_X, a, b, n):
    import numpy as np
    return MIN_X + (MAX_X - MIN_X) * np.random.beta(a, b, size=n)

# Simulation parameters
num_simulations = 10000
area = 2000                          # square meters
base_sale_price = 1500               # $ per m² (on completion)
base_construction_cost = 2000 * 1000   # total base cost = $2,000,000
financing_cost_total = 0.36          # 36% financing penalty if you wait full 18 months
discount_rate = 0.01                 # discount of 1% per month before completion
r_sale_mean = 0.005                  # average monthly market price growth (0.5%) - INFLATION
r_sale_std = 0.002                   # uncertainty in monthly market price growth - INFLATION

# Pre-sale months (1 = earliest, 18 = on completion)
pre_sale_months = np.arange(1, 19)
average_profits = []

# Monte Carlo simulation for each pre-sale month
for k in pre_sale_months:
    months_to_completion = 18 - k
    sim_profits = []
    for _ in range(num_simulations):
        # Sample a sale inflation rate (simulate uncertainty)
        r_sale = BETA_PDF(r_sale_mean-r_sale_std, r_sale_mean+r_sale_std, a=2, b=1, n=1)
        #r_sale = np.random.normal(r_sale_mean, r_sale_std)
        sale_multiplier = (1 + r_sale) ** months_to_completion
        
        # Discount required for pre-selling early: ensure the discount doesn't drop below zero.
        discount_multiplier = max(0, 1 - discount_rate * months_to_completion)
        
        sale_price = base_sale_price * sale_multiplier * discount_multiplier
        revenue = sale_price * area
        
        # Effective construction cost: financing cost is proportionally less the earlier you pre-sell.
        cost_factor = 1 + financing_cost_total * (k / 18)
        cost = base_construction_cost * cost_factor
        
        profit = revenue - cost
        sim_profits.append(profit)
    average_profit = np.mean(sim_profits)
    average_profits.append(average_profit)

# Find the pre-sale month with the maximum average profit
optimal_month = pre_sale_months[np.argmax(average_profits)]
print("Optimal pre-sale start month:", optimal_month)

# Print average profit for each month
for month, profit in zip(pre_sale_months, average_profits):
    print(f"Month {month:2d}: Average Profit = ${profit:,.0f}")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(pre_sale_months, average_profits, marker='o', linestyle='-', color='b')
plt.axvline(optimal_month, color='r', linestyle='--', label=f'Optimal Month: {optimal_month}')
plt.title("Average Profit vs. Pre-sale Start Month")
plt.xlabel("Pre-sale Start Month")
plt.ylabel("Average Profit ($)")
plt.xticks(pre_sale_months)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
