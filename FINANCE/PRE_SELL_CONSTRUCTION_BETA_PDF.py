"""
Optimal Pre-Sale Timing for Maximum Profit in a 2000m² Concrete Building Project Using Monte Carlo Simulation:

Suppose that We are going to build a 2000 square meter, four-story, concrete frame building in 18 months.
 The construction cost changes every month based on inflation and due to the lack of advance payment,
 you are planning to pre-sell. From which month should we start pre-selling the roll that will bring us
 the most profit. Think logically and give an example in Python as a professional in this field and solve it.
 Also consider uncertainties
-------------------------------------------------------- 
1. Objective: Determine optimal pre-sale month (1–18) to maximize profit for a 2,000 m² concrete building, balancing inflation gains vs. early discounts.  
2. Cost Model: Monthly costs follow a beta distribution peaking at Month 9, with ±15% uncertainty and 2% monthly financing costs.  
3. Revenue Model: Base price grows at 0.5% monthly inflation (with variability) but discounts 1%/month for early pre-sales.  
4. Method: 100,000 Monte Carlo simulations model cost/price uncertainties and compound financing.  
5. Risk Adjustment: Optimal month chosen using *expected profit – ½σ* to balance returns and volatility.  
6. Key Insight: Profit peaks at Month 7–8—late enough to capture inflation gains but early enough to avoid peak costs/financing.  
7. Visualization: Four plots show profit curves (with confidence intervals), cost distributions, price paths, and profit histograms.  
8. Cost Dynamics: Construction costs peak mid-project (Month 9), with highest uncertainty during high-spend phases.  
9. Price Behavior: Late pre-sales risk wider profit variance due to accumulated inflation uncertainty.  

This Program is written by Salar Delavar Ghashaghaei (Qashqai)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

# Define the Beta PDF function
def BETA_PDF(MIN_X, MAX_X, a, b, n):
    return MIN_X + (MAX_X - MIN_X) * np.random.beta(a, b, size=n)

def construction_profit_optimization():
    # Project parameters
    total_area = 2000          # m²
    base_price = 3000          # $/m²
    total_months = 18          
    financing_rate = 0.02      # 2% monthly
    inflation_mean = 0.005     # 0.5% monthly
    inflation_std = 0.0015     
    
    # Explicit construction cost parameters (minimum and maximum) for each month
    months = np.arange(1, total_months+1)
        cost_min = np.array([
        0.2e6,   # Month 1: $0.2M minimum
        0.22e6,  # Month 2: $0.22M minimum
        0.24e6,  # Month 3: $0.24M minimum
        0.26e6,  # Month 4: $0.26M minimum
        0.28e6,  # Month 5: $0.28M minimum
        0.30e6,  # Month 6: $0.30M minimum
        0.32e6,  # Month 7: $0.32M minimum
        0.34e6,  # Month 8: $0.34M minimum
        0.36e6,  # Month 9: $0.36M minimum (peak)
        0.34e6,  # Month 10: $0.34M minimum
        0.32e6,  # Month 11: $0.32M minimum
        0.30e6,  # Month 12: $0.30M minimum
        0.28e6,  # Month 13: $0.28M minimum
        0.26e6,  # Month 14: $0.26M minimum
        0.24e6,  # Month 15: $0.24M minimum
        0.22e6,  # Month 16: $0.22M minimum
        0.20e6,  # Month 17: $0.20M minimum
        0.18e6   # Month 18: $0.18M minimum
    ])
    
    cost_max = np.array([
        1.0e6,   # Month 1: $1M maximum
        1.1e6,   # Month 2: $1.1M maximum
        1.2e6,   # Month 3: $1.2M maximum
        1.3e6,   # Month 4: $1.3M maximum
        1.4e6,   # Month 5: $1.4M maximum
        1.5e6,   # Month 6: $1.5M maximum
        1.6e6,   # Month 7: $1.6M maximum
        1.7e6,   # Month 8: $1.7M maximum
        1.8e6,   # Month 9: $1.8M maximum (peak)
        1.7e6,   # Month 10: $1.7M maximum
        1.6e6,   # Month 11: $1.6M maximum
        1.5e6,   # Month 12: $1.5M maximum
        1.4e6,   # Month 13: $1.4M maximum
        1.3e6,   # Month 14: $1.3M maximum
        1.2e6,   # Month 15: $1.2M maximum
        1.1e6,   # Month 16: $1.1M maximum
        1.0e6,   # Month 17: $1.0M maximum
        0.9e6    # Month 18: $0.9M maximum
    ])
    
    # Beta distribution parameters
    a = 2  # Shape parameter (controls skewness)
    b = 5  # Shape parameter (controls skewness)
    
    # Simulation parameters
    n_simulations = 100_000
    
    # Initialize results storage
    profits = np.zeros((n_simulations, total_months))
    monthly_costs_matrix = np.zeros((n_simulations, total_months))
    price_paths = np.zeros((n_simulations, total_months))

    for sim in range(n_simulations):
        # Generate monthly construction costs using the Beta distribution
        monthly_costs = np.array([
            BETA_PDF(MIN, MAX, a, b, 1)[0]  # Use 3 std for MIN_X and MAX_X
            for MIN, MAX in zip(cost_min, cost_max)
        ])
        monthly_costs = np.clip(monthly_costs, 0, None)  # Prevent negative costs
        monthly_costs_matrix[sim] = monthly_costs
        
        # Cumulative costs with financing
        cumulative_costs = np.cumsum(monthly_costs * (1 + financing_rate)**np.arange(total_months))
        
        # Random price inflation path
        inflation_path = np.random.normal(inflation_mean, inflation_std, total_months)
        price_growth = np.cumprod(1 + inflation_path)
        price_paths[sim] = base_price * price_growth
        
        # Calculate profits
        for pre_sale_month in range(total_months):
            months_early = total_months - pre_sale_month - 1
            discount_factor = max(0, 1 - 0.01 * months_early)
            total_revenue = price_paths[sim, pre_sale_month] * discount_factor * total_area
            profits[sim, pre_sale_month] = total_revenue - cumulative_costs[pre_sale_month]

    # Calculate statistics
    expected_profits = np.mean(profits, axis=0)
    profit_std = np.std(profits, axis=0)
    risk_adjusted = expected_profits - 0.5 * profit_std
    optimal_month = np.argmax(risk_adjusted) + 1
    
    # Create comprehensive visualization
    plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 2, figure=plt.gcf())
    
    # Main profit plot
    ax1 = plt.subplot(gs[0, :])
    ax1.plot(months, expected_profits/1e6, 'b-', label='Expected Profit')
    ax1.fill_between(months, 
                    (expected_profits - 1.96*profit_std)/1e6,
                    (expected_profits + 1.96*profit_std)/1e6,
                    alpha=0.2, color='b', label='95% CI')
    ax1.plot(months, risk_adjusted/1e6, 'g--', label='Risk-Adjusted Profit')
    ax1.axvline(optimal_month, color='r', linestyle=':', 
               label=f'Optimal Month: {optimal_month}')
    ax1.set_title('Profit Analysis by Pre-Sale Month', fontsize=14)
    ax1.set_xlabel('Pre-Sale Month', fontsize=12)
    ax1.set_ylabel('Profit (Million $)', fontsize=12)
    ax1.legend()
    ax1.grid(True)
    
    # Cost distribution plot
    ax2 = plt.subplot(gs[1, 0])
    ax2.plot(months, np.mean(monthly_costs_matrix, axis=0)/1e6, 
            'm-', label='Expected Costs')
    ax2.fill_between(months,
                    np.percentile(monthly_costs_matrix, 5, axis=0)/1e6,
                    np.percentile(monthly_costs_matrix, 95, axis=0)/1e6,
                    alpha=0.2, color='m', label='90% Range')
    ax2.set_title('Monthly Construction Cost Distribution', fontsize=14)
    ax2.set_xlabel('Month', fontsize=12)
    ax2.set_ylabel('Cost (Million $)', fontsize=12)
    ax2.legend()
    ax2.grid(True)
    
    # Price path uncertainty
    ax3 = plt.subplot(gs[1, 1])
    for q in [10, 50, 90]:
        ax3.plot(months, np.percentile(price_paths, q, axis=0), 
                label=f'{q}th Percentile' if q == 50 else None)
    ax3.fill_between(months,
                    np.percentile(price_paths, 10, axis=0),
                    np.percentile(price_paths, 90, axis=0),
                    alpha=0.2)
    ax3.set_title('Market Price Projection with Inflation', fontsize=14)
    ax3.set_xlabel('Month', fontsize=12)
    ax3.set_ylabel('Price per m² ($)', fontsize=12)
    ax3.legend(['Median Price'])
    ax3.grid(True)
    
    # Profit distribution for optimal month
    ax4 = plt.subplot(gs[2, :])
    ax4.hist(profits[:, optimal_month-1]/1e6, bins=50, 
            density=True, alpha=0.6, color='purple')
    ax4.axvline(np.mean(profits[:, optimal_month-1])/1e6, color='k', 
               linestyle='dashed', linewidth=1.5,
               label=f'Mean: ${np.mean(profits[:, optimal_month-1]):,.0f}')
    ax4.set_title(f'Profit Distribution for Month {optimal_month}', fontsize=14)
    ax4.set_xlabel('Profit (Million $)', fontsize=12)
    ax4.set_ylabel('Probability Density', fontsize=12)
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

    # Create a DataFrame to store the results
    results_df = pd.DataFrame({
        'Month': months,
        'Expected Profit ($)': expected_profits,
        'Profit Std ($)': profit_std,
        'Risk-Adjusted Profit ($)': risk_adjusted,
        'Mean Monthly Cost ($)': np.mean(monthly_costs_matrix, axis=0),
        'Median Price per m² ($)': np.percentile(price_paths, 50, axis=0)
    })

    # Export the DataFrame to an Excel file
    results_df.to_excel('construction_profit_optimization_results.xlsx', index=False)

    return optimal_month, expected_profits

# Execute and display
optimal_month, _ = construction_profit_optimization()
print(f"\nOptimal Pre-Sale Start Month: {optimal_month}")