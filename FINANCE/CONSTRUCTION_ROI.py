"""
This is a simple example of conducting a housing market scenario analysis with three scenarios: “Optimistic,” “Moderate,” and “Pessimistic.”

- Define base parameters: 
    average price per unit, construction cost per unit, and demand factor.
- Set scenario assumptions: 
    for each scenario, specify the inflation rate, cost increase, and change in demand.
- Calculate key metrics: 
    compute revenue, actual cost, absolute profit, and return on investment (ROI)
    for each scenario.
- Present results: 
    display the metrics in a DataFrame and plot a bar chart of ROI by scenario.

##############################################################################################

1. Current Market Analysis

   First, build a clear picture of today’s housing market:
   - Prices: average housing prices across regions, price-to-income ratios, rent-to-price ratios
   - Supply & Demand: inventory of completed units, projects under construction, active vs. latent demand
   - Macro Indicators: inflation rate, bank lending rates, unemployment rate, GDP growth
   - Government Policies: mortgage subsidies, support programs (e.g. national housing initiatives), taxes (empty-home levies, transaction taxes)
   - Investor Behavior: appetite for real estate versus alternative assets (stocks, gold, foreign exchange)

2. Risk Identification

   Group the key risks into categories:

   Economic Risks
   - Sharp currency and inflation swings
   - Changes in interest rates affecting mortgage demand
   - Economic downturns and lower household incomes

   Policy Risks
   - Tightening of construction regulations
   - Changes in tax law or lending rules
   - Regulatory instability

   Construction-Related Risks
   - Rising material and labor costs
   - Delays in land acquisition or permitting
   - Shortage of skilled labor

   Socio-Demographic Risks
   - Reverse migration or shifts in settlement patterns
   - Declining youth population or smaller household sizes

3. Scenario Design

   Combine the main variables into a few plausible scenarios. For example:

   Optimistic Scenario (Stable Growth)
   - Gradual easing of inflation
   - Stable exchange rate
   - Targeted government support for supply and demand
   - Steady rise in consumer demand

   Moderate Scenario (Stagflation)
   - High inflation but weak demand
   - Reduced household purchasing power
   - Increased construction but slower sales

   Pessimistic Scenario (Supply Crunch & Demand Slump)
   - Sharp currency devaluation
   - Sudden spike in construction costs
   - Halted or delayed projects
   - Higher interest rates, further reducing buying power

4. Sensitivity Analysis

   Using Excel or financial modeling tools, test how key variables affect outcomes:
   - Project profitability versus 10% vs. 40% inflation
   - ROI impact if sales are delayed by one year
   - Break-even shifts with a 30% rise in material costs

5. Conclusions & Decision-Making

   With scenario and risk analysis in hand, make informed choices about:
   - Timing: when to enter the market or launch a project
   - Location: urban, suburban, or redevelopment areas
   - Use Type: residential, commercial, or mixed-use
   - Financing Strategy: equity vs. debt mix, timing of capital calls
   
THIS PROGRAM WRITTEN BY SALAR DELAVAR GHASHGHAEI (QASHQAI) 
EMAIL: salar.d.ghashghaei@gmail.com   

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Base parameters
base_price = 10000     # average price per unit
base_cost = 7000       # construction cost per unit
base_demand = 1.0      # demand factor (100%)

# Define scenarios
scenarios = {
    'Optimistic':    {'inflation': 0.10, 'cost_increase': 0.05, 'demand_factor': 1.10},
    'Moderate':      {'inflation': 0.25, 'cost_increase': 0.15, 'demand_factor': 0.90},
    'Pessimistic':   {'inflation': 0.40, 'cost_increase': 0.30, 'demand_factor': 0.75},
}

# Calculate profit and ROI for each scenario
results = []
for name, params in scenarios.items():
    revenue = base_price * params['demand_factor']
    cost = base_cost * (1 + params['inflation'] + params['cost_increase'])
    profit = revenue - cost
    roi = (profit / cost) * 100
    
    results.append({
        'Scenario': name,
        'Revenue': round(revenue, 2),
        'Cost': round(cost, 2),
        'Profit': round(profit, 2),
        'ROI (%)': round(roi, 2)
    })

# Create DataFrame
df = pd.DataFrame(results)
print(df)
# Display the results in an interactive table

# Plot ROI for each scenario
plt.figure()
plt.bar(df['Scenario'], df['ROI (%)'])
plt.title('ROI by Scenario')
plt.xlabel('Scenario')
plt.ylabel('ROI (%)')
plt.tight_layout()
plt.show()
