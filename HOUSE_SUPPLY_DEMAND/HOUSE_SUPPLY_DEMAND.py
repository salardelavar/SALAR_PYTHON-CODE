"""
Based on aggregated data from official reports of the Statistical Center of Iran
 and municipal construction records, supply and demand curves for the Shiraz housing
 market were developed. Housing supply was represented by the number of issued building 
 permits, while housing demand was approximated by the number of registered housing
 transactions. The results indicate that during recent years, housing demand has declined
 more rapidly than supply, primarily due to reduced purchasing power and increased
 construction costs. This imbalance has led to persistent upward pressure on housing
 prices despite fluctuations in construction activity.
 
Written By Salar Delavar Ghashghaei (Qashqai) 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data from CSV file
df = pd.read_csv("Shiraz_Housing_Supply_Demand.csv")
P = df["Avg_Price_Million_Toman_per_m2"].values
Qs = df["Building_Permits_Shiraz"].values   # Supply
Qd = df["Housing_Transactions"].values      # Demand

# Load data from your own data
P = np.array([40, 58, 52, 68])
Qs = np.array([4200, 4513, 4756, 5891])     # Supply
Qd = np.array([3800, 3500, 2900, 2100])     # # Supply

# Regression (linear fit)
supply_fit = np.polyfit(P, Qs, 1)
demand_fit = np.polyfit(P, Qd, 1)

Z = 0.1        # Sensitivity Parameter (Tolerance Band)
LOWER_LIMIT = min(P)*(1-Z)
UPPER_LIMIT = max(P)*(1+Z)
P_range = np.linspace(LOWER_LIMIT, UPPER_LIMIT, 100)

Qs_line = supply_fit[0]*P_range + supply_fit[1]
Qd_line = demand_fit[0]*P_range + demand_fit[1]

# Plot
plt.figure(figsize=(7,5))
plt.plot(Qs_line, P_range, label="Supply (Construction Permits)", color='red')
plt.plot(Qd_line, P_range, label="Demand (Transactions)", color='black')

plt.scatter(Qs, P, color='red')
plt.scatter(Qd, P, color='black')

plt.xlabel("Quantity (Units)")
plt.ylabel("Price (Million Toman / m²)")
plt.title("Supply and Demand Curves – Shiraz Housing Market")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
