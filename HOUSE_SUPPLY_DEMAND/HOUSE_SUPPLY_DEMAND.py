"""
Supply–Demand Imbalance and Persistent Price Pressures in the Housing Market:
    
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


# Linear regression
supply_fit = np.polyfit(P, Qs, 1)   # Qs = a_s P + b_s
demand_fit = np.polyfit(P, Qd, 1)   # Qd = a_d P + b_d

a_s, b_s = supply_fit
a_d, b_d = demand_fit


# Equilibrium price & quantity
P_eq = (b_d - b_s) / (a_s - a_d)
Q_eq = a_s * P_eq + b_s


# Price range
Z = 0.1
P_range = np.linspace(min(P)*(1-Z), max(P)*(1+Z), 100)

Qs_line = a_s * P_range + b_s
Qd_line = a_d * P_range + b_d


# Plot
plt.figure(figsize=(7,5))

plt.plot(Qs_line, P_range, label="Supply (Construction Permits)", color='red')
plt.plot(Qd_line, P_range, label="Demand (Transactions)", color='black')

plt.scatter(Qs, P, color='red')
plt.scatter(Qd, P, color='black')

# Equilibrium point
plt.scatter(Q_eq, P_eq, color='blue', s=80, zorder=5)
plt.axhline(P_eq, linestyle='--', color='blue', alpha=0.6)
plt.axvline(Q_eq, linestyle='--', color='blue', alpha=0.6)

plt.text(Q_eq*1.01, P_eq,
         f'Equilibrium\nP ≈ {P_eq:.1f}\nQ ≈ {Q_eq:.0f}',
         color='blue')

plt.xlabel("Quantity (Units)")
plt.ylabel("Price (Million Toman / m²)")
plt.title("Supply and Demand with Equilibrium – Shiraz Housing Market")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Print equilibrium values
print(f"Equilibrium Price: {P_eq:.2f} Million Toman per m²")
print(f"Equilibrium Quantity: {Q_eq:.0f} Units")
