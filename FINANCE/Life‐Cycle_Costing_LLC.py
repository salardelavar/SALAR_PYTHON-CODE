"""
Life‐Cycle Costing (LCC) in construction is a methodology for assessing the total cost of
 a building (or infrastructure) over its entire lifespan—from initial planning and design,
 through construction and operation, all the way to disposal or recycling.
 It helps decision-makers choose design alternatives and materials that minimize the
 present value of all costs, rather than just the upfront investment.

1. Key Cost Categories
- Initial (Capital) Costs
- Land acquisition
- Design & engineering fees
- Site works & construction
- Permits, inspections, financing fees

2. Operating & Maintenance Costs
- Energy (heating, cooling, lighting, equipment)
- Water and wastewater services
- Routine maintenance (repairs, cleaning, minor upgrades)
- Major replacements (roofs, HVAC systems, elevators)

3. End-of-Life Costs
- Demolition or deconstruction
- Waste treatment or recycling
- Site restoration or remediation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
years = np.arange(0, 41)
discount_rate = 0.03

# Option A: Concrete Structure
cfA = np.zeros_like(years, dtype=float)
cfA[0] = 50000                        # Initial cost
cfA[1:] += 1000                       # Annual maintenance
cfA[20] += 50000                      # Replacement at year 20

# Option B: Metal Structure
cfB = np.zeros_like(years, dtype=float)
cfB[0] = 80000                        # Initial cost
cfB[1:] += 500                        # Annual maintenance
cfB[40] += 5000                       # Disposal at end-of-life

# Discount factor
discount_factors = (1 + discount_rate) ** years

# Present Value of cash flows
pvA = cfA / discount_factors
pvB = cfB / discount_factors

# Cumulative LCC over time
cum_pvA = np.cumsum(pvA)
cum_pvB = np.cumsum(pvB)

# Create DataFrame for display
df = pd.DataFrame({
    'Year': years,
    'CF Option A ($)': cfA,
    'PV Option A ($)': pvA,
    'Cum. PV Option A ($)': cum_pvA,
    'CF Option B ($)': cfB,
    'PV Option B ($)': pvB,
    'Cum. PV Option B ($)': cum_pvB,
})

# Plot cumulative present value
plt.figure()
plt.plot(years, cum_pvA, label='Option A (Asphalt)')
plt.plot(years, cum_pvB, label='Option B (Metal)')
plt.title('Cumulative Present Value of Lifecycle Costs')
plt.xlabel('Year')
plt.ylabel('Cumulative PV ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print final LCC values
print(f"Total LCC over 40 years:\n"
      f" - Option A: €{cum_pvA[-1]:,.2f}\n"
      f" - Option B: €{cum_pvB[-1]:,.2f}")

