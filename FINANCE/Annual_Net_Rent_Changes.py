#### Simulation and Visualization of Net Rent Changes Over 30 Years with Inflation and Depreciation
import matplotlib.pyplot as plt

# Input parameters
initial_house_value = 1_000_000                 # intial house value (in dollars or any currency)
P_R = 6                                         # price to rent ratio
initial_rent = initial_house_value / P_R        # initial rent (in dollars or any currency)
annual_inflation_rate = 0.05                    # annual inflation rate (e.g., 5%)
annual_depreciation_rate = (0.1/1)**(1/30) - 1  # annual depreciation rate
years = 30                                      # rental period in years
#----------------------------------------------------------------------------------
# List to store the net rent for each year
rent_schedule = []

for year in range(1, years + 1):
    # Calculate the rent adjusted for inflation
    inflated_rent = initial_rent * ((1 + annual_inflation_rate) ** year)
    # Apply depreciation to the inflation-adjusted rent
    net_rent = inflated_rent * ((1 + annual_depreciation_rate) ** year)
    rent_schedule.append(net_rent)

# Print the net rent for each year
for year, rent in enumerate(rent_schedule, start=1):
    print(f"Year {year}: Net Rent = ${rent:.2f}")
#----------------------------------------------------------------------------------
# Plot the net rent over the years
years_range = list(range(1, years + 1))
plt.figure(figsize=(10, 6))
plt.plot(years_range, rent_schedule, marker='o', linestyle='-', color='b')
plt.xlabel('Year')
plt.ylabel('Net Rent')
plt.title('Annual Net Rent Changes')
plt.grid(True)
plt.show()
#----------------------------------------------------------------------------------
