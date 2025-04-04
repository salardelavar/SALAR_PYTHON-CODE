#####   #   INSURANCE RISK ANALYSIS

"""
1. Linear Regression Model for Claim Amount Prediction
Purpose: Predicts claim amounts based on policyholder characteristics.

Key Components:
- Data Structure:
  - `Age`: Policyholder age (sorted min to max)
  - `Car_Type`: Vehicle classification (1=Standard, 2=Sports, 3=Luxury)
  - `Claim_Amount`: Historical claim amounts in IRR

Analysis:
- The model quantifies how age and vehicle type influence claim severity
- Coefficients show the marginal effect of each variable:
  - Positive coefficient = Higher expected claims
  - Negative coefficient = Lower expected claims
- The intercept represents baseline claim amount when all predictors are zero

Actuarial Application:
- Premium pricing based on risk factors
- Identifying high-risk policyholder segments
- Reserve setting for expected claims
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample data
data = {
    'Age': [25, 30, 35, 40, 45, 50], # SORT MIN TO MAX
    'Car_Type': [1, 2, 1, 3, 2, 1],  # 1: Standard, 2: Sports, 3: Luxury
    'Claim_Amount': [2000000, 3500000, 1500000, 5000000, 3000000, 1800000]  # Claim amount (IRR)
}
df = pd.DataFrame(data)

# Define independent (X) and dependent (y) variables
X = df[['Age', 'Car_Type']]
y = df['Claim_Amount']

# Linear regression model
model = LinearRegression()
model.fit(X, y)

# Predictions
predictions = model.predict(X)

# Display model coefficients and results
print("Model coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Predictions:", predictions)

# Plot
plt.scatter(df['Age'], y, color='blue', label='Actual data')
plt.plot(df['Age'], predictions, color='red', label='Predictions')
plt.xlabel('Driver Age')
plt.ylabel('Claim Amount (IRR)')
plt.legend()
plt.show()

#--------------------------------------------------------------------
"""
2. Kaplan-Meier Survival Analysis
Purpose: Estimates time-to-first-claim probabilities.

Key Components:
- `Time`: Duration until claim event or censoring
- `Event`: Binary indicator (1=claim occurred, 0=censored)

Output Interpretation:
- Survival curve shows probability of remaining claim-free over time
- Censored data points represent policies that exited observation without claims
- Steeper drops indicate higher claim incidence periods

Actuarial Application:
- Persistence analysis of risk-free policyholders
- Claim incidence rate modeling
- Product design and warranty period optimization
"""

import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# Sample data
data = {
    'Time': [5, 10, 15, 20, 25, 30],  # Time until first claim (months)
    'Event': [1, 0, 1, 1, 0, 1]       # 1: Claim occurred, 0: Censored (no claim yet)
}
df = pd.DataFrame(data)

# Kaplan-Meier model
kmf = KaplanMeierFitter()
kmf.fit(df['Time'], event_observed=df['Event'])

# Plot survival function
kmf.plot_survival_function()
plt.title('Survival Function for Policyholders')
plt.xlabel('Time (months)')
plt.ylabel('Probability of No Claim')
plt.show()

# Survival probability at specific times
print("Survival probabilities at different times:")
print(kmf.survival_function_)

#--------------------------------------------------------------------
"""
3. Markov Chain for Claim State Transitions
Purpose: Models the evolution of claim states over time.

State Definitions:
- 0: No claim
- 1: Minor claim
- 2: Major claim

Transition Matrix Features:
- Rows sum to 1 (probabilistic transitions)
- Diagonal elements show state persistence
- Off-diagonals show transition likelihoods

Simulation Output:
- Shows probability distribution across states at each period
- Convergence patterns indicate long-term claim behavior

Actuarial Application:
- Multi-state claim modeling
- Reserve calculations for different claim types
- Product pricing tiered by claim severity
- Risk capital allocation

Technical Implementation Notes:
1. Data Preparation:
   - Age is sorted to ensure proper visualization
   - Categorical variables are numerically encoded

2. Model Validation:
   - While not shown here, real implementations would include:
     - Train-test splits
     - Residual analysis
     - Confidence intervals for survival estimates
     - Goodness-of-fit tests for Markov chains

3. Visualization:
   - Each model includes graphical outputs showing:
     - Actual vs predicted values (regression)
     - Survival probability curves
     - State evolution over time

This comprehensive analysis provides insurers with:
- Predictive models for claim amounts
- Temporal understanding of claim emergence
- Dynamic modeling of claim severity progression
"""

import numpy as np

# Sample transition matrix (probabilities between states)
# States: 0 (No claim), 1 (Minor claim), 2 (Major claim)
transition_matrix = np.array([
    [0.7, 0.2, 0.1],  # From No claim
    [0.4, 0.5, 0.1],  # From Minor claim
    [0.3, 0.3, 0.4]   # From Major claim
])

# Initial state (assuming starting with No claim)
state = np.array([1, 0, 0])

# Simulation for 5 steps
states_over_time = [state]
for _ in range(5):
    state = state @ transition_matrix  # Matrix multiplication to calculate next state
    states_over_time.append(state)

# Display results
for i, s in enumerate(states_over_time):
    print(f"Step {i}: State probabilities = {s}")

# Plot
import matplotlib.pyplot as plt
states_over_time = np.array(states_over_time)
plt.plot(states_over_time[:, 0], label='No claim')
plt.plot(states_over_time[:, 1], label='Minor claim')
plt.plot(states_over_time[:, 2], label='Major claim')
plt.xlabel('Step')
plt.ylabel('Probability')
plt.legend()
plt.show()