import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Input data
stories = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
bays    = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
Ke      = np.array([
    127.5,   189.0, 291.0938,   # 1‑story
     56.7,    83.3625, 121.5,   # 2‑story
     33.66,   43.56,   55.6875  # 3‑story
])

# Build DataFrame
df = pd.DataFrame({
    'Stories':   stories,
    'Bays':      bays,
    'Ke_actual': Ke
})

# 1) Fit with sklearn (for predictions)
model_sk = LinearRegression()
X = df[['Stories', 'Bays']]
y = df['Ke_actual']
model_sk.fit(X, y)
df['Ke_predicted'] = model_sk.predict(X)

# 2) Fit with statsmodels (for R^2 and p‑values)
#    – add constant term
X_sm = sm.add_constant(X)  
ols = sm.OLS(y, X_sm).fit()

# Print out the summary (includes R^2, coefficients & p‑values)
print(ols.summary())

# If you want just R^2 and the p‑values:
print(f"\nR-squared: {ols.rsquared:.4f}")
print("p-values:")
print(ols.pvalues)

# 3) Plot actual vs. predicted (as before)
plt.figure()
for s in sorted(df['Stories'].unique()):
    subset = df[df['Stories'] == s]
    plt.plot(subset['Bays'], subset['Ke_actual'],
             marker='o', label=f'Actual (Stories={s})')
    plt.plot(subset['Bays'], subset['Ke_predicted'],
             marker='x', linestyle='--',
             label=f'Predicted (Stories={s})')
plt.xlabel('Number of Bays')
plt.ylabel('Kp (plastic stiffness)')
plt.title('Actual vs. Predicted Ke by Number of Stories')
plt.legend()
plt.tight_layout()
plt.show()


# Extract coefficients
coeffs = ols.params

# Define a prediction function using the fitted model
def predict_Ke(stories, bays):
    return (
        coeffs['const']
        + coeffs['Stories'] * stories
        + coeffs['Bays'] * bays
    )

# Example: test the formula with stories=2, bays=2
example_Ke = predict_Ke(3, 3)
print(f"\n\n Predicted Ke for 2 stories and 2 bays = {example_Ke:.2f} \n\n")

#%% 3D PLOT
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# 1. Prepare the data
#stories = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
#bays    = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
Ke      = Ke

df = pd.DataFrame({
    'Stories':   stories,
    'Bays':      bays,
    'Ke_actual': Ke
})

# 2. Fit a linear regression model
model = LinearRegression()
model.fit(df[['Stories', 'Bays']], df['Ke_actual'])

# 3. Create a mesh grid for Stories and Bays
story_vals = np.linspace(df['Stories'].min(), df['Stories'].max(), 20)
bay_vals   = np.linspace(df['Bays'].min(), df['Bays'].max(), 20)
S, B = np.meshgrid(story_vals, bay_vals)

# 4. Predict Ke over the grid
grid_points = np.column_stack([S.ravel(), B.ravel()])
Ke_surface = model.predict(grid_points).reshape(S.shape)

# 5. Plot in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#   a. Actual data points
ax.scatter(df['Stories'], df['Bays'], df['Ke_actual'],
           color='blue', marker='o', label='Actual Ke')

#   b. Predicted surface
ax.plot_surface(S, B, Ke_surface,
                alpha=0.5, cmap='viridis', edgecolor='none')

ax.set_xlabel('Number of Stories')
ax.set_ylabel('Number of Bays')
ax.set_zlabel('Ke (elastic stiffness)')
ax.set_title('3D Plot of Actual Ke and Predicted Surface')
ax.legend()

plt.tight_layout()
plt.show()




