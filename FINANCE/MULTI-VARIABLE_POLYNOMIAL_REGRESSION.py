### MULTI-VARIABLE POLYNOMIAL REGRESSION MODEL SELECTION AND EVALUATION

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
#-------------------------------------------------------
# Generate random data
NUM_SIM = 1000
X = np.random.rand(NUM_SIM, 4)  # NUM_SIM samples with 4 features
Y = 3*X[:,0]**2 + 2*X[:,1] + 1.5*X[:,2]**3 + 0.5*X[:,3] + np.random.randn(NUM_SIM) * 0.1  # Nonlinear relationship

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#-------------------------------------------------------
# List of different models
models = {
    "Linear": LinearRegression(),
    "Degree 2": PolynomialFeatures(degree=2),
    "Degree 3": PolynomialFeatures(degree=3), 
    "Degree 4": PolynomialFeatures(degree=4), 
    "Degree 5": PolynomialFeatures(degree=5), 
    "Degree 6": PolynomialFeatures(degree=6), 
    "Degree 7": PolynomialFeatures(degree=7), 
    "Degree 8": PolynomialFeatures(degree=8), 
    "Degree 9": PolynomialFeatures(degree=9), 
}
#-------------------------------------------------------
best_r2 = -np.inf
best_model = None
best_name = ""
r2_scores = {}

for name, model in models.items():
    if isinstance(model, PolynomialFeatures):
        poly = model
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        regressor = LinearRegression()
        regressor.fit(X_train_poly, Y_train)
        Y_pred = regressor.predict(X_test_poly)
    else:
        regressor = model
        regressor.fit(X_train, Y_train)
        Y_pred = regressor.predict(X_test)
    
    r2 = r2_score(Y_test, Y_pred)
    r2_scores[name] = r2
    print(f"Model: {name} | R^2: {r2:.4f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = regressor
        best_name = name

print("\nAll Models Performance:")
for name, score in r2_scores.items():
    print(f"Model: {name} | R^2: {score:.4f}")

print(f"\nBest Model: {best_name} with R^2 = {best_r2:.4f}")

#-------------------------------------------------------
# Plot results
plt.figure(figsize=(8, 6))
bars = plt.bar(r2_scores.keys(), r2_scores.values(), color=['blue', 'green', 'red'])
plt.xlabel("Model")
plt.ylabel("R^2 Score")
plt.title("Model Performance Comparison")
plt.ylim(0, 1)

# Add text labels to bars
for bar, score in zip(bars, r2_scores.values()):
    plt.text(bar.get_x() + bar.get_width()/2, 0.5*(bar.get_height()), f"{score:.4f}", ha='center', fontsize=12)

plt.show()

