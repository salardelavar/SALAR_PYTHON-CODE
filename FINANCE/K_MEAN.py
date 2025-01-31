import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate 6,000 random numbers for x and y
np.random.seed(42)
x = np.random.beta(1, 2, 6000).reshape(-1, 1)  # 2D array for sklearn
y = np.random.beta(1, 1, 6000)                # 1D target variable

def CLUSTER_DATA(x, y, XLABEL, YLABEL, MAX_CLUSTERS):
    # Combine x and y into a single dataset and scale
    data = np.column_stack((x, y))
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    best_r2 = -np.inf
    best_k = 0
    best_clusters = None
    global_mean = np.mean(y)  # For handling small clusters

    for k in range(1, MAX_CLUSTERS+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        total_ss_res = 0.0
        for i in range(k):
            cluster_mask = (clusters == i)
            x_cluster = x[cluster_mask].reshape(-1, 1)  # Ensure 2D shape
            y_cluster = y[cluster_mask]
            
            if len(x_cluster) < 2:
                # Handle small clusters with global mean
                total_ss_res += np.sum((y_cluster - global_mean) ** 2)
                continue
                
            model = LinearRegression()
            model.fit(x_cluster, y_cluster)
            y_pred = model.predict(x_cluster)
            total_ss_res += np.sum((y_cluster - y_pred) ** 2)

        ss_tot = np.sum((y - global_mean) ** 2)
        r2 = 1 - (total_ss_res / ss_tot)
        
        if r2 > best_r2:
            best_r2 = r2
            best_k = k
            best_clusters = clusters

    print(f"Optimal clusters: {best_k}, R²: {best_r2:.4f}")

    # Plot clustered data
    plt.figure(figsize=(10, 6))
    for i in range(best_k):
        cluster_mask = (best_clusters == i)
        plt.scatter(x[cluster_mask], y[cluster_mask], alpha=0.5, label=f'Cluster {i+1}')
    plt.title(f"{XLABEL} vs {YLABEL} - {best_k} Clusters (R² = {best_r2:.4f})")
    plt.xlabel(XLABEL)
    plt.ylabel(YLABEL)
    plt.legend()
    plt.grid(True)
    plt.show()

#-------------------------------------------------
# Plot original data first
plt.figure(figsize=(10, 6))
plt.scatter(x, y, c='blue', alpha=0.5)
plt.title("Original Data Distribution")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()

# Run clustering analysis
CLUSTER_DATA(x, y, "X", "Y", MAX_CLUSTERS=3)