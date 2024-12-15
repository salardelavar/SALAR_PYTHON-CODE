###########################################################################################################
#                                                 IN THE NAME OF ALLAH                                    #
#     RECTANGULAR FOUNDATION INTERACTION WITH SPRING-SUPPORTED SOIL WITH FINITE DIFFERENCE METHOD (FDM)   #
#---------------------------------------------------------------------------------------------------------#
# This Python code models the interaction between a rectangular foundation and soil using a finite        #
# difference method to solve for the deflections and internal forces. The soil is represented as springs  #
# distributed across the foundation area.                                                                 #
#---------------------------------------------------------------------------------------------------------#
#                               THIS PROGRAM IS WRITTEN BY SALAR DELAVAR GHASHGHAEI (QASHQAIA)            #
#                                          EMAIL: SALAR.D.GHASHGHAEI@GMAIL.COM                            #
###########################################################################################################

import numpy as np
import matplotlib.pyplot as plt

# Define constants
E = 2.1e11       # Elastic modulus of the foundation material (Pa)
H = 0.5          # Height (thickness) of the foundation (m)
B = 5.0          # Width of the foundation (m)
L = 10.0         # Length of the foundation (m)
Es = 1e8         # Modulus of subgrade (soil) reaction (Pa/m)
P = -1e6         # Compression force applied at the center of the foundation (N)

# Derived properties
I = B * H**3 / 12  # Moment of inertia for the rectangular cross-section (m^4)

# Discretization
nx = 50  # Number of grid points along length
ny = 25  # Number of grid points along width
dx = L / (nx - 1)
dy = B / (ny - 1)

# Define finite difference matrices
nodes = nx * ny  # Total number of nodes
A = np.zeros((nodes, nodes))  # Coefficient matrix
b = np.zeros(nodes)          # Right-hand side vector

# Helper functions for indexing in 2D
index = lambda i, j: i * ny + j

# Populate finite difference matrix
for i in range(1, nx - 1):
    for j in range(1, ny - 1):
        idx = index(i, j)

        # Contributions from bending in x-direction
        A[idx, index(i - 1, j)] += E * I / dx**4
        A[idx, index(i + 1, j)] += E * I / dx**4
        A[idx, idx] -= 2 * E * I / dx**4

        # Contributions from bending in y-direction
        A[idx, index(i, j - 1)] += E * I / dy**4
        A[idx, index(i, j + 1)] += E * I / dy**4
        A[idx, idx] -= 2 * E * I / dy**4

        # Soil spring contribution
        A[idx, idx] += Es

# Apply compression force at the center of the foundation
center_x, center_y = nx // 2, ny // 2
b[index(center_x, center_y)] -= P / (dx * dy)  # Convert force to pressure

# Boundary conditions (fixed edges: zero displacement)
for i in range(nx):
    for j in [0, ny - 1]:  # Fixed along y-edges
        A[index(i, j), index(i, j)] = 1
        b[index(i, j)] = 0
for j in range(ny):
    for i in [0, nx - 1]:  # Fixed along x-edges
        A[index(i, j), index(i, j)] = 1
        b[index(i, j)] = 0

# Solve for displacements
y = np.linalg.solve(A, b).reshape((nx, ny))

# Calculate reactions (subgrade reaction)
reactions = Es * y

# Plot results
x = np.linspace(0, L, nx)
y_coords = np.linspace(0, B, ny)
X, Y = np.meshgrid(x, y_coords)

plt.figure(figsize=(18, 14))

# Plot displacement
plt.subplot(2, 2, 1)
plt.contourf(X, Y, y.T, levels=50, cmap="viridis")
plt.colorbar(label="Displacement (m)")
plt.title("Foundation Displacement")
plt.xlabel("Length (m)")
plt.ylabel("Width (m)")

# Plot soil reactions
plt.subplot(2, 2, 2)
plt.contourf(X, Y, reactions.T, levels=50, cmap="plasma")
plt.colorbar(label="Reaction Force (N/m^2)")
plt.title("Soil Reaction Forces")
plt.xlabel("Length (m)")
plt.ylabel("Width (m)")

plt.tight_layout()
plt.show()
