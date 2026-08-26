###########################################################################################################
#                   >> IN THE NAME OF ALLAH, THE MOST GRACIOUS, THE MOST MERCIFUL <<                      #
#         PARAMETER IDENTIFICATION OF A DAMPED SDOF SYSTEM WITH HARMONIC LOAD VIA EXPERIMENTAL DATA       #
#                                         P(t) = P0 * sin(ω*t)                                            #
#---------------------------------------------------------------------------------------------------------#
#                    THIS PYTHON SCRIPT IS WRITTEN BY SALAR DELAVAR GHASHGHAEI (QASHQAI)                  #
#                                   EMAIL: salar.d.ghashghaei@gmail.com                                   #
###########################################################################################################

"""
1. Defines the analytical displacement function for a damped SDOF system under harmonic load P(t)=P0 sin(w t), combining the decaying transient (homogeneous) and steady‑state (particular) solutions.  
2. Generates synthetic experimental displacement data at 7 selected time points using true physical parameters (xi, k, m, x_0, v_0, P_0, omega) with added noise to mimic real measurements.  
3. Constructs a system of 7 nonlinear equations by computing the residual error between the analytical response and the synthetic data at each time point.  
4. Applies the Newton–Raphson method via `scipy.optimize.root` (with the hybrid Powell algorithm) to iteratively solve the system and recover all 7 unknown parameters from an initial guess.  
5. Outputs the estimated parameters, calculates the Root Mean Square Error (RMSE) to quantify the fitting accuracy, and prints the results.  
6. Plots the smooth fitted analytical response over a fine time grid, overlaying the synthetic experimental points and the fitted values at the measurement times for visual validation.

Parameter Identification of a Damped Forced Vibration System
with Harmonic Excitation p(t) = p0 * sin(ωt)
Using the Newton–Raphson Method (via scipy.optimize.root)

The system equation:  m*u'' + c*u' + k*u = P0 * sin(ω*t)

The analytical displacement is the sum of the homogeneous solution
(decaying transient) and the particular solution (steady‑state).


Parameter Identification for m*u'' + c*u' + k*u = P0 * sin(ω*t)
Unknowns: ξ, k, m, x0, v0, P0, ω   (7 unknowns)
We use 7 experimental data points and Newton–Raphson (scipy.optimize.root) to set up
7 nonlinear equations.
"""
# BOOK: Dynamics of Structures in SI Units -  Anil Kumar Chopra
'https://share.google/TDN5O4eWmw5pH8zUH'
# BOOK: Differential Equations for Engineers-Wei-Chau Xie-CAMBRIDGE-2010


import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

#%% ----------------------------------------------------------------------
# 1. Analytical displacement function

def displacement(t, xi, k, m, x0, v0, P0, omega):
    """
    Returns u(t) for the forced damped system.
    Assumes underdamped (xi < 1).
    """
    wn = np.sqrt(k / m)                # natural frequency
    wd = wn * np.sqrt(1 - xi**2)       # damped frequency
    r = omega / wn                     # frequency ratio

    # Steady-state amplitude
    X = (P0 / k) / np.sqrt((1 - r**2)**2 + (2 * xi * r)**2)
    # Phase lag (using atan2 for correct quadrant)
    phi = np.arctan2(2 * xi * r, 1 - r**2)

    # Particular solution and its initial values
    xp = X * np.sin(omega * t - phi)
    xp0 = X * np.sin(-phi)              # x_p(0)
    xp_dot0 = X * omega * np.cos(-phi)  # x_p'(0)

    # Homogeneous constants from initial conditions
    A = x0 - xp0
    B = (v0 - xp_dot0 + xi * wn * A) / wd

    # Homogeneous (transient) solution
    xh = np.exp(-xi * wn * t) * (A * np.cos(wd * t) + B * np.sin(wd * t))

    return xh + xp

#%% ----------------------------------------------------------------------
# 2. Generate synthetic "experimental" data (true parameters + noise)

# True values (what we want to recover)
xi_true = 0.05
k_true = 200.0          # N/m
m_true = 10.0           # kg
x0_true = 0.01          # m
v0_true = 0.0           # m/s
P0_true = 100.0         # N
omega_true = 3.0         # rad/s

# 7 time points for 7 unknowns
t_exp = np.array([0.1, 0.6, 1.2, 2.0, 3.0, 4.5, 6.0])

# True displacements (add a tiny amount of noise)
np.random.seed(42)
x_exp = displacement(t_exp, xi_true, k_true, m_true, x0_true, v0_true, P0_true, omega_true)
x_exp += 1e-5 * np.random.randn(len(t_exp))   # measurement noise

#%% ----------------------------------------------------------------------
# 3. Nonlinear equations: residuals

def equations(U):
    xi, k, m, x0, v0, P0, omega = U
    return displacement(t_exp, xi, k, m, x0, v0, P0, omega) - x_exp

#%% ----------------------------------------------------------------------
# 4. Initial guess (must be reasonably close for Newton to converge)

guess = [0.04, 180.0, 11.0, 0.008, 0.1, 90.0, 2.8]

#%% ----------------------------------------------------------------------
# 5. Solve using Newton–Raphson (hybrid method)

sol = root(equations, guess, method='hybr')
print("Converged =", sol.success)
print()

# Extract estimated parameters
xi_est, k_est, m_est, x0_est, v0_est, P0_est, omega_est = sol.x

print("Estimated Parameters (7 unknowns)")
print("-----------------------------------")
print(f"Damping ratio (ξ)        = {xi_est:.6f}")
print(f"Stiffness (k)             = {k_est:.6f} N/m")
print(f"Mass (m)                  = {m_est:.6f} kg")
print(f"Initial displacement (x0) = {x0_est:.6f} m")
print(f"Initial velocity (v0)     = {v0_est:.6f} m/s")
print(f"Force amplitude (P0)      = {P0_est:.6f} N")
print(f"Excitation frequency (ω)  = {omega_est:.6f} rad/s")

#%% ----------------------------------------------------------------------
# 6. Compute fitted response and RMSE

t_plot = np.linspace(0, max(t_exp), 500)
x_plot = displacement(t_plot, xi_est, k_est, m_est, x0_est, v0_est, P0_est, omega_est)
x_fit = displacement(t_exp, xi_est, k_est, m_est, x0_est, v0_est, P0_est, omega_est)

rmse = np.sqrt(np.mean((x_fit - x_exp)**2))
print(f"\nRMSE = {rmse:.6e}")

#%% ----------------------------------------------------------------------
# 7. Plot results

plt.figure(figsize=(8,5))
plt.plot(t_plot, x_plot, linewidth=2, label='Fitted analytical response')
plt.scatter(t_exp, x_exp, color='red', s=70, zorder=5, label='Experimental data (synthetic)')
plt.plot(t_exp, x_fit, 'ko', markersize=6, label='Fitted points')
plt.xlabel("Time [sec]", fontsize=12)
plt.ylabel("Displacement [m]", fontsize=12)
plt.title("7‑Parameter Identification (p(t)=P₀ sin(ωt))", fontsize=13)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
