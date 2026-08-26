###########################################################################################################
#                   >> IN THE NAME OF ALLAH, THE MOST GRACIOUS, THE MOST MERCIFUL <<                      #
#  PARAMETER IDENTIFICATION OF A DAMPED SDOF SYSTEM WITH EXPONENTIALLY DECAYING HARMONIC EXCITATION       #                #
#                                    P(t) = P0 * exp(-A*ω*t) * sin(ω*t)                                   #
#---------------------------------------------------------------------------------------------------------#
#                    THIS PYTHON SCRIPT IS WRITTEN BY SALAR DELAVAR GHASHGHAEI (QASHQAI)                  #
#                                   EMAIL: salar.d.ghashghaei@gmail.com                                   #
###########################################################################################################
"""
1. Defines the closed‑form analytical displacement u(t) for a damped SDOF system subjected to P(t)=P0 e^(-A\omega t) sin(omega t)), combining the transient homogeneous solution with a particular solution derived from the decaying force envelope.  
2. Generates synthetic “experimental” data by evaluating this function at 8 selected time points using true parameters (xi,k,m,x_0,v_0,P_0,omega,A), then adds a tiny noise to mimic real measurements.  
3. Constructs a square system of 8 nonlinear residual equations, each representing the difference between the analytical displacement and the synthetic data at the corresponding time instant.  
4. Employs the Newton–Raphson method (via `scipy.optimize.root` with the hybrid Powell algorithm) to iteratively solve this system, starting from a physically plausible initial guess.  
5. Prints the recovered parameters, calculates the Root Mean Square Error (RMSE) to quantify the fit, and displays all results in the console.  
6. Plots the smooth fitted response over a fine time grid, overlaying the synthetic experimental points and the fitted values at the measurement times for visual comparison.
Parameter Identification for m*u'' + c*u' + k*u = P0 * exp(-A*ω*t) * sin(ω*t)
Unknowns: ξ, k, m, x0, v0, P0, ω, A   (8 unknowns)
We use 8 experimental data points and Newton–Raphson (scipy.optimize.root) to set up
8 nonlinear equations. 
"""
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

#%% ----------------------------------------------------------------------
# 1. Analytical displacement function (closed‑form)

def displacement(t, xi, k, m, x0, v0, P0, omega, A):
    """
    Returns u(t) for the forced damped system with P(t)=P0*exp(-A*ω*t)*sin(ω*t).
    Assumes underdamped (xi < 1).
    """
    wn = np.sqrt(k / m)                 # natural frequency
    wd = wn * np.sqrt(1 - xi**2)        # damped frequency
    lam = A * omega                     # decay rate of force envelope
    p0 = P0 / m                         # force per unit mass

    # Coefficients for particular solution
    A1 = wn**2 - omega**2 + lam**2 - 2 * xi * wn * lam
    B1 = 2 * omega * (xi * wn - lam)
    denom = A1**2 + B1**2
    C = -p0 * B1 / denom
    D = p0 * A1 / denom

    # Particular solution at t=0 and its derivative
    xp0 = C                     # u_p(0)
    xp_dot0 = -lam * C + omega * D   # u_p'(0)

    # Homogeneous constants from initial conditions
    A_h = x0 - xp0
    B_h = (v0 - xp_dot0 + xi * wn * A_h) / wd

    # Homogeneous (transient) solution
    xh = np.exp(-xi * wn * t) * (A_h * np.cos(wd * t) + B_h * np.sin(wd * t))

    # Particular solution
    xp = np.exp(-lam * t) * (C * np.cos(omega * t) + D * np.sin(omega * t))

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
A_true = 0.3            # decay exponent of force envelope

# 8 time points for 8 unknowns
t_exp = np.array([0.1, 0.5, 1.0, 1.8, 2.8, 4.0, 5.5, 7.0])

# True displacements (add a tiny amount of noise)
np.random.seed(42)
x_exp = displacement(t_exp, xi_true, k_true, m_true, x0_true, v0_true, P0_true, omega_true, A_true)
x_exp += 1e-5 * np.random.randn(len(t_exp))   # measurement noise

#%% ----------------------------------------------------------------------
# 3. Nonlinear equations: residuals

def equations(U):
    xi, k, m, x0, v0, P0, omega, A = U
    return displacement(t_exp, xi, k, m, x0, v0, P0, omega, A) - x_exp

#%% ----------------------------------------------------------------------
# 4. Initial guess (must be reasonably close for Newton to converge)

guess = [0.04, 180.0, 11.0, 0.008, 0.1, 90.0, 2.8, 0.25]

#%% ----------------------------------------------------------------------
# 5. Solve using Newton–Raphson (hybrid method)

sol = root(equations, guess, method='hybr')
print("Converged =", sol.success)
print()

# Extract estimated parameters
xi_est, k_est, m_est, x0_est, v0_est, P0_est, omega_est, A_est = sol.x

print("Estimated Parameters (8 unknowns)")
print("-----------------------------------")
print(f"Damping ratio (ξ)        = {xi_est:.6f}")
print(f"Stiffness (k)             = {k_est:.6f} N/m")
print(f"Mass (m)                  = {m_est:.6f} kg")
print(f"Initial displacement (x0) = {x0_est:.6f} m")
print(f"Initial velocity (v0)     = {v0_est:.6f} m/s")
print(f"Force amplitude (P0)      = {P0_est:.6f} N")
print(f"Excitation frequency (ω)  = {omega_est:.6f} rad/s")
print(f"Force decay exponent (A)  = {A_est:.6f}")

#%% ----------------------------------------------------------------------
# 6. Compute fitted response and RMSE

t_plot = np.linspace(0, max(t_exp), 500)
x_plot = displacement(t_plot, xi_est, k_est, m_est, x0_est, v0_est, P0_est, omega_est, A_est)
x_fit = displacement(t_exp, xi_est, k_est, m_est, x0_est, v0_est, P0_est, omega_est, A_est)

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
plt.title("8‑Parameter Identification (p(t)=P₀ e^{-Aωt} sin(ωt))", fontsize=13)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()