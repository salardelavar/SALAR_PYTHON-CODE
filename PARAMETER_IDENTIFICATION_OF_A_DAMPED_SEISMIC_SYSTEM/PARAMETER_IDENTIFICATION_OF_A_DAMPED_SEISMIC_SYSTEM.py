###########################################################################################################
#                   >> IN THE NAME OF ALLAH, THE MOST GRACIOUS, THE MOST MERCIFUL <<                      #
#         IDENTIFICATION OF 7 PARAMETERS USING 7 ARBITRARY POINTS FROM EARTHQUAKE DISPLACEMENT DATA       #
#                   U(t) = a1 + a2 * exp(-a3*a4*t) * [ a6*sin(a4*t) + a7*sin(a5*t) ]                      #
#---------------------------------------------------------------------------------------------------------#
#                    THIS PYTHON SCRIPT IS WRITTEN BY SALAR DELAVAR GHASHGHAEI (QASHQAI)                  #
#                                   EMAIL: salar.d.ghashghaei@gmail.com                                   #
###########################################################################################################
"""
Parameter identification using 7 arbitrary time points from earthquake displacement history.

Model:  U(t) = a1 + a2 * exp(-a3*a4*t) * ( a6*sin(a4*t) + a7*sin(a5*t) )

Unknowns: a1, a2, a3, a4, a5, a6, a7   (7 unknowns)
Given: 7 data points (t_i, U_i) chosen from the recorded history.
Solver: Newton–Raphson (via scipy.optimize.root)
"""
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. Analytical function (model) with 7 parameters
def displacement_func(t, a1, a2, a3, a4, a5, a6, a7):
    """
    Compute U(t) = a1 + a2 * exp(-a3*a4*t) * ( a6*sin(a4*t) + a7*sin(a5*t) )
    """
    decay = np.exp(-a3 * a4 * t)
    osc = a6 * np.sin(a4 * t) + a7 * np.sin(a5 * t)
    return a1 + a2 * decay * osc

# ----------------------------------------------------------------------
# 2. Generate synthetic "experimental" data (simulate an earthquake record)
#    Replace this section with your actual recorded displacement time history.

# True parameters (used only for generating synthetic data)
a1_true = 0.005
a2_true = 0.12
a3_true = 0.04
a4_true = 2.5
a5_true = 4.0
a6_true = 1.0
a7_true = 0.3

# Time vector for the full record (30 seconds, step 0.01)
dt = 0.01
t_full = np.arange(0, 30.0, dt)

# True displacement (noiseless)
u_true_full = displacement_func(t_full, a1_true, a2_true, a3_true, a4_true,
                                a5_true, a6_true, a7_true)

# Add small noise to mimic real measurements
np.random.seed(42)   # for reproducibility
noise = 1e-4 * np.random.randn(len(t_full))
u_exp_full = u_true_full + noise

# ----------------------------------------------------------------------
# 3. Choose 7 arbitrary time points from the history
#    In practice, these are the times at which you have measured displacements.
#    Here we select them manually (e.g., at specific instants).

t_exp = np.array([0.5, 2.0, 4.0, 7.0, 10.0, 15.0, 22.0])   # seconds

# Get the corresponding displacement values from the recorded data
# (using interpolation to get values at exact times if needed)
from scipy.interpolate import interp1d
interp_func = interp1d(t_full, u_exp_full, kind='linear', fill_value='extrapolate')
x_exp = interp_func(t_exp)

print("Selected data points (time, displacement):")
for i in range(len(t_exp)):
    print(f"  t={t_exp[i]:.3f} sec, U={x_exp[i]:.6f} m")

# ----------------------------------------------------------------------
# 4. Nonlinear equations: residuals (7 equations for 7 unknowns)
def equations(U):
    a1, a2, a3, a4, a5, a6, a7 = U
    return displacement_func(t_exp, a1, a2, a3, a4, a5, a6, a7) - x_exp

# ----------------------------------------------------------------------
# 5. Initial guess (should be reasonably close to the true solution)
#    Adjust these values based on your physical understanding.
a1_guess = 0.0
a2_guess = 0.1
a3_guess = 0.03
a4_guess = 2.5
a5_guess = 4.0
a6_guess = 1.0
a7_guess = 0.3
guess = [a1_guess, a2_guess, a3_guess, a4_guess, a5_guess, a6_guess, a7_guess]

# ----------------------------------------------------------------------
# 6. Solve the system using Newton–Raphson (hybrid method)
sol = root(equations, guess, method='hybr')
print("\nConverged =", sol.success)
print()

a1_est, a2_est, a3_est, a4_est, a5_est, a6_est, a7_est = sol.x

print("Estimated Parameters (7 unknowns)")
print("-----------------------------------")
print(f"a1 (offset)               = {a1_est:.6f}")
print(f"a2 (amplitude scale)      = {a2_est:.6f}")
print(f"a3 (damping coefficient)  = {a3_est:.6f}")
print(f"a4 (frequency 1)          = {a4_est:.6f} rad/s")
print(f"a5 (frequency 2)          = {a5_est:.6f} rad/s")
print(f"a6 (sine coefficient 1)   = {a6_est:.6f}")
print(f"a7 (sine coefficient 2)   = {a7_est:.6f}")

# ----------------------------------------------------------------------
# 7. Compute fitted response and RMSE
t_plot = np.linspace(0, max(t_full), 1000)
u_fit = displacement_func(t_plot, a1_est, a2_est, a3_est, a4_est, a5_est, a6_est, a7_est)

u_fit_exp = displacement_func(t_exp, a1_est, a2_est, a3_est, a4_est, a5_est, a6_est, a7_est)
rmse = np.sqrt(np.mean((u_fit_exp - x_exp)**2))
print(f"\nRMSE (at the selected points) = {rmse:.6e}")

# ----------------------------------------------------------------------
# 8. Plot results
plt.figure(figsize=(12, 7))

# Top subplot: full displacement history and selected points
plt.subplot(2, 1, 1)
plt.plot(t_full, u_exp_full, 'b-', linewidth=1.0, label='Experimental data (full history)')
plt.plot(t_full, u_true_full, 'g--', linewidth=1.5, label='True response (noiseless)')
plt.scatter(t_exp, x_exp, color='red', s=80, zorder=5, label='Selected points (7 points)')
plt.xlabel("Time [sec]")
plt.ylabel("Displacement [m]")
plt.title("Displacement history and selected data points")
plt.grid(True)
plt.legend()

# Bottom subplot: comparison of true and fitted responses
plt.subplot(2, 1, 2)
plt.plot(t_full, u_true_full, 'b-', linewidth=1.5, label='True response')
plt.plot(t_plot, u_fit, 'r--', linewidth=2.0, label='Fitted response (estimated)')
plt.scatter(t_exp, x_exp, color='red', s=80, zorder=5, label='Selected points')
plt.xlabel("Time [sec]")
plt.ylabel("Displacement [m]")
plt.title("Comparison of true and fitted responses (fitted to selected points)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()