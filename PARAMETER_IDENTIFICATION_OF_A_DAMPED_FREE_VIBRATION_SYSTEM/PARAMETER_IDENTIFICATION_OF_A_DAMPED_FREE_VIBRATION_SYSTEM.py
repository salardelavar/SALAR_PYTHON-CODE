###########################################################################################################
#                   >> IN THE NAME OF ALLAH, THE MOST GRACIOUS, THE MOST MERCIFUL <<                      #
#                      PARAMETER IDENTIFICATION OF A DAMPED FREE-VIBRATION SYSTEM                         #
#---------------------------------------------------------------------------------------------------------#
#                    THIS PYTHON SCRIPT IS WRITTEN BY SALAR DELAVAR GHASHGHAEI (QASHQAI)                  #
#                                   EMAIL: salar.d.ghashghaei@gmail.com                                   #
###########################################################################################################
"""
Parameter Identification of a Damped Free Vibration System
Using the Newton–Raphson Method in Python

1. The program starts with an initial guess for the four unknown
   parameters: damping ratio (xi), natural period (T),
   initial displacement (x0), and initial velocity (v0).

2. The DISPLACEMENT() function computes the theoretical free
   vibration displacement at any time.

3. Experimental displacement values are provided at several
   known time points.

4. The EQUATIONS() function computes the error between the
   analytical and experimental displacements.

5. The root() function applies the Newton–Raphson iterative
   method to minimize these errors.

6. During each iteration, the four unknown parameters are
   updated until convergence is achieved.

7. After convergence, the estimated values of xi, T, x0,
   and v0 are obtained.

8. The analytical response is evaluated over a fine time
   grid to produce a smooth displacement curve.

9. The Root Mean Square Error (RMSE) is computed to evaluate
   the accuracy of the fitted model.

10. Finally, the analytical response and experimental data
    are plotted together for visual comparison.
    
THIS PYTHON SCRIPT IS WRITTEN BY SALAR DELAVAR GHASHGHAEI (QASHQAI)    
"""
# BOOK: Dynamics of Structures in SI Units -  Anil Kumar Chopra
'https://share.google/TDN5O4eWmw5pH8zUH'
# BOOK: Differential Equations for Engineers-Wei-Chau Xie-CAMBRIDGE-2010
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

#%%----------------------------
# Experimental data
t = np.array([0.10, 5.35, 10.60, 40.90]) # TIME
dt = 0.01                                # TIME INCREMENT
x_exp = np.array([                       # [m] DISPLACEMENT
    0.0152724,
    -0.00277418,
   0.000206691,
   0.00000107
])

#%%----------------------------
# Displacement function
def DISPLACEMENT(t, xi, T, x0, v0):

    wn = 2*np.pi/T
    wd = wn*np.sqrt(1-xi**2)

    A = x0

    B = (v0 + xi*wn*x0)/wd

    return np.exp(-xi*wn*t) * (
            A*np.cos(wd*t)
            + B*np.sin(wd*t)
            )

#%%----------------------------
# Nonlinear equations
def EQUATIONS(U):

    xi,T,x0,v0 = U

    return np.array([

        DISPLACEMENT(t[0],xi,T,x0,v0)-x_exp[0],

        DISPLACEMENT(t[1],xi,T,x0,v0)-x_exp[1],

        DISPLACEMENT(t[2],xi,T,x0,v0)-x_exp[2],

        DISPLACEMENT(t[3],xi,T,x0,v0)-x_exp[3]

    ])

#%%----------------------------
# Initial guess
guess = [0.03,0.80,0.02,0.0]

#%%----------------------------
# Newton-Raphson
sol = root(EQUATIONS, guess, method='hybr')
print("Converged =",sol.success)
print()
print("Damping ratio =",sol.x[0])
print("Period =",sol.x[1])
print("Initial displacement =",sol.x[2])
print("Initial velocity =",sol.x[3])


#%%----------------------------
# Estimated parameters
xi, T, x0, v0 = sol.x

print("\nEstimated Parameters")
print("-------------------------")
print(f"Damping ratio (ξ)        = {xi:.6f}")
print(f"Natural period (T)       = {T:.6f} sec")
print(f"Initial displacement x0  = {x0:.6f} m")
print(f"Initial velocity v0      = {v0:.6f} m/s")

#%%----------------------------
# Smooth analytical response
NUM = int(max(t)/dt)
t_plot = np.linspace(0, max(t), NUM)

x_plot = DISPLACEMENT(t_plot, xi, T, x0, v0)

#%%----------------------------
# Response at experimental points
x_fit = DISPLACEMENT(t, xi, T, x0, v0)

#%%----------------------------
# Root Mean Square Error
rmse = np.sqrt(np.mean((x_fit - x_exp)**2))

print(f"RMSE = {rmse:.6e}")

#%%----------------------------
# Plot
plt.figure(figsize=(8,5))

plt.plot(t_plot,
         x_plot,
         linewidth=2,
         label='Estimated analytical response')

plt.scatter(t,
            x_exp,
            color='red',
            s=70,
            zorder=5,
            label='Experimental data')

plt.plot(t,
         x_fit,
         'ko',
         markersize=6,
         label='Fitted points')

plt.xlabel("Time [sec]", fontsize=12)
plt.ylabel("Displacement [m]", fontsize=12)
plt.title("Free Vibration Parameter Identification", fontsize=13)

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.show()