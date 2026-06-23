###########################################################################################################
#                   >> IN THE NAME OF ALLAH, THE MOST GRACIOUS, THE MOST MERCIFUL <<                      #
#                     PUSHOVER ANALYSIS OF STRUCTURE WITH FORCE ANALOGY METHOD                            #
#---------------------------------------------------------------------------------------------------------#
#                    THIS PYTHON SCRIPT WRITTEN BY SALAR DELAVAR GHASHGHAEI (QASHQAI)                     #
#                                   EMAIL: salar.d.ghashghaei@gmail.com                                   #
###########################################################################################################
"""
This python script performs a displacement-controlled pushover analysis of a fixed-guided column 
using the Force Analogy Method (FAM). 
1. The global stiffness of the structure remains constant at its initial elastic state (K_elastic).
2. Material nonlinearity (hinge formation) is captured by solving for plastic rotations (theta_p)
   at the base and top, which act as analog strain-like parameters reducing internal forces.
3. The moment and shear equations are decomposed into elastic and plastic components.
4. A return-mapping algorithm tracks the onset of yielding and computes the plastic rotations.
5. Three subplots are generated: Pushover Curve, Moment Evolution, and Plastic Rotation Evolution.
"""
import numpy as np
import matplotlib.pyplot as plt
# YOUTUBE: Pushover Analysis Force Analogy Method with Force Control Based on Timoshenko Beam Theory in C
'https://www.youtube.com/watch?v=u9M6klJO6uU&t=58s'
# BOOK: Theory of Nonlinear Structural Analysis: The Force Analogy Method for Earthquake Engineering
# Gang Li, Kevin K. F. Wong (2014) - Wiley
'https://onlinelibrary.wiley.com/doi/book/10.1002/9781118718070'
# EXAMPLE: PAGE 123-129

import numpy as np
import matplotlib.pyplot as plt

#%% 1. Parameter Definitions (L, EI, and X)
L = 3.0                           # [m] Column length
E = 3.0e7                         # [kPa] Concrete Elastic Modulus
G = 1.15e7                        # [kPa] Concrete Shear Modulus
b, h = 0.5, 0.5                   # [m] Section dimensions

A = b * h                         # [m^2] Cross-sectional area
I_b = (1.0 / 12.0) * b * h**3     # [m^4] Bending moment of inertia
A_s = A / 1.2                     # [m^2] Effective shear area

# Key analytical variables requested:
EI = E * I_b                       # Flexural Rigidity (kN-m^2) = 156250.0
X = (12.0 * EI) / (G * A_s * L**2) # Shear parameter chi (often written as 'X') = 0.0869565

# Key structural capacities (Tables 4.1 & 4.2)
m_cr, m_y = 91.95, 229.5          # Cracking and yielding moments (kN-m)
V_cr, V_y = 61.3, 153.0           # Cracking and yielding shears (kN)
tau_u = 0.0075                    # Shear ultimate deformation limit (m)

# Elastic rotational and shear stiffness parameters
k_f = 2.9e5                       # Rotational stiffness (kN-m/rad)
k_s = 7.7e5                       # Shear stiffness (kN/m)

#%% 2. Analytical Stiffness Matrix Assembly (Eq. 4.34)
K_val = (12.0 * EI) / ((1.0 + X) * L**3)

Kp_1 = -(6.0 * EI) / ((1.0 + X) * L**2)
Kp_2 = -(6.0 * EI) / ((1.0 + X) * L**2)
Kp_3 = (12.0 * EI) / ((1.0 + X) * L**3)

Kpp_11 = ((4.0 + X) / (1.0 + X)) * (EI / L)
Kpp_12 = ((2.0 - X) / (1.0 + X)) * (EI / L)
Kpp_13 = -(6.0 * EI) / ((1.0 + X) * L**2)

# Full 4x4 Elastic Stiffness Matrix
K_elastic = np.array([
    [K_val,  Kp_1,   Kp_2,   Kp_3],
    [Kp_1,   Kpp_11, Kpp_12, Kpp_13],
    [Kp_2,   Kpp_12, Kpp_11, Kpp_13],
    [Kp_3,   Kpp_13, Kpp_13, K_val]
])

# Submatrices for the inelastic degrees of freedom [q] = [-theta1", -theta2", -tau"]^T
K_sub = np.array([
    [Kpp_11, Kpp_12, Kpp_13],
    [Kpp_12, Kpp_11, Kpp_13],
    [Kpp_13, Kpp_13, K_val ]
])
K_coupling = np.array([Kp_1, Kp_2, Kp_3])

#%% 3. Dynamic Trial-and-Error Structural Solver
def solve_system(x):
    """
    Solves the nonlinear response of the SDOF column for a given input 
    displacement 'x' (meters) by dynamically checking capacities.
    """
    # --- TRIAL 1: Assume Elastic State ---
    u_elastic = np.array([x, 0.0, 0.0, 0.0])
    forces = K_elastic @ u_elastic
    F, m1, m2, V = forces[0], forces[1], forces[2], forces[3]
    
    if abs(m1) <= m_cr and abs(V) <= V_cr:
        return F, m1, m2, V, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Elastic Stage"
        
    # --- TRIAL 2: Assume Hardening Stage (Example 4.3) ---
    w_0_hard = np.array([-m_cr, -m_cr, V_cr])
    alpha_hard = np.diag([8.6e4, 8.6e4, 4.0e4])
    
    # Solve: (K_sub + alpha) * q = w_0 - K_coupling * x
    q = np.linalg.solve(K_sub + alpha_hard, w_0_hard - K_coupling * x)
    theta1_p, theta2_p, tau_p = -q[0], -q[1], -q[2]
    
    m1_trial = -m_cr + 8.6e4 * theta1_p
    m2_trial = -m_cr + 8.6e4 * theta2_p
    V_trial  = V_cr + 4.0e4 * tau_p
    F_trial  = V_trial
    
    if abs(m1_trial) <= m_y and abs(V_trial) <= V_y:
        theta1 = theta1_p + m1_trial / k_f
        theta2 = theta2_p + m2_trial / k_f
        tau    = tau_p + V_trial / k_s
        return F_trial, m1_trial, m2_trial, V_trial, theta1_p, theta2_p, tau_p, theta1, theta2, tau, "Hardening Stage"
        
    # --- TRIAL 3: Assume Yielding Platform (Example 4.4) ---
    w_0_yield = np.array([-m_y, -m_y, V_y])
    alpha_yield = np.diag([0.1, 0.1, 0.1]) # Trace values near zero
    
    q = np.linalg.solve(K_sub + alpha_yield, w_0_yield - K_coupling * x)
    theta1_p, theta2_p, tau_p = -q[0], -q[1], -q[2]
    
    m1_trial = -m_y + 0.1 * theta1_p
    m2_trial = -m_y + 0.1 * theta2_p
    V_trial  = V_y + 0.1 * tau_p
    F_trial  = V_trial
    
    tau_total = tau_p + V_trial / k_s
    
    if abs(tau_total) <= tau_u:
        theta1 = theta1_p + m1_trial / k_f
        theta2 = theta2_p + m2_trial / k_f
        return F_trial, m1_trial, m2_trial, V_trial, theta1_p, theta2_p, tau_p, theta1, theta2, tau_total, "Yielding Platform"
        
    # --- TRIAL 4: Softening Stage (Example 4.5) ---
    w_0_soft = np.array([-407.02, -407.02, 390.67])
    alpha_soft = np.diag([-1.1e4, -1.1e4, -3.3e4])
    
    q = np.linalg.solve(K_sub + alpha_soft, w_0_soft - K_coupling * x)
    theta1_p, theta2_p, tau_p = -q[0], -q[1], -q[2]
    
    m1_final = -407.02 - 1.1e4 * theta1_p
    m2_final = -407.02 - 1.1e4 * theta2_p
    V_final  = 390.67 - 3.3e4 * tau_p
    F_final  = V_final
    
    theta1 = theta1_p + m1_final / k_f
    theta2 = theta2_p + m2_final / k_f
    tau    = tau_p + V_final / k_s
    
    return F_final, m1_final, m2_final, V_final, theta1_p, theta2_p, tau_p, theta1, theta2, tau, "Softening Stage"

#%% 4. Generate Continuous Output & Plots
x_vec = np.linspace(0, 0.065, 1000) 
F_vec, m1_vec, V_vec, state_vec = [], [], [], []

for x in x_vec:
    F, m1, m2, V, _, _, _, _, _, _, state = solve_system(x)
    F_vec.append(F)
    m1_vec.append(m1)
    V_vec.append(V)
    state_vec.append(state)

F_vec = np.array(F_vec)
m1_vec = np.array(m1_vec)
V_vec = np.array(V_vec)
state_vec = np.array(state_vec)

# Dynamically locate structural state boundaries in mm
elastic_idx = np.where(state_vec == "Elastic Stage")[0]
hardening_idx = np.where(state_vec == "Hardening Stage")[0]
yielding_idx = np.where(state_vec == "Yielding Platform")[0]

x_cr_mm = x_vec[elastic_idx[-1]] * 1000 if len(elastic_idx) > 0 else 0.96
x_y_mm  = x_vec[hardening_idx[-1]] * 1000 if len(hardening_idx) > 0 else 11.35
x_u_mm  = x_vec[yielding_idx[-1]] * 1000 if len(yielding_idx) > 0 else 32.60
x_max_mm = x_vec[-1] * 1000

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Unified structural stage colors
zones = [
    (0, x_cr_mm, '#F8F9F9', 'Elastic'),
    (x_cr_mm, x_y_mm, '#FEF9E7', 'Hardening'),
    (x_y_mm, x_u_mm, '#E8F8F5', 'Yielding Platform'),
    (x_u_mm, x_max_mm, '#FDEDEC', 'Softening')
]

for ax in [ax1, ax2]:
    for x_start, x_end, bg_color, label_text in zones:
        ax.axvspan(x_start, x_end, color=bg_color, alpha=0.9, 
                   label=label_text if ax == ax1 else "")

# --- Left Plot: Capacity Curve ---
ax1.plot(x_vec*1000, F_vec, color='#2C3E50', lw=2.5, label='Capacity Curve')

# Reference textbook marker annotations
ax1.annotate("Example 4.3\n(2 mm, 72.5 kN)", xy=(2, 72.48), xytext=(8, 110),
             arrowprops=dict(arrowstyle="->", color="#1A252C", lw=1),
             fontsize=9, fontweight='bold', ha='center', color="#1A252C")

ax1.annotate("Example 4.4\n(20 mm, 153.0 kN)", xy=(20, 153.00), xytext=(26, 175),
             arrowprops=dict(arrowstyle="->", color="#1A252C", lw=1),
             fontsize=9, fontweight='bold', ha='center', color="#1A252C")

ax1.annotate("Example 4.5\n(60 mm, 145.3 kN)", xy=(60, 145.25), xytext=(48, 115),
             arrowprops=dict(arrowstyle="->", color="#1A252C", lw=1),
             fontsize=9, fontweight='bold', ha='center', color="#1A252C")

ax1.set_title("Column Lateral Response (Pushover)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Lateral Displacement, $x$ (mm)", fontsize=11)
ax1.set_ylabel("Lateral Force, $F$ (kN)", fontsize=11)
ax1.set_xlim(0, x_max_mm)
ax1.set_ylim(0, 195)
ax1.grid(True, linestyle=":", alpha=0.5, color='gray')
ax1.legend(loc="lower right", framealpha=1)

# --- Right Plot: Bending and Shear Demands ---
ax2.plot(x_vec*1000, -m1_vec, label="Bending Moment $|m_1|$ (kN-m)", color="#8E44AD", lw=2.5)
ax2.plot(x_vec*1000, V_vec, label="Shear Force $V$ (kN)", color="#16A085", lw=2.5, linestyle="--")

# Draw structural capacity indicators
ax2.axhline(y=m_y, color="#8E44AD", linestyle=":", lw=1.5, label="Moment Yield Capacity ($m_y = 229.5$)")
ax2.axhline(y=V_y, color="#16A085", linestyle=":", lw=1.5, label="Shear Yield Capacity ($V_y = 153.0$)")
ax2.axhline(y=m_cr, color="#8E44AD", linestyle="-.", lw=1.0, alpha=0.5, label="Moment Crack Capacity ($m_{cr} = 91.95$)")
ax2.axhline(y=V_cr, color="#16A085", linestyle="-.", lw=1.0, alpha=0.5, label="Shear Crack Capacity ($V_{cr} = 61.3$)")

ax2.set_title("Internal Demands vs. Lateral Displacement", fontsize=12, fontweight='bold')
ax2.set_xlabel("Lateral Displacement, $x$ (mm)", fontsize=11)
ax2.set_ylabel("Internal Demands ($kN$ or $kN-m$)", fontsize=11)
ax2.set_xlim(0, x_max_mm)
#ax2.set_ylim(0, 270)
ax2.grid(True, linestyle=":", alpha=0.5, color='gray')
ax2.legend(loc="lower right", framealpha=1)

plt.tight_layout()
plt.show()