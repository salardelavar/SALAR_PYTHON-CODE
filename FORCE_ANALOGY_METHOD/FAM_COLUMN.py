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
# BOOK: Theory of Nonlinear Structural Analysis: The Force Analogy Method for Earthquake Engineering
# Gang Li, Kevin K. F. Wong (2014) - Wiley
'https://onlinelibrary.wiley.com/doi/book/10.1002/9781118718070'
# YOUTUBE: Pushover Analysis Force Analogy Method with Force Control Based on Timoshenko Beam Theory in C
'https://www.youtube.com/watch?v=u9M6klJO6uU&t=58s'

import numpy as np
import matplotlib.pyplot as plt
import STEEL_ELASTIC_SECTION_ANALYSIS_FUN as S01
import CONCRETE_ELASTIC_SECTION_ANALYSIS_FUN as S02
import COMPOSITE_ELASTIC_SECTION_ANALYSIS_FUN as S03

#%% Input Parameters
L = 3000.0        # [mm] Height of the column
# STEEL SECTION
#x_c, y_c, A_total, Ix_total, Iy_total = S01.STEEL_ELASTIC_SECTION_ANALYSIS_FUN()
# CONCRETE SECTION
#x_c, y_c, A_total, Ix_total, Iy_total = S02.CONCRETE_ELASTIC_SECTION_ANALYSIS_FUN()
# COMPOSITE SECTION
x_el, y_el, A_trans_total, Ix_trans_total, Iy_trans_total, E_ref = S03.COMPOSITE_ELASTIC_SECTION_ANALYSIS_FUN()
E = E_ref          # [N/mm^2] Young's Modulus of Steel
I = Ix_trans_total # [mm^4] Moment of Inertia

L = 3000.0             # [mm] Height of the column
Mp_base = 150e6        # [N.mm] Plastic Moment Capacity at Base
Mp_top = 250e6         # [N.mm] Plastic Moment Capacity at Top
D_target = 5.0         # [mm] Ultimate Target Displacement

#%% 1. Define Constant FAM Matrices
K_elastic = 12 * E * I / (L**3)

# Displacement-to-moment vector
K_d = np.array([6 * E * I / (L**2), 6 * E * I / (L**2)])

# Plastic rotation to end-moments restoring stiffness matrix
K_R = np.array([[4 * E * I / L, 2 * E * I / L],
                [2 * E * I / L, 4 * E * I / L]])

# Plastic rotation to lateral force recovery vector
K_p = np.array([6 * E * I / (L**2), 6 * E * I / (L**2)])

#%% 2. FAM Return-Mapping Solver Function
def FAM_ANALYSIS(d, E, I, L, Mp_base, Mp_top):
    """
    Solves for the state of the column at a given displacement 'd' 
    using the Force Analogy Method (FAM) formulation.
    """
    # Step A: Compute Trial Elastic State (assuming no plastic rotations)
    theta_p = np.zeros(2) # [theta_p_base, theta_p_top]
    M_trial = K_d * d - K_R @ theta_p
    
    # Check yield criteria
    yield_base = abs(M_trial[0]) > Mp_base
    yield_top = abs(M_trial[1]) > Mp_top
    
    # Step B: Apply Correction based on Active Yield Set
    if not yield_base and not yield_top:
        # State 1: Fully Elastic
        M = M_trial
        theta_p = np.zeros(2)
        
    elif yield_base and not yield_top:
        # State 2: Base Yielded, Top Elastic
        sign_base = np.sign(M_trial[0])
        # M_base is capped -> K_d[0]*d - K_R[0,0]*theta_p_base = sign*Mp_base
        theta_p[0] = (K_d[0] * d - sign_base * Mp_base) / K_R[0, 0]
        theta_p[1] = 0.0
        M = K_d * d - K_R @ theta_p
        
        # Double check if corrective action pushed top moment past its limit
        if abs(M[1]) > Mp_top:
            yield_top = True
            
    if yield_top and not yield_base:
        # State 2b: Top Yielded, Base Elastic (For general loading path)
        sign_top = np.sign(M_trial[1])
        theta_p[0] = 0.0
        theta_p[1] = (K_d[1] * d - sign_top * Mp_top) / K_R[1, 1]
        M = K_d * d - K_R @ theta_p
        
        if abs(M[0]) > Mp_base:
            yield_base = True
            
    if yield_base and yield_top:
        # State 3: Both Yielded (Plastic Mechanism)
        sign_base = np.sign(M_trial[0])
        sign_top = np.sign(M_trial[1])
        M_target = np.array([sign_base * Mp_base, sign_top * Mp_top])
        # Solve the system of equations for plastic rotations: K_R * theta_p = K_d * d - M_target
        theta_p = np.linalg.solve(K_R, K_d * d - M_target)
        M = M_target
        
    # Step C: Compute Restoring Force using FAM Force Recovery Matrix
    V = K_elastic * d - np.dot(K_p, theta_p)
    
    return V, M[0], M[1], theta_p[0], theta_p[1]

#%% 3. Generate Pushover Data Points
disp_steps = np.linspace(0, D_target, 50)
force = np.zeros_like(disp_steps)
M_base = np.zeros_like(disp_steps)
M_top = np.zeros_like(disp_steps)
theta_p_base = np.zeros_like(disp_steps)
theta_p_top = np.zeros_like(disp_steps)
step = 0
# Run the displacement-controlled FAM analysis
for i, d in enumerate(disp_steps):
    step += 1
    V, Mb, Mt, tp_b, tp_t = FAM_ANALYSIS(d, E, I, L, Mp_base, Mp_top)
    force[i] = V
    M_base[i] = Mb
    M_top[i] = Mt
    theta_p_base[i] = tp_b
    theta_p_top[i] = tp_t
    print(f'STEP: {step} - DISP.: {d:.3f} mm - REACTION: {V:.3f}')

# Identify hinge transition points for visualization
idx_hinge1 = np.where(theta_p_base > 1e-9)[0]
idx_hinge2 = np.where(theta_p_top > 1e-9)[0]

D1, V1 = (disp_steps[idx_hinge1[0]], force[idx_hinge1[0]]) if len(idx_hinge1) > 0 else (None, None)
D2, V2 = (disp_steps[idx_hinge2[0]], force[idx_hinge2[0]]) if len(idx_hinge2) > 0 else (None, None)

#%% 4. Plot the Results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

# Plot 1: Pushover Curve (Base Shear vs. Displacement)
ax1.plot(disp_steps, force, label='FAM Pushover Curve', color='navy', lw=2.5)
if D1 is not None:
    ax1.scatter(D1, V1, color='crimson', zorder=20, 
                label=f'1st Hinge (Base): {V1/1e3:.1f} kN, {D1:.2f} mm')
if D2 is not None:
    ax1.scatter(D2, V2, color='darkorange', zorder=20, 
                label=f'2nd Hinge (Top): {V2/1e3:.1f} kN, {D2:.2f} mm')
ax1.set_title('Pushover Curve (Base Shear vs. Disp.)', fontsize=11, fontweight='bold')
ax1.set_xlabel('Lateral Displacement (mm)', fontsize=10)
ax1.set_ylabel('Base Shear (N)', fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='lower right', fontsize=9)
ax1.set_xlim(0, D_target)
ax1.set_ylim(0, max(force) * 1.15)

# Plot 2: Moment Evolution
ax2.plot(disp_steps, M_base, label='Base Moment ($M_{base}$)', color='crimson', lw=2)
ax2.plot(disp_steps, M_top, label='Top Moment ($M_{top}$)', color='forestgreen', lw=2)
ax2.axhline(y=Mp_base, color='crimson', linestyle=':', alpha=0.7, label='Base Capacity ($M_{p,base}$)')
ax2.axhline(y=Mp_top, color='forestgreen', linestyle=':', alpha=0.7, label='Top Capacity ($M_{p,top}$)')

ax2.set_title('Bending Moment Evolution', fontsize=11, fontweight='bold')
ax2.set_xlabel('Lateral Displacement (mm)', fontsize=10)
ax2.set_ylabel('Bending Moment (N.mm)', fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='lower right', fontsize=9)
ax2.set_xlim(0, D_target)
ax2.set_ylim(0, max(M_top) * 1.15)

# Plot 3: Plastic Rotation Evolution (Unique feature of FAM tracking)
ax3.plot(disp_steps, theta_p_base, label=r'Base Hinge (Rotation_{base})', color='crimson', lw=2)
ax3.plot(disp_steps, theta_p_top, label=r'Top Hinge (Rotation_{top})', color='forestgreen', lw=2)

ax3.set_title('Plastic Rotation Evolution', fontsize=11, fontweight='bold')
ax3.set_xlabel('Lateral Displacement (mm)', fontsize=10)
ax3.set_ylabel('Plastic Rotation (rad)', fontsize=10)
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.legend(loc='upper left', fontsize=9)
ax3.set_xlim(0, D_target)
ax3.set_ylim(0, max(theta_p_base) * 1.15 if len(idx_hinge1) > 0 else 0.1)

plt.tight_layout()
plt.show()