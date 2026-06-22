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

import STEEL_ELASTIC_SECTION_ANALYSIS_FUN as S01
import CONCRETE_ELASTIC_SECTION_ANALYSIS_FUN as S02
import COMPOSITE_ELASTIC_SECTION_ANALYSIS_FUN as S03

# YOUTUBE: Pushover Analysis Force Analogy Method with Force Control Based on Timoshenko Beam Theory in C
'https://www.youtube.com/watch?v=u9M6klJO6uU&t=58s'
# BOOK: Theory of Nonlinear Structural Analysis: The Force Analogy Method for Earthquake Engineering
# Gang Li, Kevin K. F. Wong (2014) - Wiley
'https://onlinelibrary.wiley.com/doi/book/10.1002/9781118718070'
# EXAMPLE: PAGE 37-41
#%% Input Parameters
"""
L = 3000.0        # [mm] Height of the column
# STEEL SECTION
#x_c, y_c, A_total, Ix_total, Iy_total = S01.STEEL_ELASTIC_SECTION_ANALYSIS_FUN()
# CONCRETE SECTION
#x_c, y_c, A_total, Ix_total, Iy_total = S02.CONCRETE_ELASTIC_SECTION_ANALYSIS_FUN()
# COMPOSITE SECTION
x_el, y_el, A_trans_total, Ix_trans_total, Iy_trans_total, E_ref = S03.COMPOSITE_ELASTIC_SECTION_ANALYSIS_FUN()
E = E_ref          # [N/mm^2] Young's Modulus of Steel
I = Ix_trans_total # [mm^4] Moment of Inertia  
"""
#%% 1. Define Model Parameters (Normalized)
E = 1.0
I = 1.0
L = 1.0  
Fo = 1.0

# Yield properties and post-yield rotational stiffnesses
my1 = 2.0 * Fo * L          # Yield moment of Hinge 1
my2 = Fo * L                # Yield moment of Hinge 2
Kt1 = E * I / L             # Post-yield stiffness of Hinge 1
Kt2 = 2.0 * E * I / L       # Post-yield stiffness of Hinge 2

#%% 2. Multi-Stage Structural Solver
def solve_sdof_system(F):
    """
    Solves the SDOF system for a given applied force F.
    Returns:
        x: Displacement
        theta1: Plastic rotation at Hinge 1
        theta2: Plastic rotation at Hinge 2
        m1: Moment at Hinge 1
        m2: Moment at Hinge 2
        state: Integer (1 = Both Elastic, 2 = Hinge 2 Yielded, 3 = Both Yielded)
    """
    # --- State 1: Assume Both Hinges Elastic (Example 2.7) ---
    x_trial = (F * L**3) / (12.0 * E * I)
    m1_trial = (6.0 * E * I / L**2) * x_trial
    m2_trial = (6.0 * E * I / L**2) * x_trial
    
    # Check if this assumption is valid
    if m1_trial <= my1 and m2_trial <= my2:
        return x_trial, 0.0, 0.0, m1_trial, m2_trial, 1
        
    # --- State 2: Assume Hinge 2 Yielded, Hinge 1 Elastic (Example 2.8) ---
    # Solve system: [ K_sub ] * [ x, -theta2 ]^T = [ F, my2 ]^T
    K2 = np.array([
        [12.0 * E * I / L**3, 6.0 * E * I / L**2],
        [6.0 * E * I / L**2,  4.0 * E * I / L + Kt2]
    ])
    F2 = np.array([F, my2])
    sol2 = np.linalg.solve(K2, F2)
    x_val = sol2[0]
    neg_theta2 = sol2[1]
    theta2_val = -neg_theta2
    
    # Calculate moments in this state
    m1_val = (6.0 * E * I / L**2) * x_val + (2.0 * E * I / L) * neg_theta2
    m2_val = my2 + Kt2 * theta2_val
    
    # Check if Hinge 1 remains elastic
    if m1_val <= my1:
        return x_val, 0.0, theta2_val, m1_val, m2_val, 2
        
    # --- State 3: Both Hinges Yielded (Example 2.9) ---
    # Solve system: [ K_full ] * [ x, -theta1, -theta2 ]^T = [ F, my1, my2 ]^T
    K3 = np.array([
        [12.0 * E * I / L**3, 6.0 * E * I / L**2,   6.0 * E * I / L**2],
        [6.0 * E * I / L**2,  4.0 * E * I / L + Kt1,  2.0 * E * I / L],
        [6.0 * E * I / L**2,  2.0 * E * I / L,      4.0 * E * I / L + Kt2]
    ])
    F3 = np.array([F, my1, my2])
    sol3 = np.linalg.solve(K3, F3)
    x_val = sol3[0]
    theta1_val = -sol3[1]
    theta2_val = -sol3[2]
    
    # Calculate moments in this state
    m1_val = my1 + Kt1 * theta1_val
    m2_val = my2 + Kt2 * theta2_val
    
    return x_val, theta1_val, theta2_val, m1_val, m2_val, 3

#%% 3. Compute Results for the Examples
example_forces = [1.0 * Fo, 3.0 * Fo, 5.0 * Fo]
example_names = ["Example 2.7 (F = Fo)", "Example 2.8 (F = 3Fo)", "Example 2.9 (F = 5Fo)"]

print("="*65)
print("              SUMMARY OF BOOK EXAMPLES RESULTS")
print("="*65)
for force, name in zip(example_forces, example_names):
    x, t1, t2, m1, m2, state = solve_sdof_system(force)
    print(f"\n--- {name} ---")
    print(f"State:       {'Both Elastic' if state==1 else 'PHL #2 Yielded' if state==2 else 'Both Yielded'}")
    print(f"Displacement (x):        {x:.4f} * (Fo*L^3/EI)")
    print(f"Hinge 1 Moment (m1):     {m1:.4f} * (Fo*L)")
    print(f"Hinge 2 Moment (m2):     {m2:.4f} * (Fo*L)")
    print(f"Hinge 1 Rotation (th1):  {t1:.4f} * (Fo*L^2/EI)")
    print(f"Hinge 2 Rotation (th2):  {t2:.4f} * (Fo*L^2/EI)")

#%% 4. Generate Pushover Capacity Curves
forces = np.linspace(0.0, 6.0 * Fo, 600)
results = [solve_sdof_system(f) for f in forces]

x_arr = np.array([r[0] for r in results])
t1_arr = np.array([r[1] for r in results])
t2_arr = np.array([r[2] for r in results])
m1_arr = np.array([r[3] for r in results])
m2_arr = np.array([r[4] for r in results])
state_arr = np.array([r[5] for r in results])

# Create Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Define colors for different structural states
colors = {1: 'blue', 2: 'orange', 3: 'red'}
labels = {1: 'Elastic State', 2: 'Hinge #2 Yielded', 3: 'Both Hinges Yielded'}

# --- Left Plot: Force vs. Displacement ---
for s in [1, 2, 3]:
    mask = state_arr == s
    if np.any(mask):
        ax1.plot(x_arr[mask], forces[mask], color=colors[s], lw=3, label=labels[s])

# Highlight book examples
ax1.scatter([1/12, 1/3, 16/15], [1.0, 3.0, 5.0], color='black', zorder=5)
ax1.text(1/12 + 0.03, 1.0, "Ex. 2.7\n(1/12, 1.0)", fontsize=9, fontweight='bold')
ax1.text(1/3 + 0.03, 3.0, "Ex. 2.8\n(1/3, 3.0)", fontsize=9, fontweight='bold')
ax1.text(16/15 - 0.22, 5.0, "Ex. 2.9\n(1.07, 5.0)", fontsize=9, fontweight='bold')

# Labels and styling
ax1.set_title("Force vs. Displacement (Pushover Curve)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Displacement, $x$ / ($F_o L^3 / EI$)", fontsize=11)
ax1.set_ylabel("Lateral Force, $F$ / $F_o$", fontsize=11)
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(loc="lower right")

# --- Right Plot: Moments vs. Displacement ---
ax2.plot(x_arr, m1_arr, label="Moment $m_1$ (Hinge 1)", color='purple', lw=2.5)
ax2.plot(x_arr, m2_arr, label="Moment $m_2$ (Hinge 2)", color='teal', lw=2.5, linestyle="--")

# Draw horizontal lines for yields
ax2.axhline(y=my1, color='purple', linestyle=":", alpha=0.7, label="Yield limit $m_{y1}$ (2.0)")
ax2.axhline(y=my2, color='teal', linestyle=":", alpha=0.7, label="Yield limit $m_{y2}$ (1.0)")

# Vertical transition markers
ax2.axvline(x=1/6, color='black', linestyle='--', alpha=0.3)  # Transition at x = 1/6
ax2.axvline(x=5/12, color='black', linestyle='--', alpha=0.3) # Transition at x = 5/12

ax2.set_title("Hinge Moments vs. Displacement", fontsize=12, fontweight='bold')
ax2.set_xlabel("Displacement, $x$ / ($F_o L^3 / EI$)", fontsize=11)
ax2.set_ylabel("Hinge Moment, $m$ / ($F_o L$)", fontsize=11)
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(loc="lower right")

plt.tight_layout()
plt.show()