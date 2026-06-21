###########################################################################################################
#                   >> IN THE NAME OF ALLAH, THE MOST GRACIOUS, THE MOST MERCIFUL <<                      #
#                     PUSHOVER ANALYSIS OF STRUCTURE WITH HINGE-BY-HINGE METHOD                           #
#---------------------------------------------------------------------------------------------------------#
#                    THIS PYTHON SCRIPT WRITTEN BY SALAR DELAVAR GHASHGHAEI (QASHQAI)                     #
#                                   EMAIL: salar.d.ghashghaei@gmail.com                                   #
###########################################################################################################
"""
This python script performs a hinge‑by‑hinge pushover analysis of a fixed‑guided column, sequentially tracking plastic hinge formation and stiffness degradation.  
1. It imports composite elastic section properties (E, I) to define the member’s flexural rigidity.  
2. Plastic moment capacities are assigned as weaker at the base (Mp_base) and stronger at the top (Mp_top).  
3. State 1: The column is fully elastic with both ends rotationally fixed; lateral stiffness is K₁=12EI/L³ and end moments are equal (V·L/2).  
4. First hinge forms at the base when the base moment reaches Mp_base, yielding force V₁ and displacement D₁.  
5. State 2: The base becomes a plastic hinge (pinned), while the top remains fixed against rotation; incremental stiffness drops to K₂=3EI/L³.  
6. Base moment stays constant at Mp_base, and the top moment increases linearly with additional lateral load.  
7. Second hinge develops at the top when the top moment attains Mp_top, defining total force V₂ and displacement D₂.  
8. State 3: A mechanism is formed (both ends hinged); lateral stiffness becomes zero and the force remains constant at V₂.  
9. The pushover curve (base shear vs. displacement) and moment evolution are plotted to visualize the sequential yielding.  
10. This hinge‑by‑hinge method explicitly captures moment redistribution, stiffness reduction, and the incremental formation of a collapse mechanism.
"""
# BOOK: Plastic Design and Second-Order Analysis of Steel Frames
'https://link.springer.com/book/10.1007/978-1-4613-8428-1'
# YOUTUBE: Pushover Analysis of Steel Frame Structures with Hinge by Hinge Method in EXCEL
'https://www.youtube.com/watch?v=UqZxtnJgWWA&t=35s'    
# YOUTUBE: Pushover Analysis of Fixed Support Beam with Hinge by Hinge Method in C programming
'https://www.youtube.com/watch?v=Zd0JRMtRBkk'    
import numpy as np
import matplotlib.pyplot as plt
import STEEL_ELASTIC_SECTION_ANALYSIS_FUN as S01
import CONCRETE_ELASTIC_SECTION_ANALYSIS_FUN as S02
import COMPOSITE_ELASTIC_SECTION_ANALYSIS_FUN as S03

#%% 1. Input Parameters
L = 3000.0        # [mm] Height of the column
# STEEL SECTION
#x_c, y_c, A_total, Ix_total, Iy_total = S01.STEEL_ELASTIC_SECTION_ANALYSIS_FUN()
# CONCRETE SECTION
#x_c, y_c, A_total, Ix_total, Iy_total = S02.CONCRETE_ELASTIC_SECTION_ANALYSIS_FUN()
# COMPOSITE SECTION
x_el, y_el, A_trans_total, Ix_trans_total, Iy_trans_total, E_ref = S03.COMPOSITE_ELASTIC_SECTION_ANALYSIS_FUN()
E = E_ref          # [N/mm^2] Young's Modulus of Steel
I = Ix_trans_total # [mm^4] Moment of Inertia
Mp_base = 150e6    # [N.mm] Plastic Moment Capacity at Base
Mp_top = 250e6     # [N.mm] Plastic Moment Capacity at Top

# Target displacement to complete the pushover curve
D_target = 5.0    # [mm] Ultimate Displacement

#%% 2. Hinge-by-Hinge Analytical Formulation

# --- STATE 1: Fully Elastic (Fixed-Guided Column) ---
# Lateral Stiffness: K1 = 12 * E * I / L^3
# Bending moments at both ends under lateral load V: M = V * L / 2
K1 = 12 * E * I / (L**3)

# First hinge will form at the base since Mp_base < Mp_top
V1 = 2 * Mp_base / L
D1 = V1 / K1
M_top_at_1 = V1 * L / 2  # Moment at the top when the base yields

# --- STATE 2: Base Hinged, Top Elastic (Pinned-Guided Column) ---
# Incremental Stiffness: K2 = 3 * E * I / L^3
# Incremental bending moment at the top under incremental load dV: dM = dV * L
# Bottom moment remains constant at Mp_base
K2 = 3 * E * I / (L**3)

# Incremental force needed to yield the top
delta_V2 = (Mp_top - M_top_at_1) / L
V2 = V1 + delta_V2

# Incremental displacement needed to yield the top
delta_D2 = delta_V2 / K2
D2 = D1 + delta_D2

# --- STATE 3: Both Hinges Formed (Mechanism) ---
# Lateral stiffness is now zero (K3 = 0)
# Force remains constant at V2 for displacements beyond D2

#%% 3. Generate Pushover Data Points
disp_steps = np.linspace(0, D_target, 500)
force = np.zeros_like(disp_steps)
M_base = np.zeros_like(disp_steps)
M_top = np.zeros_like(disp_steps)

for i, d in enumerate(disp_steps):
    if d <= D1:
        # State 1: Fully Elastic
        force[i] = K1 * d
        M_base[i] = force[i] * L / 2
        M_top[i] = force[i] * L / 2
    elif d <= D2:
        # State 2: Pinned-Guided (Base Hinged)
        delta_d = d - D1
        force[i] = V1 + K2 * delta_d
        M_base[i] = Mp_base
        M_top[i] = M_top_at_1 + K2 * delta_d * L
    else:
        # State 3: Mechanism
        force[i] = V2
        M_base[i] = Mp_base
        M_top[i] = Mp_top

#%% 4. Plot the Results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# Plot 1: Pushover Curve (Base Shear vs. Displacement)
ax1.plot(disp_steps, force, label='Pushover Curve', color='navy', lw=2.5)
ax1.scatter(D1, V1, color='crimson', zorder=5, 
            label=f'1st Hinge (Base Yields): {V1:.1f} N, {D1:.1f} mm')
ax1.scatter(D2, V2, color='darkorange', zorder=5, 
            label=f'2nd Hinge (Top Yields): {V2:.1f} N, {D2:.1f} mm')

ax1.set_title('Column Pushover Curve (Base Shear vs. Displacement)', fontsize=12, fontweight='bold')
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

ax2.set_title('Bending Moment Evolution at Column Ends', fontsize=12, fontweight='bold')
ax2.set_xlabel('Lateral Displacement (mm)', fontsize=10)
ax2.set_ylabel('Bending Moment (N.mm)', fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='lower right', fontsize=9)
ax2.set_xlim(0, D_target)
ax2.set_ylim(0, max(M_top) * 1.15)

plt.tight_layout()
plt.show()