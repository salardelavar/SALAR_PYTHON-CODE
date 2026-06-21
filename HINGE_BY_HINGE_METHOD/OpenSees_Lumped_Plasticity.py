import openseespy.opensees as ops
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Structural Parameters
# ==========================================
L = 3000.0        # [mm] Column Height
E = 500000         # Elastic Modulus (Pa)
I = 1865466772.9036107        # Moment of Inertia (m^4)
Mp_base = 150e6   # [N.mm] Yield moment of base hinge
Mp_top = 250e6    # [N.mm] Yield moment of top hinge
D_target = 5      # [mm] Target Displacement
num_steps = 500

# To model plastic hinges as discrete rotational springs, we assign high 
# elastic stiffness to the springs so they behave as nearly rigid-plastic.
K_spring = 1.0e5 * E * I / L 

# ==========================================
# 2. OpenSees Py Model Definition
# ==========================================
ops.wipe()
ops.model('basic', '-ndm', 2, '-ndf', 3)

# Nodes: 
# Rotational springs (zero-length) are defined between duplicated nodes at the ends.
ops.node(1, 0.0, 0.0)    # Base support node
ops.node(2, 0.0, 0.0)    # Column bottom node
ops.node(3, 0.0, L)      # Column top node
ops.node(4, 0.0, L)      # Guided top node (where load is applied)

# Boundary Conditions
ops.fix(1, 1, 1, 1)      # Base support is fully fixed
ops.equalDOF(1, 2, 1, 2) # Pin bottom of column translationally to base support

ops.fix(4, 0, 1, 1)      # Top support is guided (UX free, UY and RZ fixed)
ops.equalDOF(3, 4, 1, 2) # Pin top of column translationally to top support

# Rotational Spring Materials (Elastoplastic with low strain hardening)
ops.uniaxialMaterial('Steel01', 1, Mp_base, K_spring, 1e-4)
ops.uniaxialMaterial('Steel01', 2, Mp_top, K_spring, 1e-4)

# Elements
# Zero-length elements act as rotational hinges (dir 3 is RZ rotation in 2D)
ops.element('zeroLength', 1, 1, 2, '-mat', 1, '-dir', 3)
ops.element('zeroLength', 2, 3, 4, '-mat', 2, '-dir', 3)

# Column elastic segment
ops.geomTransf('Linear', 1)
A_arbitrary = 0.1  # Cross-sectional area (not sensitive if axial load is 0)
ops.element('elasticBeamColumn', 3, 2, 3, A_arbitrary, E, I, 1)

# ==========================================
# 3. Nonlinear Pushover Analysis
# ==========================================
ops.timeSeries('Linear', 1)
ops.pattern('Plain', 1, 1)
ops.load(4, 1.0, 0.0, 0.0) # Apply unit reference load in UX

ops.system('BandGeneral')
ops.numberer('Plain')
ops.constraints('Transformation')
ops.test('NormDispIncr', 1e-8, 100)
ops.algorithm('Newton')

disp_step = D_target / num_steps
ops.integrator('DisplacementControl', 4, 1, disp_step)
ops.analysis('Static')

# Data Storage arrays
ops_disp = []
ops_shear = []

for step in range(num_steps):
    ok = ops.analyze(1)
    if ok != 0:
        print(f"Analysis failed at step {step}")
        #break
    
    # Store displacement
    ops_disp.append(ops.nodeDisp(4, 1))
    
    # Calculate reactions to get base shear
    ops.reactions()
    ops_shear.append(ops.nodeReaction(1, 1)) # Base horizontal reaction

# ==========================================
# 4. Plotting OpenSeesPy Numerical Curve
# ==========================================

plt.figure(figsize=(7, 5))
plt.plot(ops_disp, ops_shear, color='darkviolet', lw=2.5, label='OpenSeesPy Model')
plt.title('OpenSeesPy Lumped Plasticity Pushover', fontsize=12, fontweight='bold')
plt.xlabel('Lateral Displacement (mm)', fontsize=10)
plt.ylabel('Base Shear (N)', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xlim(0, D_target)
plt.ylim(0, max(ops_shear) * 1.15)
plt.legend()
plt.show()