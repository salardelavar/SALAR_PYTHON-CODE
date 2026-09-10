"""
%***********************************************************%
%                  >> IN THE NAME OF GOD <<                 %
% Pushover Analysis of Nonlinear Springs with Force Control %
%-----------------------------------------------------------%
%     This program is written by salar delavar ghashghaei   %  
%            E-mail:salar.d.ghashghaei@gmail.com            %
%             Publication Date : 25 - May - 2017            %
%***********************************************************%
"""
import numpy as np
import matplotlib.pyplot as plt

# Parameters
P = 1.0
m = 10000
itermax = 500
tolerance = 1e-12
u = 0.0

# Spring properties
Force = np.array([20.0, 30.0, 32.0, 35.0])
Displacement = np.array([5.0, 35.0, 50.0, 80.0])
Coff = np.array([0.5, 0.8, 1.0, 0.8, 0.5])

Dmax = np.max(Displacement)

Rk1 = (Force[0] - 0.0) / (Displacement[0] - 0.0)
Rk2 = (Force[1] - Force[0]) / (Displacement[1] - Displacement[0])
Rk3 = (Force[2] - Force[1]) / (Displacement[2] - Displacement[1])
Rk4 = (Force[3] - Force[2]) / (Displacement[3] - Displacement[2])

f = np.zeros(5)

print('#################################################')
print('#    Pushover Analysis of Nonlinear Springs     #')
print('#################################################')

F1i, U1, DU1, I1, IT1 = [], [], [], [], []
last_i = 0

# Nonlinear springs analysis
for i in range(1, m + 1):
    F = P * i

    K = np.zeros(5)
    Kini = 0.0

    for j in range(5):
        abs_fj = abs(f[j])
        abs_u = abs(u)

        if 0.0 <= abs_fj <= Force[0]:
            K[j] = Coff[j] * Rk1
        elif Force[0] < abs_fj <= Force[1]:
            K[j] = Coff[j] * (Force[0] + Rk2 * (abs_u - Displacement[0])) / abs_u
        elif Force[1] < abs_fj <= Force[2]:
            K[j] = Coff[j] * (Force[1] + Rk3 * (abs_u - Displacement[1])) / abs_u
        elif Force[2] < abs_fj <= Force[3]:
            K[j] = Coff[j] * (Force[2] + Rk4 * (abs_u - Displacement[2])) / abs_u
        else:
            K[j] = 0.0

        Kini += K[j]

    it = 0
    residual = 100.0

    while residual > tolerance:
        Ko = 0.0
        Kt = np.zeros(5)

        for j in range(5):
            abs_fj = abs(f[j])
            abs_u = abs(u)

            if 0.0 <= abs_fj <= Force[0]:
                Kt[j] = Coff[j] * Rk1
            elif Force[0] < abs_fj <= Force[1]:
                Kt[j] = Coff[j] * (Force[0] + Rk2 * (abs_u - Displacement[0])) / abs_u
            elif Force[1] < abs_fj <= Force[2]:
                Kt[j] = Coff[j] * (Force[1] + Rk3 * (abs_u - Displacement[1])) / abs_u
            elif Force[2] < abs_fj <= Force[3]:
                Kt[j] = Coff[j] * (Force[2] + Rk4 * (abs_u - Displacement[2])) / abs_u
            else:
                Kt[j] = 0.0

            Ko += Kt[j]

        ff = Ko * u - F
        du = -ff / Kini
        residual = abs(du)

        it += 1

        if it == itermax:
            print(f'(-)For increment {i:.0f} trail iteration reached to Ultimate {it:.0f}')
            print('    ## The solution for this step is not converged ##')
            break

        u += du

    if it < itermax:
        print(f'(+)Increment {i:.0f} : It is converged in {it:.0f} iterations')

    for k in range(5):
        f[k] = K[k] * u

    F1i.append(F)
    U1.append(u)
    DU1.append(residual)
    I1.append(i)
    IT1.append(it)

    last_i = i

    if abs(u) >= Dmax:
        print('  ## Displacement reached to ultimate displacement ##')
        break

D1 = np.concatenate(([0.0], np.array(U1)))
F1 = np.concatenate(([0.0], np.array(F1i)))

print('#################################################')
print('#      Pushover Analysis of Linear Springs      #')
print('#################################################')

F2i, U2, DU2, I2, IT2 = [], [], [], [], []

# Linear springs analysis
for i in range(1, last_i + 1):
    F = P * i

    K = np.zeros(5)
    Kini = 0.0

    for j in range(5):
        K[j] = Coff[j] * Rk1
        Kini += K[j]

    it = 0
    residual = 100.0

    while residual > tolerance:
        Ko = 0.0
        Kt = np.zeros(5)

        for j in range(5):
            Kt[j] = Coff[j] * Rk1
            Ko += Kt[j]

        ff = Ko * u - F
        du = -ff / Kini
        residual = abs(du)

        it += 1

        if it == itermax:
            print(f'(-)For increment {i:.0f} trail iteration reached to Ultimate {it:.0f}')
            print('    ## The solution for this step is not converged ##')
            break

        u += du

    if it < itermax:
        print(f'(+)Increment {i:.0f} : It is converged in {it:.0f} iterations')

    for k in range(5):
        f[k] = K[k] * u

    F2i.append(F)
    U2.append(u)
    DU2.append(residual)
    I2.append(i)
    IT2.append(it)

D2 = np.concatenate(([0.0], np.array(U2)))
F2 = np.concatenate(([0.0], np.array(F2i)))

# Figure 1: optional image
try:
    IMAGE = plt.imread('PushoverAnalysisNonlinearSpringsFC.jpg')
    plt.figure(1)
    plt.imshow(IMAGE)
    plt.axis('off')
except Exception as e:
    print(f'Image could not be loaded: {e}')

# Figure 2
plt.figure(2)
plt.plot(I1, DU1, color='black', linewidth=2, label='Nonlinear')
plt.plot(I2, DU2, color='green', linestyle='--', linewidth=2, label='Linear')
plt.grid(True)
plt.xlabel('increment')
plt.ylabel('Residual')
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
plt.title('Residual-Increment diagram', color='b')

# Figure 3
plt.figure(3)
plt.plot(I1, IT1, color='black', linewidth=2, label='Nonlinear')
plt.plot(I2, IT2, color='green', linestyle='--', linewidth=2, label='Linear')
plt.grid(True)
plt.xlabel('increment')
plt.ylabel('Iteration')
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
plt.title('Iteration-Increment diagram', color='b')

# Figure 4
plt.figure(4)
plt.plot(D1, F1, color='black', linewidth=2, label='Nonlinear')
plt.plot(D2, F2, color='red', linestyle='--', linewidth=2, label='Linear')
plt.grid(True)
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
plt.xlabel('Displacement')
plt.ylabel('Force')
plt.title('Force-Displacement Diagram of Pushover Analysis Linear and Nonlinear Springs', color='b')

plt.tight_layout()
plt.show()
