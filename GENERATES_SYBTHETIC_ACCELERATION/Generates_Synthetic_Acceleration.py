######################################################################################################################
#                          >> IN THE NAME OF ALLAH, THE MOST GRACIOUS, THE MOST MERCIFUL <<                          #
#                STRUCTURAL MODAL IDENTIFICATION: FROM SYNTHETIC ACCELERATION TO DYNAMIC PROPERTIES                  #
#                                        a(t) = A * exp(-B*C*t) * sin(C*t) + noise                                   #
#--------------------------------------------------------------------------------------------------------------------#
#                         THIS OYTHON SCRIPT WRITTEN BY SALAR DELAVAR GHASHGHAEI (QASHQAI)                           #
#                                       EMAIL: salar.d.ghashghaei@gmail.com                                          #
######################################################################################################################
"""
Structural Dynamic Response Analysis Using Synthetic Acceleration Data

This code generates a synthetic acceleration time history representing the damped dynamic response of a structure.
The acceleration signal is numerically integrated to obtain velocity and displacement time histories.
Frequency-domain analysis (FFT) is used to identify the dominant natural frequency and corresponding period.
The structural damping ratio is estimated from the displacement response using the logarithmic decrement method.

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.signal import detrend, find_peaks
from scipy.fft import fft, fftfreq

#%% Parameters
dt = 0.01          # [s] Time step
t_end = 10.0       # [s] Total duration
t = np.arange(0, t_end, dt)

C = 2 * np.pi * 1.2
B = 0.05           # True damping ratio (5%)
A = 2.0            # [m/s^2] Acceleration amplitude

# --------------------------------------------------
# Synthetic acceleration time history
# a(t) = A * exp(-B*C*t) * sin(C*t) + noise
# --------------------------------------------------
noise = 0.2 * np.random.randn(len(t))
acc = A * np.exp(-B * C * t) * np.sin(C * t) + noise

#dt = 0.01         # Time step (seconds)
#acc = np.loadtxt("acc.txt")  # Synthetic acceleration time history
#t = np.arange(0, len(acc)*dt, dt)

#%% Plot acceleration time history
plt.figure(figsize=(8, 4))
plt.plot(t, acc, color='black')
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Synthetic Acceleration Time History")
plt.grid()
plt.show()

#%% Remove trend (drift correction)
acc = detrend(acc)

#%% Compute velocity and displacement by integration
vel = cumtrapz(acc, t, initial=0)
disp = cumtrapz(vel, t, initial=0)

#%% FFT analysis to obtain dominant frequency
N = len(acc)
freqs = fftfreq(N, dt)
fft_vals = np.abs(fft(acc))

positive_freqs = freqs[freqs > 0]
positive_fft = fft_vals[freqs > 0]

dominant_freq = positive_freqs[np.argmax(positive_fft)]
T = 1 / dominant_freq

#%% Plot acceleration, velocity, displacement
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(t, acc, color='black')
plt.ylabel("Acceleration (m/s²)")
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t, vel, color='red')
plt.ylabel("Velocity (m/s)")
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t, disp, color='green')
plt.ylabel("Displacement (m)")
plt.xlabel("Time (s)")
plt.grid()

plt.tight_layout()
plt.show()

print(f"Dominant Frequency = {dominant_freq:.2f} Hz")
print(f"Natural Period T = {T:.3f} s")

# --------------------------------------------------
# Damping ratio estimation (Logarithmic decrement)
# --------------------------------------------------
peaks, _ = find_peaks(disp, distance=20)

x1 = disp[peaks[0]]
x2 = disp[peaks[1]]

delta = np.log(x1 / x2)
zeta_est = delta / np.sqrt((2 * np.pi)**2 + delta**2)

print(f"Estimated Damping Ratio ζ = {zeta_est:.4f}")
