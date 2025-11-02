##############################################################################################################################
#                         >> IN THE NAME OF ALLAH, THE MOST GRACIOUS, THE MOST MERCIFUL <<                                   #
# GENERATION OF SYNTHETIC EARTHQUAKE ACCELEROGRAM USING DUAL BETA PROBABILITY FUNCTIONS FOR TIME AND AMPLITUDE MODULATION    #
#----------------------------------------------------------------------------------------------------------------------------#
#                                        THIS PROGRAM WRITTEN BY SALAR DELAVAR GHASHGHAEI (QASHQAI)                          #
#                                                     EMAIL: salar.d.ghashghaei@gmail.com                                    #
##############################################################################################################################
"""
[1] This Python code generates a synthetic ground motion (accelerogram) using two Beta probability distribution functions to shape both the time envelope and acceleration amplitude modulation.
[2] The first Beta function controls the temporal evolution of shaking intensity — defining when the strongest motion occurs.
[3] The second Beta function governs how the acceleration amplitude varies across time.
[4] A band-limited random-phase sinusoidal signal represents the base motion, simulating realistic earthquake frequency content.
[5] The envelopes are multiplied by the base signal to obtain a non-stationary, physically meaningful accelerogram.
[6] The final acceleration is scaled to match a target peak ground acceleration (PGA), e.g., 0.6 g.
[7] The program outputs graphs of the accelerogram and Beta envelopes, plus key parameters such as duration, RMS, and PGA.
[8] This technique is valuable in seismic engineering and structural dynamics, where realistic synthetic ground motions are required for response-history analysis.
[9] It allows engineers to control shaking characteristics statistically while maintaining realistic energy distribution.
[10] Such generated signals can serve as input motions in OpenSeesPy or other structural simulation tools.
"""
#%%------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import pandas as pd
#%%------------------------------------------------------------------------------
# Parameters (you can change these)
duration = 30.0   # [s] seconds
dt = 0.01         # [s] time step
t = np.arange(0, duration, dt)
n = t.size
#%%------------------------------------------------------------------------------
# Beta parameters for the TIME envelope (controls when shaking occurs)
a_time = 2.5
b_time = 3.0
#%%------------------------------------------------------------------------------
# Beta parameters for the ACCELERATION modulation (controls how amplitudes grow/decay)
a_acc = 4.0
b_acc = 1.5
#%%------------------------------------------------------------------------------
# Frequency content for the base signal (band-limited)
f_MIN = 0.2   # [Hz]
f_MAX = 10.0  # [Hz]
n_freqs = 80

freqs = np.linspace(f_MIN, f_MAX, n_freqs)
#%%------------------------------------------------------------------------------
# Build a band-limited random-phase time series by summing sinusoids
MIN_PHASE = 0.0
MAX_PHASE = 2*np.pi
phases = np.random.uniform(MIN_PHASE, MAX_PHASE, size=n_freqs)
amps = 1.0 / np.sqrt(freqs)              # spectral shaping (more energy at low freq)
amps = amps / np.max(amps)               # normalize

base = np.zeros_like(t)
for i, f in enumerate(freqs):
    base += amps[i] * np.sin(MAX_PHASE*f*t + phases[i])
#%%------------------------------------------------------------------------------
# Normalize base to unit RMS
base = base / np.sqrt(np.mean(base**2))
#%%------------------------------------------------------------------------------
# Time envelope using Beta PDF (map time to [0,1])
t_norm = (t - t.min()) / (t.max() - t.min())
env_time = beta.pdf(t_norm, a_time, b_time)
env_time = env_time / np.max(env_time)   # normalize to 1 peak
#%%------------------------------------------------------------------------------
# Acceleration modulation envelope using another Beta PDF (also along time)
# This second beta acts as an additional amplitude-shaping function (interpreted as 'acceleration probability')
env_acc = beta.pdf(t_norm, a_acc, b_acc)
env_acc = env_acc / np.max(env_acc)

# Combine envelopes (multiply) and apply to base motion
acc = base * env_time * env_acc

# Scale to a target PGA (peak ground acceleration), e.g., 0.6 g (g=9.81 m/s^2)
target_pga_g = 0.6
g = 9.81
target_pga = target_pga_g * g
scale = target_pga / np.max(np.abs(acc))
acc = acc * scale
#%%------------------------------------------------------------------------------
# Compute some summary statistics
pga = np.max(np.abs(acc))
rms = np.sqrt(np.mean(acc**2))
duration_nonzero = duration

summary = pd.DataFrame({
    "duration_s": [duration_nonzero],
    "dt_s": [dt],
    "n_samples": [n],
    "PGA_m/s2": [pga],
    "RMS_m/s2": [rms],
    "target_PGA_m/s2": [target_pga]
})

#%%------------------------------------------------------------------------------
# Plot the accelerogram
plt.figure(figsize=(10,4))
plt.plot(t, acc, color='black')
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")
plt.title("Synthetic accelerogram shaped by two Beta PDFs (time & acceleration modulation)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Also show the two envelopes for clarity
plt.figure(figsize=(10,3))
plt.plot(t, env_time, label=f"Time envelope Beta(a={a_time}, b={b_time})")
plt.plot(t, env_acc, label=f"Accel envelope Beta(a={a_acc}, b={b_acc})")
plt.xlabel("Time (s)")
plt.ylabel("Normalized amplitude")
plt.title("Beta envelopes (normalized)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%------------------------------------------------------------------------------
# Save accelerogram to text file (time, acc) for user download if desired
out = np.column_stack([t, acc])
np.savetxt("salar_synthetic_accelerogram.txt", out, header="time_s acc_m/s2")
print("[Saved] salar_synthetic_accelerogram.txt")

#%%------------------------------------------------------------------------------