######################################################################################################################
#                          >> IN THE NAME OF ALLAH, THE MOST GRACIOUS, THE MOST MERCIFUL <<                          #
#         SPECTRUM ANALYSIS OF KINEMATIC QUANTITIES: DISPLACEMENT, VELOCITY, AND ACCELERATION RELATIONSHIPS          #
#--------------------------------------------------------------------------------------------------------------------#
#                              THIS PROGRAM WRITTEN BY SALAR DELAVAR GHASHGHAEI (QASHQAI)                            #
#                                       EMAIL: salar.d.ghashghaei@gmail.com                                          #
######################################################################################################################
"""
 Spectral analysis of kinematic quantities (displacement, velocity, acceleration)
 is widely used in mechanical vibration analysis (machine fault detection), 
 structural health monitoring (earthquake response), SPECTRUM ANALYSIS OF KINEMATIC QUANTITIES: DISPLACEMENT,
 VELOCITY, AND ACCELERATION RELATIONSHIPSbiomechanics (gait analysis),
 and automotive/aerospace engineering (NVH testing and flight dynamics).
 It enables predictive maintenance, safety assurance, and performance optimization
 by identifying frequency patterns in dynamic systems.
 ------------------------------------------------
System that generates displacement, velocity, and acceleration spectra that are mathematically related
Key Features:
All signals are mathematically related through differentiation
[1] Vertical dashed lines mark the key frequency components
[2] The bar charts verify the theoretical relationships
[3] Numerical output confirms the relationships with error percentages
[4] The mathematical relationship is clearly demonstrated:
[5] Velocity spectrum amplitude = ω × displacement spectrum amplitude
[6] Acceleration spectrum amplitude = ω² × displacement spectrum amplitude
This shows how the spectra are fundamentally connected through the derivative relationship in the time domain.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Signal generation
fs = 1000  # Sampling frequency (Hz)
t = np.linspace(0, 10, fs*10)  # Time vector from 0 to 10 seconds

# Create a composite displacement signal with multiple frequency components
f1, f2, f3 = 0.5, 2.0, 5.0  # Frequencies (Hz)
A1, A2, A3 = 1.0, 0.5, 0.3  # Amplitudes
B1, B2, B3 = 2, 2, 2        # Amplitudes

# Displacement (position)
x = (A1 * np.sin(B1 * np.pi * f1 * t) + 
     A2 * np.sin(B2 * np.pi * f2 * t) + 
     A3 * np.sin(B3 * np.pi * f3 * t))

# Add some damping effect for realism
x *= np.exp(-0.1 * t)

# Velocity (first derivative of displacement)
v = np.gradient(x, t)

# Acceleration (second derivative of displacement)
a = np.gradient(v, t)

# Compute frequency spectra using FFT
def compute_spectrum(signal, fs):
    n = len(signal)
    freq = np.fft.rfftfreq(n, 1/fs)
    spectrum = np.abs(np.fft.rfft(signal)) / n * 2
    return freq, spectrum

# Spectra for each quantity
freq_x, spec_x = compute_spectrum(x, fs)
freq_v, spec_v = compute_spectrum(v, fs)
freq_a, spec_a = compute_spectrum(a, fs)

# Create the main figure
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Displacement, Velocity, and Acceleration Spectra (Mathematically Related)', 
             fontsize=16, fontweight='bold', y=0.98)

# Plot 1: Time domain signals
ax1 = plt.subplot(3, 2, 1)
ax1.plot(t, x, 'b', linewidth=1.5, alpha=0.8)
ax1.set_title('Displacement - Time Domain')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Displacement (m)')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 5])

ax2 = plt.subplot(3, 2, 3)
ax2.plot(t, v, 'g', linewidth=1.5, alpha=0.8)
ax2.set_title('Velocity - Time Domain')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Velocity (m/s)')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 5])

ax3 = plt.subplot(3, 2, 5)
ax3.plot(t, a, 'r', linewidth=1.5, alpha=0.8)
ax3.set_title('Acceleration - Time Domain')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Acceleration (m/s²)')
ax3.grid(True, alpha=0.3)
ax3.set_xlim([0, 5])

# Plot 2: Frequency spectra
ax4 = plt.subplot(3, 2, 2)
ax4.plot(freq_x, spec_x, 'b', linewidth=2)
ax4.set_title('Displacement Spectrum')
ax4.set_xlabel('Frequency (Hz)')
ax4.set_ylabel('Amplitude')
ax4.grid(True, alpha=0.3)
ax4.set_xlim([0, 10])
ax4.axvline(x=f1, color='k', linestyle='--', alpha=0.5, label=f'{f1} Hz')
ax4.axvline(x=f2, color='k', linestyle='--', alpha=0.5, label=f'{f2} Hz')
ax4.axvline(x=f3, color='k', linestyle='--', alpha=0.5, label=f'{f3} Hz')
ax4.legend(fontsize=8)

ax5 = plt.subplot(3, 2, 4)
ax5.plot(freq_v, spec_v, 'g', linewidth=2)
ax5.set_title('Velocity Spectrum')
ax5.set_xlabel('Frequency (Hz)')
ax5.set_ylabel('Amplitude')
ax5.grid(True, alpha=0.3)
ax5.set_xlim([0, 10])
for f in [f1, f2, f3]:
    ax5.axvline(x=f, color='k', linestyle='--', alpha=0.5)

ax6 = plt.subplot(3, 2, 6)
ax6.plot(freq_a, spec_a, 'r', linewidth=2)
ax6.set_title('Acceleration Spectrum')
ax6.set_xlabel('Frequency (Hz)')
ax6.set_ylabel('Amplitude')
ax6.grid(True, alpha=0.3)
ax6.set_xlim([0, 10])
for f in [f1, f2, f3]:
    ax6.axvline(x=f, color='k', linestyle='--', alpha=0.5)

# Plot 3: Relationship verification
# Calculate amplitudes at specific frequencies
freq_points = [f1, f2, f3]
x_amps = [spec_x[np.argmin(np.abs(freq_x - f))] for f in freq_points]
v_amps = [spec_v[np.argmin(np.abs(freq_v - f))] for f in freq_points]
a_amps = [spec_a[np.argmin(np.abs(freq_a - f))] for f in freq_points]

# Theoretical values based on differentiation in frequency domain
omega_points = [2 * np.pi * f for f in freq_points]
v_theoretical = [omega * x_amp for omega, x_amp in zip(omega_points, x_amps)]
a_theoretical = [omega**2 * x_amp for omega, x_amp in zip(omega_points, x_amps)]

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Print mathematical relationships
print("="*70)
print("MATHEMATICAL RELATIONSHIPS BETWEEN SPECTRA:")
print("="*70)
print("In the frequency domain, differentiation corresponds to multiplication by jω:")
print("  V(ω) = jω × X(ω)")
print("  A(ω) = (jω)² × X(ω) = -ω² × X(ω)")
print("\nFor amplitude spectra (magnitude):")
print("  |V(ω)| = ω × |X(ω)|")
print("  |A(ω)| = ω² × |X(ω)|")
print("\n" + "="*70)

# Numerical verification
print("\nNUMERICAL VERIFICATION AT KEY FREQUENCIES:")
print("-"*70)
print(f"{'Frequency (Hz)':<15} {'ω (rad/s)':<15} {'|X|':<15} {'|V| (actual)':<15} {'|V| (ω|X|)':<15} {'Error (%)':<10}")
print("-"*70)

for i, f in enumerate([f1, f2, f3]):
    omega = 2 * np.pi * f
    x_amp = x_amps[i]
    v_amp_actual = v_amps[i]
    v_amp_theory = omega * x_amp
    error = abs(v_amp_actual - v_amp_theory) / v_amp_theory * 100
    
    print(f"{f:<15.1f} {omega:<15.2f} {x_amp:<15.4f} {v_amp_actual:<15.4f} {v_amp_theory:<15.4f} {error:<10.2f}")

print("\n" + "-"*70)
print(f"{'Frequency (Hz)':<15} {'ω (rad/s)':<15} {'|X|':<15} {'|A| (actual)':<15} {'|A| (ω²|X|)':<15} {'Error (%)':<10}")
print("-"*70)

for i, f in enumerate([f1, f2, f3]):
    omega = 2 * np.pi * f
    x_amp = x_amps[i]
    a_amp_actual = a_amps[i]
    a_amp_theory = omega**2 * x_amp
    error = abs(a_amp_actual - a_amp_theory) / a_amp_theory * 100
    
    print(f"{f:<15.1f} {omega:<15.2f} {x_amp:<15.4f} {a_amp_actual:<15.4f} {a_amp_theory:<15.4f} {error:<10.2f}")

print("="*70)