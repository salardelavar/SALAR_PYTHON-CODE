###########################################################################################################
#                     >> IN THE NAME OF ALLAH, THE MOST GRACIOUS, THE MOST MERCIFUL <<                    #
#                                 GENERATE ARTIFICIAL GROUND ACCELERATION                                 #
#---------------------------------------------------------------------------------------------------------#
#                          THIS PROGRAM WRITTEN BY SALAR DELAVAR GHASHGHAEI (QASHQAI)                     #
#                                   EMAIL: salar.d.ghashghaei@gmail.com                                   #
###########################################################################################################
import numpy as np
import matplotlib.pyplot as plt
import SALAR_MATH as S01
#%% ------------------------------------------------------------
# Generate Artificial Ground Acceleration Record
def GENERATE_ARTIFICIAL_ACCEL(duration, dt, max_accel=0.3*9.81):
    """
    Generate artificial acceleration record
    """
    import numpy as np
    npts = int(duration/dt)
    t = np.linspace(0, duration, npts)
    
    # Combine multiple sine waves with different frequencies
    acc = np.zeros(npts)
    frequencies = [0.5, 1.0, 2.0, 3.0, 5.0]  # Hz
    
    for freq in frequencies:
        amplitude = max_accel / len(frequencies)
        phase = np.random.random() * 2 * np.pi
        acc += amplitude * np.sin(2 * np.pi * freq * t + phase)
    
    # Add random noise
    noise = 0.1 * max_accel * np.random.randn(npts)
    acc += noise
    
    # Windowing
    window = np.ones(npts)
    ramp_up = int(0.1 * npts)
    ramp_down = int(0.9 * npts)
    
    window[:ramp_up] = np.linspace(0, 1, ramp_up)
    window[ramp_down:] = np.linspace(1, 0, npts - ramp_down)
    
    acc *= window
    
    return t, acc
#%% ------------------------------------------------------------    
print("Generating artificial acceleration record...")
t_acc, acc_record = GENERATE_ARTIFICIAL_ACCEL(duration=20.0, dt=0.01, max_accel=0.3*9.81)
#%% ------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(t_acc, acc_record, color='black', linewidth=1)
plt.xlabel('Time [s]')
plt.ylabel('Earthquake Ground Acceleration [m/s²]')
plt.title('Artificial Acceleration Record for Response Spectrum Analysis')
plt.grid(True)
plt.tight_layout()
plt.show()
#%% -----------------------------
# Plot Histogram
X = acc_record
HISTO_COLOR = 'cyan' 
LABEL = 'Generate Artificial Acceleration Record Histogram'
S01.HISROGRAM_BOXPLOT(X, HISTO_COLOR, LABEL)

# Plot Histogram of First derivative (changes)
changes = np.diff(X)
HISTO_COLOR = 'lime' 
LABEL = 'First derivative Histogram of Generate Artificial Acceleration Record'
S01.HISROGRAM_BOXPLOT(changes, HISTO_COLOR, LABEL)

# Calculate cumulative sum of absolute acceleration and normalize it
cum_sum_acc = np.cumsum(np.abs(acc_record))
#cum_sum_acc = np.cumsum(np.abs(np.sort(acc_record)))
sum_acc = np.sum(np.abs(acc_record))
Normalized = cum_sum_acc / sum_acc

plt.figure(figsize=(12, 8))
plt.plot(t_acc, Normalized, color='red', linewidth=4.5)
plt.xlabel('Time [s]')
plt.ylabel('Normalized Cumulative Value')
plt.title('Normalized Cumulative Absolute Acceleration')
plt.grid(True, alpha=0.3)
plt.axhline(y=0.1, color='blue', linestyle='--', alpha=0.7, label='10% of total energy')
plt.axhline(y=0.9, color='gray', linestyle='--', alpha=0.7, label='90% of total energy')
plt.legend()
plt.tight_layout()
plt.show()
#%% ------------------------------------------------------------
np.savetxt("Artificial_Ground_Acceleration_Record.txt", np.column_stack((t_acc, acc_record)))
np.savetxt(
    "Artificial_Ground_Acceleration_Record.txt",
    np.column_stack((t_acc, acc_record)),
    header="Time(s)   Acceleration(m/s^2)"
)
#%% ------------------------------------------------------------
