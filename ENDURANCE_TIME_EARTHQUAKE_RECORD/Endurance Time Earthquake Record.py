import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
import SALAR_MATH as S01
#%% -----------------------------
# Parameters
dt = 0.01                  # [s] Time step
T_total = 10.0             # [s] Total duration
PGA_target = 0.30 * 9.81   # [m/s^2] Final PGA
time = np.arange(0, T_total, dt)
#%% -----------------------------
# Base acceleration (filtered noise)
noise = np.random.randn(len(time))

# Band-pass filter (earthquake-like)
f_low = 0.1
f_high = 10.0
fs = 1.0 / dt

sos = butter(
    4,
    [f_low / (fs / 2), f_high / (fs / 2)],
    btype='bandpass',
    output='sos'
)
a0 = sosfiltfilt(sos, noise)

# Normalize
a0 /= np.max(np.abs(a0))

#%% -----------------------------
# Endurance Time intensity function
f_t = time / T_total     # Linear intensity growth
acc_record = f_t * a0

# Scale to target PGA
acc_record *= PGA_target / np.max(np.abs(acc_record))

#%% -----------------------------
# Plot
plt.figure(figsize=(10,4))
plt.plot(time, acc_record,color='black')
plt.xlabel("Time [s]")
plt.ylabel("Earthquake Ground Acceleration [m/s²]")
plt.title("Endurance Time Earthquake Record")
plt.grid(True)
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
plt.plot(time, Normalized, color='red', linewidth=4.5)
plt.xlabel('Time [s]')
plt.ylabel('Normalized Cumulative Value')
plt.title('Normalized Cumulative Absolute Acceleration')
plt.grid(True, alpha=0.3)
plt.axhline(y=0.1, color='blue', linestyle='--', alpha=0.7, label='10% of total energy')
plt.axhline(y=0.9, color='gray', linestyle='--', alpha=0.7, label='90% of total energy')
plt.legend()
plt.tight_layout()
plt.show()
#%% -----------------------------