######################################################################################################################
#                          >> IN THE NAME OF ALLAH, THE MOST GRACIOUS, THE MOST MERCIFUL <<                          #
#                        DESIGNING RESONANCE-ORIENTED ACCELEROGRAMS FOR ELASTIC SDOF SYSTEMS                         #
#--------------------------------------------------------------------------------------------------------------------#
#                              THIS PROGRAM WRITTEN BY SALAR DELAVAR GHASHGHAEI (QASHQAI)                            #
#                                       EMAIL: salar.d.ghashghaei@gmail.com                                          #
######################################################################################################################
"""
In earthquake engineering, resonance occurs when the dominant frequency content of ground motion coincides with the natural frequency of a structure. For a single-degree-of-freedom (SDOF) system, this alignment can amplify displacements, accelerations, and internal forces far beyond non-resonant excitations.
The natural period of an SDOF oscillator is defined by stiffness and mass, Tn = 2π√(m/k). The corresponding circular frequency ωn governs free vibration. When an earthquake record carries strong energy near ω_n, the system exhibits resonance-like behavior.
To design an accelerogram that deliberately excites resonance, shape the input so its spectral energy is concentrated around the target frequency. A practical way is to construct narrow-band signals modulated by realistic envelopes, ensuring gradual build-up, peak intensity, and decay, consistent with real earthquakes.
One formulation sums several sinusoidal components with frequencies clustered around ω_n, with random phase angles to avoid unrealistic periodicity. Multiply the signal by a smooth window (e.g., Hanning or Gaussian) to control duration and amplitude evolution.
Alternatively, synthesize near-fault velocity pulses with periods tuned to the structural period. These pulses deliver concentrated energy in one or few cycles that match the natural mode, effectively driving resonance—especially relevant for long-period structures with directivity effects.
A third method uses decaying sinusoids to replicate shock/response spectra. By adjusting decay constants and frequency spacing, you can reproduce a target spectral shape with a pronounced peak at T_n, enabling precise control of spectral acceleration demand.
After constructing the accelerogram, scale it so the computed spectral acceleration at T_n matches a target level (e.g., 0.8g at 5% damping). This ensures the motion imposes the intended structural demand.
Validate by simulating the SDOF response via numerical integration (Duhamel integral or direct time-stepping). Inspect displacement, velocity, and acceleration histories to confirm effective resonance. Refine bandwidth, envelope, and amplitude if needed.
Damping moderates response growth: even at resonance, 5% critical damping prevents unbounded amplification, though peak responses remain significantly higher than off-resonance cases.
From a design perspective, resonance-oriented accelerograms are valuable for sensitivity studies, fragility analysis, and model validation, enabling exploration of worst-case scenarios driven by frequency alignment.
In summary: (1) identify the natural frequency of the SDOF, (2) shape input with concentrated energy around that frequency, (3) apply realistic temporal envelopes, (4) scale to meet target spectral demands, and (5) validate through dynamic analysis. This systematic process provides controlled investigation of resonance effects in structural systems.
--------------------------------------------------
This Python code generates artificial earthquake ground motions specifically
 designed to induce resonance in single-degree-of-freedom (SDOF) structures by
 creating frequency content concentrated around the system's natural frequency 
 (0.1s period, 10Hz). The enhanced version introduces sophisticated temporal
 envelope design options including compound rise-plateau-decay envelopes,
 Gaussian shapes, asymmetric pulses, and empirical models to realistically 
 simulate different earthquake characteristics while maintaining resonance-oriented
 spectral energy. The algorithm combines narrow-band harmonic components clustered
 around the natural frequency with filtered broadband noise, applies the selected
 temporal envelope for amplitude modulation, and iteratively scales the accelerogram
 to achieve a target spectral acceleration of 0.8g at the natural period. The code produces
 comprehensive visualizations of ground motion, frequency content, and structural response
 while saving the results in multiple formats for further seismic analysis applications.
------------------------------------------------------------------
Key Enhancements Added:
1. Multiple Envelope Types
Simple: Original squared sine envelope
Hanning: Cosine-based smooth envelope
Gaussian: Bell-shaped energy concentration
Compound: Customizable rise-plateau-decay phases
Asymmetric: Fast rise, slow decay (pulse-like)
Empirical: Based on real earthquake characteristics

2. Parameterized Envelope Design
Customizable rise time, plateau duration, and decay time
Adjustable Gaussian width parameter
Normalized amplitude (max = 1)

3. Enhanced Frequency Content
Gaussian-weighted amplitudes for frequency components
Better frequency clustering around natural frequency
Increased number of harmonic components

4. Comprehensive Visualization
Separate envelope plot
Six-panel comprehensive response analysis
Frequency content visualization
Phase plane plot
Comparison of different envelope types

5. Improved Output
Detailed printout of all parameters
Multiple file outputs for each envelope type
Consistent naming conventions

6. Temporal Envelope Design Features
Each envelope type simulates different real-world scenarios:
Compound: Simulates typical earthquake phases (build-up, strong motion, decay)
Gaussian: Useful for isolated resonant pulses
Asymmetric: Represents near-fault pulse-like motions
Empirical: Based on statistical analysis of real earthquakes

This enhanced code provides a more realistic and flexible framework
 for generating resonance-oriented accelerograms with sophisticated 
 temporal envelope control, making it suitable for advanced seismic
 analysis and research applications.
"""
# PAPER: An investigation on the maximum earthquake input energy for elastic SDOF systems
'https://www.researchgate.net/publication/332766402_An_investigation_on_the_maximum_earthquake_input_energy_for_elastic_SDOF_systems'
# PAPER: SP-240—2 The Nature of Wind Loads and Dynamic Response
'https://www.researchgate.net/publication/268061536_SP-240-2_The_Nature_of_Wind_Loads_and_Dynamic_Response?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6Il9kaXJlY3QiLCJwYWdlIjoiX2RpcmVjdCJ9fQ'
# PAPER: A direct derivation method for acceleration and displacement floor response spectra 
'https://www.sciencedirect.com/science/article/pii/S0141029625008715?via%3Dihub'
# PAPER: New Formulation for Dynamic Analysis of Nonlinear Time-History of Vibrations of Structures under Earthquake Loading
'https://ceej.tabrizu.ac.ir/jufile?ar_sfile=388013&lang=en&utm_source=copilot.com'
# PAPER: Shock-Response Spectrum
'https://link.springer.com/chapter/10.1007/978-3-662-08587-5_10'
#%% ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy.integrate import odeint
from scipy.special import erf

# Target SDOF parameters
Tn = 0.1                  # [s] Natural period
fn = 1.0 / Tn             # [Hz] Natural frequency
wn = 2.0 * np.pi * fn     # [rad/s] Circular frequency
zeta = 0.05               # Damping ratio (5%)

# Time setting
Td = 20.0             # [s] total duration
dt = 0.005            # [s] Time step
Sa_target = 0.8 * 9.81  # target 0.8g at Tn

# Temporal envelope parameters
envelope_type = "compound"  # Options: "simple", "hanning", "gaussian", "compound", "asymmetric"
t_rise = 2.0               # [s] Rise time for compound envelope
t_plateau = 12.0           # [s] Plateau duration for compound envelope
t_decay = 6.0              # [s] Decay duration for compound envelope
sigma = 0.15               # Standard deviation for Gaussian envelope (as fraction of Td)

#%% ------------------------------------------------------------
def design_temporal_envelope(t, envelope_type="compound", Td=None, 
                            t_rise=2.0, t_plateau=12.0, t_decay=6.0, sigma=0.15):
    """
    Design various types of temporal envelopes for ground motion.
    
    Parameters:
    -----------
    t : numpy array
        Time vector
    envelope_type : str
        Type of envelope: 'simple', 'hanning', 'gaussian', 'compound', 'asymmetric'
    Td : float
        Total duration (used for some envelope types)
    t_rise : float
        Rise time for compound envelope [s]
    t_plateau : float
        Plateau duration for compound envelope [s]
    t_decay : float
        Decay duration for compound envelope [s]
    sigma : float
        Standard deviation for Gaussian envelope (as fraction of Td)
    
    Returns:
    --------
    E : numpy array
        Temporal envelope values
    """
    
    if Td is None:
        Td = t[-1] + dt
    
    if envelope_type == "simple":
        # Simple squared sine envelope (from original code)
        E = np.sin(np.pi * t / Td)**2
        envelope_name = "Simple Squared Sine"
        
    elif envelope_type == "hanning":
        # Hanning window envelope
        E = 0.5 * (1 - np.cos(2 * np.pi * t / Td))
        envelope_name = "Hanning Window"
        
    elif envelope_type == "gaussian":
        # Gaussian envelope centered at Td/2
        t_center = Td / 2
        sigma_time = sigma * Td
        E = np.exp(-0.5 * ((t - t_center) / sigma_time)**2)
        envelope_name = "Gaussian"
        
    elif envelope_type == "compound":
        # Compound envelope with rise, plateau, and decay phases
        E = np.zeros_like(t)
        
        # Rise phase (0 to t_rise)
        mask_rise = t <= t_rise
        E[mask_rise] = (t[mask_rise] / t_rise)**2
        
        # Plateau phase (t_rise to t_rise + t_plateau)
        mask_plateau = (t > t_rise) & (t <= t_rise + t_plateau)
        E[mask_plateau] = 1.0
        
        # Decay phase (t_rise + t_plateau to end)
        mask_decay = t > t_rise + t_plateau
        if np.any(mask_decay):
            decay_start = t_rise + t_plateau
            decay_duration = Td - decay_start
            t_decay_local = np.minimum(t_decay, decay_duration)
            decay_time = t[mask_decay] - decay_start
            E[mask_decay] = np.exp(-3.0 * decay_time / t_decay_local)
        
        envelope_name = "Compound (Rise-Plateau-Decay)"
        
    elif envelope_type == "asymmetric":
        # Asymmetric envelope with different rise and decay rates
        t_peak = 0.3 * Td  # Peak at 30% of duration
        E = np.zeros_like(t)
        
        # Rise phase (quadratic)
        mask_rise = t <= t_peak
        E[mask_rise] = (t[mask_rise] / t_peak)**1.5
        
        # Decay phase (exponential)
        mask_decay = t > t_peak
        decay_param = 0.1 * Td
        E[mask_decay] = np.exp(-(t[mask_decay] - t_peak) / decay_param)
        
        envelope_name = "Asymmetric (Fast Rise, Slow Decay)"
        
    elif envelope_type == "empirical":
        # Empirical envelope based on real earthquake characteristics
        t0 = 0.1 * Td  # Start of strong motion
        t1 = 0.6 * Td  # End of strong motion
        
        E = np.zeros_like(t)
        
        # Build-up phase
        mask_build = t < t0
        E[mask_build] = 0.5 * (1 - np.cos(np.pi * t[mask_build] / t0))
        
        # Strong motion phase
        mask_strong = (t >= t0) & (t <= t1)
        E[mask_strong] = 1.0
        
        # Decay phase
        mask_decay = t > t1
        decay_param = 0.2 * Td
        E[mask_decay] = np.exp(-(t[mask_decay] - t1) / decay_param)
        
        envelope_name = "Empirical (Build-Strong-Decay)"
        
    else:
        raise ValueError(f"Unknown envelope type: {envelope_type}")
    
    # Ensure envelope is normalized (max value = 1)
    if np.max(E) > 0:
        E = E / np.max(E)
    
    return E, envelope_name

#%% ------------------------------------------------------------
def ARTIFICIAL_GROUND_MOTION(Tn, wn, zeta, Td, dt, Sa_target, 
                            envelope_type="compound", t_rise=2.0, 
                            t_plateau=12.0, t_decay=6.0, sigma=0.15):
    """
    Generate artificial ground motion with resonance-oriented frequency content
    and sophisticated temporal envelope design.
    
    Parameters:
    -----------
    Tn : float
        Natural period [s]
    wn : float
        Natural circular frequency [rad/s]
    zeta : float
        Damping ratio
    Td : float
        Total duration [s]
    dt : float
        Time step [s]
    Sa_target : float
        Target spectral acceleration [m/s²]
    envelope_type : str
        Type of temporal envelope
    t_rise, t_plateau, t_decay : float
        Parameters for compound envelope [s]
    sigma : float
        Parameter for Gaussian envelope
    """
    
    # Time Discretization
    t = np.arange(0, Td, dt)
    
    # Design Temporal Envelope
    E, envelope_name = design_temporal_envelope(
        t, envelope_type, Td, t_rise, t_plateau, t_decay, sigma
    )
    
    # Plot envelope
    plt.figure(figsize=(10, 4))
    plt.plot(t, E, 'b-', linewidth=2)
    plt.fill_between(t, 0, E, alpha=0.3)
    plt.title(f'Temporal Envelope: {envelope_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Time [s]', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, Td])
    plt.ylim([0, 1.1])
    
    # Annotate envelope parameters
    if envelope_type == "compound":
        plt.axvline(t_rise, color='r', linestyle='--', alpha=0.7, label=f'Rise end: {t_rise}s')
        plt.axvline(t_rise + t_plateau, color='g', linestyle='--', alpha=0.7, 
                   label=f'Decay start: {t_rise+t_plateau}s')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Narrow-Band Resonant Ground Motion 
    np.random.seed(42)  # For reproducibility
    Ncomp = 8  # Increased number of components for richer frequency content
    band = 0.12 * wn  # +/-12% around wn for narrow band
    
    # Generate frequencies clustered around natural frequency
    omegas = wn + (2*np.random.rand(Ncomp)-1) * band
    
    # Adjust amplitudes to create a peak at natural frequency
    # Amplitude follows Gaussian distribution centered at wn
    freq_deviation = np.abs(omegas - wn) / band
    amps = np.exp(-2 * freq_deviation**2) * (0.8 + 0.4*np.random.rand(Ncomp))
    
    # Random phases for each component
    phis = 2*np.pi*np.random.rand(Ncomp)
    
    # Generate narrow-band signal
    ag_narrow = np.zeros_like(t)
    for w, phi, A in zip(omegas, phis, amps):
        ag_narrow += A * np.sin(w*t + phi)
    
    # Apply temporal envelope
    ag_narrow = ag_narrow * E
    
    # Add Filtered Broadband Noise for realism
    wb_noise = 0.25 * np.random.randn(len(t))
    
    # Create bandpass filter around natural frequency
    wn_nyq = np.pi / dt  # Nyquist frequency
    low_cut = 0.6 * wn / wn_nyq
    high_cut = 1.4 * wn / wn_nyq
    sos = butter(4, [low_cut, high_cut], btype='band', output='sos')
    wb_filt = sosfiltfilt(sos, wb_noise)
    
    # Scale broadband component and apply envelope
    ag_broadband = 0.35 * wb_filt * E
    
    # Combine components
    ag = ag_narrow + ag_broadband
    
    # SDOF relative motion equation for scaling
    def sdof(state, tau):
        z, zd = state
        ag_t = np.interp(tau, t, ag)
        zdd = -2*zeta*wn*zd - wn**2*z - ag_t
        return [zd, zdd]
    
    # Initial scaling iteration
    state0 = [0.0, 0.0]
    sol = odeint(sdof, state0, t)
    z = sol[:,0]
    zd = sol[:,1]
    zdd = np.gradient(zd, dt)
    aa = zdd + ag  # Absolute acceleration
    Sa_current = np.max(np.abs(aa))
    
    # Iterative scaling to target Sa
    scale = Sa_target / (Sa_current + 1e-9)
    ag *= scale
    
    # Recompute response after scaling
    sol = odeint(sdof, state0, t)
    z = sol[:,0]
    zd = sol[:,1]
    zdd = np.gradient(zd, dt)
    aa = zdd + ag
    
    # Compute additional response metrics
    velocity = np.gradient(z, dt)  # Relative velocity
    abs_velocity = velocity + np.cumsum(ag) * dt  # Approximate absolute velocity
    
    # Frequency analysis
    from scipy.signal import welch
    frequencies, psd = welch(ag, fs=1/dt, nperseg=min(2048, len(ag)))
    
    # Print results
    print("="*60)
    print(f"RESONANCE-ORIENTED ACCELEROGRAM GENERATION")
    print("="*60)
    print(f"Target SDOF Parameters:")
    print(f"  Natural Period Tn = {Tn:.3f} s")
    print(f"  Natural Frequency fn = {fn:.2f} Hz")
    print(f"  Damping Ratio ζ = {zeta:.3f}")
    print(f"  Target Spectral Acceleration = {Sa_target/9.81:.2f} g")
    print(f"\nTemporal Envelope: {envelope_name}")
    if envelope_type == "compound":
        print(f"  Rise time: {t_rise:.1f} s")
        print(f"  Plateau duration: {t_plateau:.1f} s")
        print(f"  Decay time: {t_decay:.1f} s")
    print(f"\nGenerated Motion Characteristics:")
    print(f"  Duration: {Td:.1f} s")
    print(f"  Time step: {dt:.3f} s")
    print(f"  PGA: {np.max(np.abs(ag))/9.81:.3f} g")
    print(f"  PGV: {np.max(np.abs(abs_velocity)):.3f} m/s")
    print(f"  PGD: {np.max(np.abs(z)):.4f} m")
    print(f"  Achieved Sa(Tn): {np.max(np.abs(aa))/9.81:.3f} g")
    print(f"  Scaling factor applied: {scale:.3f}")
    print("="*60)
    
    # Comprehensive plotting
    fig = plt.figure(figsize=(14, 10))
    
    # Plot 1: Ground acceleration with envelope
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(t, ag/9.81, 'k-', linewidth=1, label='Acceleration')
    ax1.plot(t, E * np.max(ag)/9.81, 'r--', linewidth=1.5, alpha=0.7, label='Envelope')
    ax1.set_ylabel('Ground Acceleration [g]', fontsize=11)
    ax1.set_title(f'Ground Motion Acceleration ({envelope_name})', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    ax1.set_xlim([0, Td])
    
    # Plot 2: Frequency content
    ax2 = plt.subplot(3, 2, 2)
    ax2.semilogy(frequencies, psd, 'b-', linewidth=1.5)
    ax2.axvline(x=fn, color='r', linestyle='--', linewidth=2, 
               label=f'Natural freq: {fn:.2f} Hz')
    ax2.axvspan(0.9*fn, 1.1*fn, alpha=0.2, color='red', label='±10% band')
    ax2.set_xlabel('Frequency [Hz]', fontsize=11)
    ax2.set_ylabel('Power Spectral Density', fontsize=11)
    ax2.set_title('Frequency Content of Ground Motion', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim([0, 5*fn])
    
    # Plot 3: Displacement response
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(t, z*1000, 'b-', linewidth=1.5)  # Convert to mm
    ax3.set_ylabel('Relative Displacement [mm]', fontsize=11)
    ax3.set_title('SDOF Displacement Response', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, Td])
    
    # Plot 4: Absolute acceleration response
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(t, aa/9.81, 'r-', linewidth=1.5)
    ax4.set_ylabel('Absolute Acceleration [g]', fontsize=11)
    ax4.set_title('SDOF Absolute Acceleration Response', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, Td])
    
    # Plot 5: Velocity response
    ax5 = plt.subplot(3, 2, 5)
    ax5.plot(t, velocity, 'g-', linewidth=1.5, label='Relative')
    ax5.plot(t, abs_velocity, 'm--', linewidth=1.5, alpha=0.7, label='Absolute')
    ax5.set_xlabel('Time [s]', fontsize=11)
    ax5.set_ylabel('Velocity [m/s]', fontsize=11)
    ax5.set_title('SDOF Velocity Response', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.set_xlim([0, Td])
    
    # Plot 6: Phase plane (velocity vs displacement)
    ax6 = plt.subplot(3, 2, 6)
    ax6.plot(z*1000, velocity, 'b-', linewidth=1)
    ax6.set_xlabel('Displacement [mm]', fontsize=11)
    ax6.set_ylabel('Velocity [m/s]', fontsize=11)
    ax6.set_title('Phase Plane Plot', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Save data
    save_data(t, ag, E, envelope_type)
    
    return t, ag, E, z, aa

#%% ------------------------------------------------------------
def save_data(t, ag, E, envelope_type):
    """Save generated data to files."""
    
    # Save acceleration time history
    np.savetxt(
        f"artificial_accel_{envelope_type}.txt",
        np.column_stack((t, ag)),
        header="Time[s] Acceleration[m/s^2]"
    )
    
    # Save envelope
    np.savetxt(
        f"envelope_{envelope_type}.txt",
        np.column_stack((t, E)),
        header="Time[s] Envelope_Amplitude"
    )
    
    # Save in AT2 format
    def write_at2(filename, acc, dt):
        with open(filename, "w") as f:
            f.write("Artificial ground motion - resonance oriented\n")
            f.write(f"Envelope type: {envelope_type}\n")
            f.write(f"Generated by Python - Resonance SDOF Analysis\n")
            f.write(f"NPTS= {len(acc)}, DT= {dt}\n")
            
            count = 0
            for a in acc:
                f.write(f"{a:12.6e}")
                count += 1
                if count % 5 == 0:
                    f.write("\n")
            if count % 5 != 0:
                f.write("\n")
    
    write_at2(f"artificial_resonance_{envelope_type}.at2", ag, dt)
    print(f"\nData saved to files with prefix: artificial_resonance_{envelope_type}")

#%% ------------------------------------------------------------
# Run with different envelope types
envelope_types = ["simple", "compound", "gaussian", "asymmetric"]

# Test all envelope types
for env_type in envelope_types:
    print(f"\n{'='*80}")
    print(f"GENERATING WITH {env_type.upper()} ENVELOPE")
    print('='*80)
    
    t, ag, E, z, aa = ARTIFICIAL_GROUND_MOTION(
        Tn=Tn,
        wn=wn,
        zeta=zeta,
        Td=Td,
        dt=dt,
        Sa_target=Sa_target,
        envelope_type=env_type,
        t_rise=t_rise,
        t_plateau=t_plateau,
        t_decay=t_decay,
        sigma=sigma
    )

# Generate summary comparison
print("\n" + "="*80)
print("SUMMARY COMPARISON OF DIFFERENT ENVELOPE TYPES")
print("="*80)
print("Envelope Type            | Key Characteristics")
print("-"*80)
print("Simple                   | Smooth, symmetric, zero at boundaries")
print("Hanning                  | Cosine-based, smooth rise and fall")
print("Gaussian                 | Bell-shaped, concentrated energy")
print("Compound                 | Customizable rise, plateau, decay phases")
print("Asymmetric               | Fast rise, slow decay (pulse-like)")
print("Empirical                | Based on real earthquake observations")
print("="*80)