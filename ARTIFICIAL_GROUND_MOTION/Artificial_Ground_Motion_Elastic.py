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
Important Notes:
Code Objective: To generate an artificial ground acceleration record that produces a spectral acceleration of 0.8g at the target natural frequency (Tn=0.1s).
Generation Method: Combination of narrow-band harmonic components around the natural frequency with a filtered broadband component.
Scaling: The generated record is iteratively scaled to achieve the target spectral acceleration.
Output: The ground acceleration record is saved in various formats (text file and AT2 format).
Key Feature: This method produces a "resonance-oriented" record at the target frequency, meaning its frequency content is concentrated around the system's natural frequency.
Resonance Principle: By tuning the frequency content to match the structure's natural frequency, this method creates ground motions that are particularly effective at exciting the target mode of vibration, which is important for seismic testing and analysis.
Applications: Useful for generating site-specific ground motions, performing parametric studies, and creating input motions for seismic analysis where resonance effects are of interest.
"""
# PAPER: An investigation on the maximum earthquake input energy for elastic SDOF systems
'https://www.researchgate.net/publication/332766402_An_investigation_on_the_maximum_earthquake_input_energy_for_elastic_SDOF_systems'
# PAPER: SP-240—2 The Nature of Wind Loads and Dynamic Response
'https://www.researchgate.net/publication/268061536_SP-240-2_The_Nature_of_Wind_Loads_and_Dynamic_Response?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6Il9kaXJlY3QiLCJwYWdlIjoiX2RpcmVjdCJ9fQ'
# PAPER: A direct derivation method for acceleration and displacement floor response spectra 
'https://www.sciencedirect.com/science/article/pii/S0141029625008715?via%3Dihub'
# PAPER: New Formulation for Dynamic Analysis of Nonlinear Time-History of Vibrations of Structures under Earthquake Loading
'https://ceej.tabrizu.ac.ir/jufile?ar_sfile=388013&lang=en'
# PAPER: Shock-Response Spectrum
'https://link.springer.com/chapter/10.1007/978-3-662-08587-5_10'
#%% ------------------------------------------------------------
import numpy as np
# Target SDOF parameters
Tn   = 0.1                  # [s] Natural period
fn   = 1.0 / Tn             # [Hz] Natural frequency
wn   = 2.0 * np.pi * fn     # [rad/s] Circular frequency
zeta = 0.05                 # Damping ratio (5%)

# Time setting
Td = 20.0                   # [s] total duration
dt = 0.005                  # [s] Time step

Sa_target  = 0.8 * 9.81  # target 0.8g at Tn
#%% ------------------------------------------------------------
def ARTIFICIAL_GROUND_MOTION(Tn, wn, zeta, Td, dt, Sa_target):
    import matplotlib.pyplot as plt
    from scipy.signal import butter, sosfiltfilt
    from scipy.integrate import odeint
    import numpy as np
    # Time Discretization
    t = np.arange(0, Td, dt)
    # Ground Motion Envelope
    E = np.sin(np.pi * t / Td)**2
    
    # Narrow-Band Resonant Ground Motion 
    #np.random.seed(7)
    Ncomp = 5
    band = 0.10 * wn  # +/-10% around wn
    omegas = wn + (2*np.random.rand(Ncomp)-1) * band
    phis   = 2*np.pi*np.random.rand(Ncomp)
    amps   = np.ones(Ncomp)
    
    ag = np.zeros_like(t)
    for w,phi,A in zip(omegas, phis, amps):
        ag += A * np.sin(w*t + phi)
    ag = ag * E
    
    # Add Filtered Broadband Noise (Optional: small broadband component filtered around wn)
    wb_noise = 0.2*np.random.randn(len(t))
    sos = butter(4, [0.7*wn/(np.pi/dt), 1.3*wn/(np.pi/dt)], btype='band', output='sos')
    wb_filt = sosfiltfilt(sos, wb_noise)
    ag += 0.3*wb_filt
    
    # Scale accelerogram to target Sa at Tn (iterative quick heuristic)
    # SDOF relative motion: z'' + 2*zeta*wn*z' + wn^2*z = -ug'' ; here ug'' = ag
    # Linear SDOF Model (for scaling to target Sa)
    def sdof(state, tau):
        z, zd = state
        ag_t = np.interp(tau, t, ag)
        zdd = -2*zeta*wn*zd - wn**2*z - ag_t
        return [zd, zdd]
    
    state0 = [0.0, 0.0]
    sol = odeint(sdof, state0, t)
    z = sol[:,0]
    zd = sol[:,1]
    zdd = np.gradient(zd, dt)
    
    # Absolute acceleration response (at mass)
    aa = zdd + ag
    
    Sa_current = np.max(np.abs(aa))
    
    scale = Sa_target / (Sa_current + 1e-9)
    ag *= scale
    
    # Recompute response after scaling
    sol = odeint(sdof, state0, t)
    z = sol[:,0]
    zd = sol[:,1]
    zdd = np.gradient(zd, dt)
    aa = zdd + ag
    
    print(f"Sa(Tn≈{Tn:.2f}s, ζ={zeta:.2f}) ≈ {np.max(np.abs(aa))/9.81:.2f} g")
    
    # Time History Plots
    fig, axs = plt.subplots(3,1, figsize=(9,7), sharex=True)
    axs[0].plot(t, ag/9.81, 'k')
    axs[0].set_ylabel('ag [g]')
    axs[0].grid(True)
    axs[1].plot(t, z, 'b')
    axs[1].set_ylabel('z [m]')
    axs[1].grid(True)
    axs[2].plot(t, aa/9.81, 'r')
    axs[2].set_ylabel('abs acc [g]')
    axs[2].set_xlabel('time [s]')
    axs[2].grid(True)
    plt.tight_layout()
    plt.show()
    
    np.savetxt("artificial_accel_elastic.txt", np.column_stack((t, ag)))
    np.savetxt(
        "artificial_ground_motion_Elastic.txt",
        np.column_stack((t, ag)),
        header="Time(s)   Acceleration(m/s^2)"
    )
    
    def write_at2(filename, acc, dt):
        with open(filename, "w") as f:
            f.write("Artificial ground motion - resonance oriented\n")
            f.write("Generated by Python\n")
            f.write(f"NPTS= {len(acc)}, DT= {dt}\n")
            
            count = 0
            for a in acc:
                f.write(f"{a:12.6e}")
                count += 1
                if count % 5 == 0:
                    f.write("\n")
            if count % 5 != 0:
                f.write("\n")
    
    write_at2("artificial_resonance.at2", ag, dt)
#%% ------------------------------------------------------------

ARTIFICIAL_GROUND_MOTION(Tn, wn, zeta, Td, dt, Sa_target)
