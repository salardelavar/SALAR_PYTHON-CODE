This Python script identifies the unknown parameters of a damped free-vibration system specifically the damping ratio (ξ), natural period (T), initial displacement (x₀), and initial velocity (v₀) by fitting an analytical displacement function to four experimental data points. It defines a nonlinear system of equations representing the difference between the theoretical displacements and the measured values at given time instants, then employs the Newton–Raphson method (via SciPy's [root] solver with the 'hybr' algorithm) to iteratively refine an initial guess until convergence is achieved. Once the optimal parameters are obtained, the script computes the analytical response over a fine time grid, evaluates the Root Mean Square Error (RMSE) to quantify the fitting accuracy, and generates a plot that overlays the smooth fitted displacement curve with the experimental data points and the fitted values at those specific times for visual validation.

![alt text](https://github.com/salardelavar/SALAR_PYTHON-CODE/blob/main/PARAMETER_IDENTIFICATION_OF_A_DAMPED_FREE_VIBRATION_SYSTEM/COVER_RESULT_PARAMETER_IDENTIFICATION_OF_A_DAMPED_FREE_VIBRATION_SYSTEM.png) 

![alt text](https://github.com/salardelavar/SALAR_PYTHON-CODE/blob/main/PARAMETER_IDENTIFICATION_OF_A_DAMPED_FREE_VIBRATION_SYSTEM/DOC_PARAMETER_IDENTIFICATION_OF_A_DAMPED_FREE_VIBRATION_SYSTEM.png) 

 
THIS PYTHON SCRIPT IS WRITTEN BY SALAR DELAVAR GHASHGHAEI (QASHQAI)
