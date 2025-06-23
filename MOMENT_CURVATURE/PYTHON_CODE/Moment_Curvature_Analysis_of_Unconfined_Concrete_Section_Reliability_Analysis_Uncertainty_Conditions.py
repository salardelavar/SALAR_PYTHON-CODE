"""
#***********************************************************#
#                >> IN THE NAME OF ALLAH <<                 #
#  Moment-Curvature analysis of Unconfined concrete section #
#  With Axial Load effect in Uncertainty Conditions         #
#           Reliability Analysis Monte-Carlo Method         #
#-----------------------------------------------------------#
#     This program is written by salar Delavar Qashqai      #  
#          E-mail:salar.d.ghashghaei@gmail.com              #
#-----------------------------------------------------------#
#Unit: Newton-Milimeter                                     #
#Given:Section Properties , Concrete properties ,           #
# Reinforcing steel properties                              #
#Calculate: Moment-Curavture                                #
# Note: No limit for accounting plurality steel rebar       #
# Newton-Raphson Method : Tangent procedure                 #
#***********************************************************#
#   _    ______________________________________             #
#   |   |                                      |            #
#       |     #     #     #     #    #    #    |            #
#       |     #                           #    |            #
#   b   |    As1   As2   As3   As4  As5  As6   |            #
#       |     #                           #    |            #
#   |   |     #     #     #     #    #    #    |            #
#   _   |______________________________________|            #
#       |<-                 h                ->|            #
#       |<-d1->|                                            #
#       |<-  d2   ->|                                       #
#       |<-     d3      ->|                                 #
#       |<-        d4          ->|                          #
#       |<-            d5          ->|                      #
#       |<-               d6             >|                 #
#    X                                                      #
#    ^                                                      #
#    |             (Moment - Curvature along X axis)        #
#    |                                                      #
#    +----> Y                                               #
#***********************************************************#
"""
import os
os.system('cls' if os.name == 'nt' else 'clear')
import numpy as np
import time
#%%--------------------------------------------------------------------
def AXIAL_LOAD_ANALYSIS(J, Ptarget,As, d, fc ,ecu, Ec, ec0, fct, ect1, fy, Es, fu, ey, esh, esu, Esh, N, itermax, tolerance):
    An = len(As)
    R=(1/N)
    c=np.array([(.5*R+R*(k-1))*h for k in range(1,N+1)])
    fsi = np.zeros(An)
    fstani = np.zeros(An)
    Cci = np.zeros(N)
    Cctani = np.zeros(N)
    Fsi = np.zeros(An)
    Fstani = np.zeros(An)
    if abs(Ptarget) > 0:
        it = 0 # initialize iteration count
        residual = 100 # initialize residual
        eci = 1e-12 
        while residual > tolerance:
                for u in range(An):
                    # As
                    if eci > 0 and eci < ey:
                        fsi[u] = Es * eci
                        fstani[u] = Es
                    elif eci < 0 and eci > -ey:
                        fsi[u] = Es * eci
                        fstani[u] = -Es
                    elif eci >= ey and eci < esh:
                        fsi[u] = fy
                        fstani[u] = 0
                    elif eci <= -ey and eci > -esh:
                        fsi[u] = -fy
                        fstani[u] = 0
                    elif eci >= esh and eci < esu:
                        fsi[u] = fy + Esh * (abs(eci) - esh)
                        fstani[u] = Esh
                    elif eci <= -esh and eci > -esu:
                        fsi[u] = -fy - Esh * (abs(eci) - esh)
                        fstani[u] = -Esh
                    elif eci >= esu or eci <= -esu:
                        fsi[u] = 0
                        fstani[u] = 0

                    Fsi[u] = As[u] * fsi[u]
                    Fstani[u] = As[u] * fstani[u] # tangent steel force

                if eci > 0 and eci < ec0: # in this step: Unconfined concrete force in rebar area is omitted (F=As*fc
                        for z in range(N):  # in this step: concrete force for each fiber is calculated
                            # -------------- Cc --------------%
                            if eci > 0 and eci < ec0:
                                Ci = fc * ((2 * eci / ec0) - (eci / ec0) ** 2)
                                Ctani = fc * ((2 / ec0) - ((2 * eci) / ec0 ** 2))
                            elif eci >= ec0 and eci < ecu:
                                Ci = fc * (1 - (0.15 * (eci - ec0) / (ecu - ec0)))
                                Ctani = -3 * fc / (20 * (ecu - ec0))
                            elif eci >= ecu:
                                Ci = 0
                                Ctani = 0
                            elif eci < 0 and eci >= ect1:
                                Ci = 0.5 * Ec * eci
                                Ctani = 0.5 * Ec
                            elif eci < ect1 and eci >= ect2:
                                Ci = fct - (0.5 * fct / (ect2 - ect1)) * (eci - ect1)
                                Ctani = -(0.5 * fct / (ect2 - ect1))
                            elif eci < ect2 and eci >= ect3:
                                Ci = .5 * fct - (0.5 * fct / (ect3 - ect2)) * (eci - ect2)
                                Ctani = -(0.5 * fct / (ect3 - ect2))
                            elif eci < ect3:
                                Ci = 0
                                Ctani = 0

                            Cci[z] = b * R * h * Ci
                            Cctani[z] = b * R * h * Ctani  # tangent concrete force

                # ----------------------------------%
                FsTOTAL = sum(Fsi)
                CcTOTAL = sum(Cci)
                Ai = CcTOTAL + FsTOTAL - Ptarget
                FsTOTAL_tan = sum(Fstani)
                CcTOTAL_tan = sum(Cctani)
                A_tani = CcTOTAL_tan + FsTOTAL_tan
                dxi = (-Ai) / A_tani
                residual = abs(dxi)  # evaluate residual
                it += 1  # increment iteration count
                eci += dxi  # update x
                if it == itermax:  # stop the analysis of this step please of Convergence
                    print(f'(-) trail iteration reached to Ultimate {it} - strain: {eci:.6e} - error: [{residual:.2e}]')
                    print(' ## The solution for this step is not converged. Please check your model ##')

                if residual < tolerance: # iteration control
                    print(f'(+)Random Number {J+1} Axial Load effect: It is converged in {it} iterations - Initial axial strain: {eci:.6e}\n\n')
                    break
                else:
                    eci = 0
    return eci  
#%%--------------------------------------------------------------------
def MOMENT_CURVATURE_ANALYSIS(ECI, Ptarget, As, d, fc ,ecu, Ec, ec0, fct, ect1, fy, Es, fu, ey, esh, esu, Esh, N, itermax, tolerance, x):
    # Newton Method Procedure
    An = len(As)
    R=(1/N)
    c=np.array([(.5*R+R*(k-1))*h for k in range(1,N+1)])
    EC=np.array([.1*abs(ect1), .2*abs(ect1), .4*abs(ect1), .6*abs(ect1), .8*abs(ect1), abs(ect1), abs(ect2), .33*abs(ect3), .67*abs(ect3), .8*abs(ect3), .9*abs(ect3), abs(ect3), .4*ecu, .5*ecu, .6*ecu, .7*ecu, .8*ecu, .9*ecu, ecu])
    q=len(EC)
    fs = np.zeros(An)
    fstan = np.zeros(An)
    CS = np.zeros(An)
    CD = np.zeros(N)
    Cc = np.zeros(N)
    Cctan = np.zeros(N)
    CtanS = np.zeros(N)
    Fss = np.zeros(An)
    Fstans = np.zeros(An)
    TSCSs = np.zeros(q)
    BSCSs = np.zeros(q)
    TSCS = np.zeros(q)
    BSCS = np.zeros(q)
    ECCC = np.zeros(q)
    TUCS = np.zeros(q)
    Cur = np.zeros(q)
    Mom = np.zeros(q)
    CUR = np.zeros(q)
    XX = np.zeros(q)
    CrackDepth = np.zeros(q)
    #print('======================================================================================== ')
    #print('   Increment      Iterations      Strain      Neutral-Axis      Curvature      Moment    ')
    #print('======================================================================================== ')
    for j in range(q):
        eC = ECI + EC[j]
        it = 0 # initialize iteration count
        residual = 100 # initialize residual
        while (residual > tolerance):
            for u in range(An):
                es = eC * (x - d[u]) / x
                # As
                if es > 0 and es < ey:
                    fs[u] = Es * es
                    fstan[u] = (Es * eC * d[u]) / (x ** 2)
                elif es < 0 and es > -ey:
                    fs[u] = Es * es
                    fstan[u] = (Es * eC * d[u]) / (x ** 2)
                elif es >= ey and es < esh:
                    fs[u] = fy
                    fstan[u] = 0
                elif es <= -ey and es > -esh:
                    fs[u] = -fy
                    fstan[u] = 0
                elif es >= esh and es < esu:
                    fs[u] = fy + Esh * (abs(es) - esh)
                    fstan[u] = (Esh * eC * d[u]) / (x ** 2)
                elif es <= -esh and es > -esu:
                    fs[u] = -fy - Esh * (abs(es) - esh)
                    fstan[u] = (Esh * eC * d[u]) / (x ** 2)
                elif es >= esu or es <= -esu:
                    fs[u] = 0
                    fstan[u] = 0
                Fs = As * fs
                Fstan = As * fstan # tangent steel force
                if es > 0 and es < ec0: # in this step: Unconfined concrete force in rebar area is omitted (F=As*fc)
                    Cs = fc * ((2 * es / ec0) - (es / ec0) ** 2)
                    Ctans = ((2 * fc) / (ec0 ** 2 * x ** 3)) * (ec0 * eC * d[u] * x - d[u] * eC ** 2 * x + 2 * eC ** 2 * d[u] ** 2)
                elif es >= ec0 and es <= ecu:
                    Cs = fc * (1 - (0.15 * (es - ec0) / (ecu - ec0)))
                    Ctans = -(3 * eC * d[u] * fc) / (20 * (ecu - ec0) * x ** 2)
                elif es > ecu:
                    Cs = 0
                    Ctans = 0
                elif es < 0 and es >= ect1:
                    Cs = -0.5 * Ec * es
                    Ctans = -(0.5 * Ec * d[u] * eC) / x ** 2
                elif es < ect1 and es >= ect2:
                    Cs = fct + (0.5 * fct / (ect2 - ect1)) * (es - ect1)
                    Ctans = +(0.5 * fct * eC * d[u]) / ((ect2 - ect1) * x ** 2)
                elif es < ect2 and es >= ect3:
                    Cs = -( .5*fct-(0.5*fct/(ect3-ect2))*(es-ect2))
                    Ctans = +(0.5*fct*eC*d[u])/((ect3-ect2)*x**2)
                elif es < ect3:
                    Cs = 0
                    Ctans = 0

                CS[u] = Cs
                CtanS[u] = Ctans
                Fss[u] = -As[u] * CS[u]
                Fstans[u] = -As[u] * CtanS[u] # tangent Minus of concrete force  
            for z in range(N): # in this step: concrete force for each fiber is calculated  
                ec = eC * (x - c[z]) / x
                CD = np.zeros(N)
                if ec > 0 and ec < ec0:
                    C = fc * ((2 * ec / ec0) - (ec / ec0) ** 2)
                    Ctan = ((2 * fc) / (ec0 ** 2 * x ** 3)) * (ec0 * eC * c[z] * x - c[z] * eC ** 2 * x + 2 * eC ** 2 * c[z] ** 2)
                    CD[z] = 0
                elif ec >= ec0 and ec <= ecu:
                    C = fc * (1 - (0.15 * (ec - ec0) / (ecu - ec0)))
                    Ctan = -(3 * eC * c[z] * fc) / (20 * (ecu - ec0) * x ** 2)
                    CD[z] = 0
                elif ec > ecu:
                    C = 0
                    Ctan = 0
                    CD[z] = 0
                elif ec < 0 and ec >= ect1:
                    C = 0.5 * Ec * ec
                    Ctan = (0.5 * Ec * c[z] * eC) / x ** 2
                    CD[z] = 0
                elif ec < ect1 and ec >= ect2:
                    C = fct - (0.5 * fct / (ect2 - ect1)) * (ec - ect1)
                    Ctan = -(0.5 * fct * eC * c[z]) / ((ect2 - ect1) * x ** 2)
                    CD[z] = 0
                elif ec < ect2 and ec >= ect3:
                    C = .5*fct-(0.5*fct/(ect3-ect2))*(ec-ect2)
                    Ctan=-(0.5*fct*eC*c[z])/((ect3-ect2)*x**2)
                    CD[z]=0
                elif ec < ect3:
                    C = 0
                    Ctan = 0
                    CD[z] = h - c[z] # Crack Depth

                Cc[z] = b*R*h*C
                Cctan[z] = b*R*h*Ctan # tangent concrete force

            FsTOTAL = sum(Fs)
            CcTOTAL = sum(Cc)
            FssTOTAL = sum(Fss)
            A = CcTOTAL + FsTOTAL - Ptarget
            FsTOTAL_tan = sum(Fstan)
            CcTOTAL_tan = sum(Cctan)
            A_tan = CcTOTAL_tan + FsTOTAL_tan
            dx = A_tan ** -1 * (-A)
            residual = np.max(np.abs(dx)) # evaluate residual
            it += 1 # increment iteration count
            x += dx # update x
            if it == itermax: # stop the the analysis of this step please of Convergence
                #print(f'      {j+1}             {it} : trail iteration reached to Ultimate - strain: {eC:.6f} - error: [{A:.2f}]')
                #print('    ## The solution for this step is not converged. Please check your model ##')
                break

            if it == itermax:
                break # stop the analysis at all because last Convergence

            e = x - d # distance of each rebar from Neuteral axis
            cc = x - c # distance of each concrete fiber from Neuteral axis
            Pc1 = x - .5 * h# distance of Axial Load from Neuteral axis

        if residual < tolerance: # iteration control
            #print(f'      {j+1}             {it}            {eC:.6f}           {x:.2f}      {(eC/x)*1000:.5f}         {(np.dot(Fs,e)+np.dot(Fss,e)+np.dot(Cc,cc))*10**-6:.2f}')

            TSCSs[j] = eC * (x - d[0]) / x; TSCS[j] = fs[0] # Top Steel compression strain-stress
            BSCSs[j] = eC * (x - d[5]) / x; BSCS[j] = fs[5] # Bottom Steel compression strain-stress

            if eC > 0 and eC < ec0:
                ECCC[j] = eC; TUCS[j] = fc * ((2 * eC / ec0) - (eC / ec0) ** 2)# Concrete Strain-Stress
            elif eC >= ec0 and eC <= ecu:
                ECCC[j] = eC; TUCS[j] = fc * (1 - (0.15 * (eC - ec0) / (ecu - ec0)))# Concrete Strain-Stress

            if CD[z] == 0:
                CrackDepth[j] = 0
            else:
                CrackDepth[j] = max(abs(CD)) # Crack Depth of each increment

            # Calculate Moment and Curavture
            Cur[j] = (eC / x) * 1000;CUR[j] = Cur[j];XX[j] = x;
            Mom[j] = (np.dot(Fs,e) + np.dot(Fss,e) + np.dot(Cc,cc) + Pc1 * Ptarget)

    Cur = np.insert(Cur, 0, 0)
    Mom = np.insert(Mom, 0, 0)
    s = len(Cur)
    EI = np.zeros(s-1)
    for i in range(s-1):
        EI[i] = (Mom[i+1] - Mom[i]) / (Cur[i+1] - Cur[i]) # Flextural Rigidity

    if round(eC, 5) == ecu:
        print(f'\n      ## Unconfined Concrete Strain Reached to Ultimate Strain: {eC:.4f} ## \n\n')
    return CrackDepth, EC, XX, CUR, EI, Cur, Mom, TSCSs, TSCS, BSCSs, BSCS, ECCC, TUCS  
#%%--------------------------------------------------------------------
def BETA_PDF(MIN_X, MAX_X, a, b):
    import numpy as np
    return MIN_X + (MAX_X - MIN_X) * np.random.beta(a, b)
#%%--------------------------------------------------------------------
def Normal_CDF_Newton_Raphson(P_f, EPS=1e-3, tol=1e-6, max_iter=1000000):
    from scipy.stats import norm
    x = 0.0  # Initial guess (you can choose any value)
    
    for i in range(max_iter):
        xmin = x - EPS
        xmax = x + EPS
        f = norm.cdf(-x) - P_f
        fmin = norm.cdf(-xmin) - P_f
        fmax = norm.cdf(-xmax) - P_f
        df = (fmax - fmin) / (2 * EPS)
        dx = f / df
        f_prime_x = -norm.pdf(-x)
        
        if abs(dx) < tol:
            break
        
        x -= dx
    
    return x
#%%--------------------------------------------------------------------
def HISTOGRAM_BOXPLOT_PLOTLY( DATA, XLABEL='X', TITLE='A', COLOR='cyan'):
    # Plotting histogram and boxplot
    import plotly.express as px
    fig = px.histogram(x=DATA, marginal="box", color_discrete_sequence=[COLOR])
    fig.update_layout(title=TITLE, xaxis_title=XLABEL, yaxis_title="Frequency")
    fig.show()
    #fig = px.ecdf(irr, title=TITLE)
    #fig.show()
#%%--------------------------------------------------------------------
def MIX_HISTOGRAM(x, y, BINS, X, Y, TITLE):
    plt.figure(figsize=(8, 6))
    plt.hist(x, bins=BINS, alpha=0.5, label=X, color='blue')
    plt.hist(y, bins=BINS, alpha=0.5, label=Y, color='red')
    plt.legend(loc='upper right')
    plt.xlabel("Samples")
    plt.ylabel("Frequency")
    plt.title(TITLE)
    plt.show()
#%%--------------------------------------------------------------------
ShowText = "Moment_Curvature_Analysis_of_Unconfined_Concrete_Section_Reliability_Analysis_Uncertainty_Conditions-outputEXCEL.csv"
def OUTPUT_EXCEL(Ptarget, AxialStrain, Cur, Mom, CrackDepth, EI):
    with open(ShowText, "w") as OutputFile:
        OutputFile.write("           ### Moment Curvature Analysis of Unconfined Concrete Section Reliability Analysis Uncertainty Conditions ###\n")
        OutputFile.write(" Increment, Axial Load, Axial Strain, Curvature, Moment, Crack Depth, Flextural Rigidity-EI\n")
        for i in range(len(Ptarget)):
            OutputFile.write("%d,%e,%e,%e,%e,%e,%e\n" % (i + 1, Ptarget[i], AxialStrain[i], Cur[i], Mom[i], CrackDepth[i], EI[i]))
#%%--------------------------------------------------------------------
NUM_SIM = 2000 # Monte carlo Number of Simulations

# Initialize lists to store results
AxialStrain_results = []
Ptarget_results = []
Cur_results = []
Mom_results = []
CrackDepth_results = []
EI_results = []

# monitor cpu time
starttime = time.process_time()

# Loop over Ptarget array
for I in range(NUM_SIM):
    # Input Datas
    Ptarget = BETA_PDF(40000, 40100, .5, .5)# [N] Target axial load [+ : Compression]
    b = BETA_PDF(495, 505, .5, .5) # [mm]
    h = BETA_PDF(495, 505, .5, .5) # [mm]
    RD = BETA_PDF(24.5, 25.5, 2.25, 2.25) # Rebar Diameter
    RDA = 3.1415*(RD**2)/4 # Rebar Area
    # As: As1 As2 As3 As4 As5 As6
    As = np.array([RDA*5, 0, 0, 0, 0, RDA*5]) # NOTE: As1 & As6 = 5fi25
    # d:d1 d2 d3 d4 d5 d6
    d = np.array([BETA_PDF(45, 55, .5, .5), 0, 0, 0, 0, BETA_PDF(445, 455, .5, .5)])
    # Concrete Properties
    fc = BETA_PDF(24, 26, 1, 2) # [N/mm^2] Unconfined concrete strength
    ecu = BETA_PDF(0.0035, .0045, 1, 2) # Ultimate concrete strain
    Ec = 5000 * np.sqrt(fc)
    ec0 = (2 * fc) / Ec
    fct = -0.7 * np.sqrt(fc) # Concrete tension stress
    ect1 = (2 * fct) / Ec; ect2 = (2.625 * fct) / Ec; ect3 = (9.292 * fct) / Ec # Concrete tension strain
    # Reinforcing steel Properties
    fy = BETA_PDF(390, 410, 1, 2) # [N/mm^2] Yield strength of reinforcing steel
    Es = BETA_PDF(195000, 200100, 1, 2) # [N/mm^2] Modulus of elasticity of steel
    fu = 1.5 * fy # Ultimate steel stress
    ey = fy / Es # Yield steel strain
    esh = BETA_PDF(.008,0.012, 1, 2) # Strain at steel strain-hardening
    esu = BETA_PDF(0.08, 0.1, 1, 2) # Ultimate steel strain
    Esh = (fu - fy) / (esu - esh)
    N = 1000 # Number of concrete Fiber
    itermax = 10000 # maximum number of iterations
    tolerance = 10e-6 # specified tolerance for convergence
    x = .5 * h # initial guess of Neuteral axis
    ECI = AXIAL_LOAD_ANALYSIS(I, Ptarget, As, d, fc ,ecu, Ec, ec0, fct, ect1, fy, Es, fu, ey, esh, esu, Esh, N, itermax, tolerance)
    CrackDepth, EC, XX, CUR, EI, Cur, Mom, TSCSs, TSCS, BSCSs, BSCS, ECCC, TUCS = MOMENT_CURVATURE_ANALYSIS(ECI , Ptarget, As, d, fc ,ecu, Ec, ec0, fct, ect1, fy, Es, fu, ey, esh, esu, Esh, N, itermax, tolerance, x)
    # Store the results
    Ptarget_results.append(Ptarget)
    AxialStrain_results.append(ECI)
    Cur_results.append(Cur[-1])
    Mom_results.append(Mom[-1])
    CrackDepth_results.append(CrackDepth[-1])
    EI_results.append(EI[-1])

# Convert lists to numpy arrays for plotting
Ptarget_results = np.array(Ptarget_results)
AxialStrain_results = np.array(AxialStrain_results)
Cur_results = np.array(Cur_results)
Mom_results = np.array(Mom_results)
CrackDepth_results = np.array(CrackDepth_results)
EI_results = np.array(EI_results)

totaltime = time.process_time() - starttime
print(f'\nTotal time (s): {totaltime:.4f} \n\n')

#%%--------------------------------------------------------------------
# Output to CSV file
OUTPUT_EXCEL(Ptarget_results, AxialStrain_results, Cur_results, Mom_results, CrackDepth_results, EI_results)
#%%--------------------------------------------------------------------
HISTOGRAM_BOXPLOT_PLOTLY(AxialStrain_results, XLABEL='Axial Strain',TITLE='Monte Carlo Simulation of Axial Strain', COLOR='purple')
#%%--------------------------------------------------------------------
HISTOGRAM_BOXPLOT_PLOTLY(Cur_results, XLABEL='Ultimate Curvature',TITLE='Monte Carlo Simulation of Ultimate Curvature', COLOR='cyan')
#%%--------------------------------------------------------------------
HISTOGRAM_BOXPLOT_PLOTLY(Mom_results, XLABEL='Ultimate Moment',TITLE='Monte Carlo Simulation of Ultimate Moment', COLOR='lime')
#%%--------------------------------------------------------------------
HISTOGRAM_BOXPLOT_PLOTLY(CrackDepth_results, XLABEL='Crack Depth',TITLE='Monte Carlo Simulation of Crack Depth', COLOR='orange')
#%%--------------------------------------------------------------------
HISTOGRAM_BOXPLOT_PLOTLY(EI_results, XLABEL='Flextural Rigidity',TITLE='Monte Carlo Simulation of Flextural Rigidity', COLOR='pink')
#%%--------------------------------------------------------------------
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Given data (mean and standard deviation)
mean_applied_moment = 450e6  # Mean Applied Moment 
std_applied_moment = 50e6    # Std Applied Moment 
mean_resistance_moment = np.mean(Mom_results) # Mean Resistance Moment 
std_resistance_moment = np.std(Mom_results) # Mean Resistance Moment 

# Calculate reliability index (beta)
g_mean = mean_resistance_moment - mean_applied_moment
g_std = np.sqrt(std_applied_moment**2 + std_resistance_moment**2)
beta = g_mean / g_std

# Calculate failure probability
P_f = norm.cdf(-beta)

print(f"Reliability index (beta): {beta:.4f}")
print(f"Failure probability (P_f): {100 * P_f:.2f} ٪")

# Plot reliability histogram
x = np.random.normal(mean_applied_moment, std_applied_moment, 1000)
y = np.random.normal(mean_resistance_moment, std_resistance_moment, 1000)
MIX_HISTOGRAM(x, y, BINS=100, X='Applied Moment', Y='Resistance Moment', TITLE='Applied & Resistance Moment PDF')



# Plot reliability diagram
beta_values = np.linspace(-3, 3, 100)
failure_probs = norm.cdf(-beta_values)

plt.figure(figsize=(8, 6))
plt.plot(beta_values, failure_probs, label="Reliability Diagram")
plt.xlabel("Reliability Index (beta)")
plt.ylabel("Failure Probability")
plt.title("Reliability Analysis")
plt.grid(True)
plt.legend()
#plt.semilogx();plt.semilogy();
plt.show()

#%%--------------------------------------------------------------------
# If we Calculate Reliability Index and Mean Applied Moment Based on Failure Probability
failure_probability = 0.1  # Set your desired failure probability
root = Normal_CDF_Newton_Raphson(failure_probability)

print(f"Reliability Index (beta): {root:.6f}")

# Calculate Mean Applied Moment
mean_applied_moment = root  * g_std - mean_resistance_moment 
print(f"Mean Applied Moment {mean_applied_moment:.3f} Based on failure probability {100 * failure_probability:.3f} %")
#%%--------------------------------------------------------------------
