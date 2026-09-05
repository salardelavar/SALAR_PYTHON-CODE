###########################################################################################################
#                   >> IN THE NAME OF ALLAH, THE MOST GRACIOUS, THE MOST MERCIFUL <<                      #
#       DISPLACEMENT-CONTROLLED NONLINEAR STATIC PUSHOVER ANALYSIS OF A MULTI-STORY BUILDING WITH         #
#                   COLUMNS AND SHEAR WALL, USING A PARALLEL AND SERIES SPRING CONCEPT                    #
#---------------------------------------------------------------------------------------------------------#
#                 THIS PYTHON SCRIPT IS WRITTEN BY SALAR DELAVAR GHASHGHAEI (QASHQAI)                     #
#                                   EMAIL: salar.d.ghashghaei@gmail.com                                   #
###########################################################################################################
"""
This python script performs a nonlinear static pushover analysis of a n-story
 building modeled with multiple column types and a shear wall in parallel at each story.
 It defines element backbone force-displacement relationships (including post-yield stiffness,
 strength degradation, and residual capacity) and implements a displacement-controlled solution
 scheme. Using a triangular lateral load pattern, the analysis incrementally increases the roof
 displacement, solving for the free story displacements via a Newton–Raphson iteration with line
 search until force equilibrium is satisfied to a tight tolerance. The code calculates story
 shears, element forces, and tangent stiffness matrices, then exports global and element-level
 results to an Excel file. 
 Finally, it generates four plots: the pushover curve (base shear vs. roof displacement),
 element force-displacement curves for the first story, story shear vs. inter-story drift
 for all stories, and maximum story shear per story. This analysis allows engineers to
 evaluate the inelastic behavior, strength, and deformation capacity of the structural
 system under lateral loading.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% -----------------------------------------------------
# Element data (columns and shear wall)
element_data = [
    {"name": "C1", "FY": 0.240, "FU_ratio": 1.1818, "Ke": 200.0,  "DSU": 0.36,
     "residual_ratios": [0.2, 0.1], "disp_multipliers": [1.1, 1.25]},
    {"name": "C2", "FY": 0.320, "FU_ratio": 1.25,   "Ke": 280.0,  "DSU": 0.42,
     "residual_ratios": [0.25, 0.12], "disp_multipliers": [1.15, 1.30]},
    {"name": "C3", "FY": 0.180, "FU_ratio": 1.15,   "Ke": 150.0,  "DSU": 0.50,
     "residual_ratios": [0.18, 0.08], "disp_multipliers": [1.20, 1.40]},
    {"name": "C4", "FY": 0.260, "FU_ratio": 1.20,   "Ke": 220.0,  "DSU": 0.38,
     "residual_ratios": [0.22, 0.10], "disp_multipliers": [1.12, 1.28]},
    {"name": "Shear-wall", "FY": 2.260, "FU_ratio": 1.20, "Ke": 22000.0, "DSU": 0.8,
     "residual_ratios": [0.22, 0.10], "disp_multipliers": [1.12, 1.28]},
]

n_stories = 10 # Setup n‑story building

# Pre‑compute backbone points for each element
element_backbones = []
for elem in element_data:
    uy = elem["FY"] / elem["Ke"]
    fu = elem["FY"] * elem["FU_ratio"]
    d1 = elem["disp_multipliers"][0] * elem["DSU"]
    d2 = elem["disp_multipliers"][1] * elem["DSU"]
    f1 = elem["residual_ratios"][0] * elem["FY"]
    f2 = elem["residual_ratios"][1] * elem["FY"]
    pts = [(0.0, 0.0), (uy, elem["FY"]), (elem["DSU"], fu), (d1, f1), (d2, f2)]
    pts.sort(key=lambda x: x[0])
    element_backbones.append(pts)

#%% -----------------------------------------------------
# Function to compute element force and tangent stiffness
def ELEMENT_FORCE_STIFF(u, backbone, Ke):
    """Return force and tangent stiffness for displacement u (symmetric behaviour)."""
    sign = 1.0 if u >= 0 else -1.0
    x = abs(u)
    pts = backbone
    # Elastic branch
    if x <= pts[0][0]:
        return sign * Ke * x, Ke
    # Find the appropriate linear segment
    for i in range(len(pts) - 1):
        x0, f0 = pts[i]
        x1, f1 = pts[i+1]
        if x <= x1:
            slope = (f1 - f0) / (x1 - x0) if x1 != x0 else 0.0
            f = f0 + slope * (x - x0)
            return sign * f, slope
    # After last point: constant force
    f = pts[-1][1]
    return sign * f, 0.0

#%% -----------------------------------------------------
# Function to compute story shear and tangent stiffness
def STORY_RESPONSE(drift, backbones, Ke_list):
    """Total shear and tangent stiffness for a story (parallel elements)."""
    V = 0.0
    K = 0.0
    for bb, ke in zip(backbones, Ke_list):
        f, k = ELEMENT_FORCE_STIFF(drift, bb, ke)
        V += f
        K += k
    return V, K

#%% -----------------------------------------------------
# Global structure response (R, K, story shears, drifts)
def STRUCTURE_RESPONSE(u, story_backbones, story_Ke):
    n = len(u)
    # Drift transformation matrix: drifts = B @ u
    B = np.zeros((n, n))
    for i in range(n):
        B[i, i] = 1.0
        if i > 0:
            B[i, i-1] = -1.0

    drifts = B @ u
    V = np.zeros(n)
    Kdiag = np.zeros(n)
    for i in range(n):
        V[i], Kdiag[i] = STORY_RESPONSE(drifts[i], story_backbones[i], story_Ke[i])

    R = B.T @ V
    K = B.T @ np.diag(Kdiag) @ B
    return R, K, V, drifts

#%% -----------------------------------------------------
# Setup n‑story building (same elements in every story)
story_backbones = [element_backbones] * n_stories
story_Ke = [[elem["Ke"] for elem in element_data]] * n_stories

# Triangular load pattern (story 1 = 1, ..., roof = 10)
P = np.arange(1, n_stories + 1, dtype=float)
P_free = P[:-1]
P_roof = P[-1]

#%% -----------------------------------------------------
# Displacement‑controlled pushover analysis

max_roof_disp = 2.0        # maximum roof displacement
n_steps = 3000
targets = np.linspace(0, max_roof_disp, n_steps + 1)[1:]   # skip zero

MAX_TOL = 1e-8
MAX_ITER = 500

def RUN_PUSHOVER_ANALYSIS(
    story_backbones,
    story_Ke,
    element_data,
    element_backbones,
    P,
    max_roof_disp,
    n_steps,
    MAX_TOL,
    MAX_ITER,
    verbose=True
    ):
    """
    Perform displacement‑controlled nonlinear static pushover analysis.

    Parameters
    ----------
    story_backbones : list of list of tuples
        Each story contains a list of element backbone points (force-displacement pairs).
    story_Ke : list of list of float
        Initial elastic stiffness for each element in each story.
    element_data : list of dict
        Element metadata (must include 'name' and 'Ke' keys).
    element_backbones : list of list of tuples
        Backbone data for each unique element type.
    P : array_like
        Reference lateral load pattern (one value per story).
    max_roof_disp : float, optional
        Maximum roof displacement to impose.
    n_steps : int, optional
        Number of displacement increments.
    tol : float, optional
        Convergence tolerance for Newton‑Raphson residual.
    max_iter : int, optional
        Maximum iterations per displacement step.
    verbose : bool, optional
        If True, print progress messages.

    Returns
    -------
    df_global : pandas.DataFrame
        Global results (step, roof_disp, base_shear, load_factor).
    df_elements : pandas.DataFrame
        Element‑level results (step, story, element, drift, force).
    """
    import numpy as np
    import pandas as pd
    
    # Derived values
    n_stories = len(P)
    P = np.asarray(P, dtype=float)
    P_free = P[:-1]
    P_roof = P[-1]

    targets = np.linspace(0, max_roof_disp, n_steps + 1)[1:]   # skip zero

    global_results = []
    element_results = []

    prev_u = None
    prev_target = 0.0

    for step, target in enumerate(targets):
        # Initial guess for free displacements
        if prev_u is not None and prev_target > 1e-6:
            scale = target / prev_target
            u_free = prev_u[:-1] * scale
        else:
            u_free = target * np.arange(1, n_stories) / n_stories

        u_roof = target
        converged = False

        for iteration in range(MAX_ITER):
            u = np.concatenate([u_free, [u_roof]])
            R, K, V, drifts = STRUCTURE_RESPONSE(u, story_backbones, story_Ke)

            R_free = R[:-1]
            R_roof = R[-1]

            alpha = R_roof / P_roof
            g = R_free - alpha * P_free
            norm_g = np.linalg.norm(g)

            if norm_g < MAX_TOL:
                converged = True
                break

            # Jacobian of the reduced system
            K_ff = K[:-1, :-1]
            K_rf = K[-1, :-1]
            J = K_ff - np.outer(P_free / P_roof, K_rf)

            try:
                delta = np.linalg.solve(J, -g)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(J, -g, rcond=None)[0]

            # Simple line search
            step_size = 1.0
            accepted = False
            for _ in range(20):
                u_free_new = u_free + step_size * delta
                u_new = np.concatenate([u_free_new, [u_roof]])
                R_new, _, _, _ = STRUCTURE_RESPONSE(u_new, story_backbones, story_Ke)
                R_free_new = R_new[:-1]
                R_roof_new = R_new[-1]
                alpha_new = R_roof_new / P_roof
                g_new = R_free_new - alpha_new * P_free
                if np.linalg.norm(g_new) < norm_g:
                    u_free = u_free_new
                    accepted = True
                    break
                step_size *= 0.5
            if not accepted:
                u_free = u_free_new   # forced accept

        if not converged:
            if verbose:
                print(f"{step+1} - Disp = {target:.4f} - Iteration: {iteration+1} - Warning: convergence failed")
            continue
        else:
            if verbose:
                print(f"{step+1} - Disp = {target:.4f} - Iteration: {iteration+1}")

        # Save converged results
        u = np.concatenate([u_free, [u_roof]])
        R, K, V, drifts = STRUCTURE_RESPONSE(u, story_backbones, story_Ke)
        alpha = R[-1] / P_roof
        base_shear = alpha * np.sum(P)

        global_results.append({
            'step': step,
            'roof_disp': u_roof,
            'base_shear': base_shear,
            'load_factor': alpha,
        })

        # Save element results
        for story_idx in range(n_stories):
            d = drifts[story_idx]
            for elem_idx, elem_data_i in enumerate(element_data):
                f, _ = ELEMENT_FORCE_STIFF(d, element_backbones[elem_idx], elem_data_i["Ke"])
                element_results.append({
                    'step': step,
                    'story': story_idx + 1,
                    'element': elem_data_i["name"],
                    'drift': d,
                    'force': f,
                })

        prev_u = u.copy()
        prev_target = target

    # Convert to DataFrames
    df_global = pd.DataFrame(global_results)
    df_elements = pd.DataFrame(element_results)
    return df_global, df_elements

df_global, df_elements = RUN_PUSHOVER_ANALYSIS(
    story_backbones,
    story_Ke,
    element_data,
    element_backbones,
    P,
    max_roof_disp,
    n_steps,
    MAX_TOL,
    MAX_ITER,
    verbose=True
)
#%% -----------------------------------------------------
# Convert to DataFrames and export to Excel

with pd.ExcelWriter('DC_MDOF_PUSHOVER_MULTI_STORY_RESULTS.xlsx') as writer:
    df_global.to_excel(writer, sheet_name='Global', index=False)
    df_elements.to_excel(writer, sheet_name='Elements', index=False)

print("Analysis completed. Results saved to 'DC_MDOF_PUSHOVER_MULTI_STORY_RESULTS.xlsx'.")
#%% -----------------------------------------------------
# Plot 1: Pushover curve (base shear vs roof displacement)

plt.figure(figsize=(8,6))
plt.plot(df_global['roof_disp'], df_global['base_shear'], 'b-o', markersize=3)
plt.xlabel('Roof Displacement')
plt.ylabel('Base Shear')
plt.title(f'Pushover Curve   -   {n_stories}-Story Building')
plt.grid(True)
plt.show()

#%% -----------------------------------------------------
# Plot 2: Element force-displacement for Story 1

story1 = df_elements[df_elements['story'] == 1]
elements_in_story1 = story1['element'].unique()

fig, axes = plt.subplots(len(elements_in_story1), 1, figsize=(8, 12), sharex=True)
if len(elements_in_story1) == 1:
    axes = [axes]   # ensure list

for ax, elem_name in zip(axes, elements_in_story1):
    data = story1[story1['element'] == elem_name]
    ax.plot(data['drift'], data['force'], 'b-o')
    ax.set_ylabel('Force')
    ax.set_title(f'Story 1 - {elem_name}')
    ax.grid(True)
axes[-1].set_xlabel('Drift')
plt.tight_layout()
plt.show()

#%% -----------------------------------------------------
# Additional: Story shear force plots and export
# Sum element forces per step and story
story_shear_df = df_elements.groupby(['step', 'story'])['force'].sum().reset_index()
story_shear_df.rename(columns={'force': 'story_shear'}, inplace=True)

# Get the drift for each story (identical for all elements in a story)
story_drift_df = df_elements.groupby(['step', 'story'])['drift'].first().reset_index()

# Merge to have story shear and drift side by side
story_response = pd.merge(story_shear_df, story_drift_df, on=['step', 'story'])

# Plot story shear vs. story drift for each story
plt.figure(figsize=(10, 6))
for story_num in range(1, n_stories + 1):
    data = story_response[story_response['story'] == story_num]
    plt.plot(df_global['roof_disp'], data['story_shear'], #data['drift']
             label=f'Story {story_num}', linewidth=2.5)

plt.xlabel('Roof Displacement')
plt.ylabel('Story Shear Force')
plt.title('Story Shear vs. Roof Displacement for All Stories')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: export story shear data to Excel
# Read the existing Excel file, add a new sheet, and rewrite
# (If you prefer not to overwrite, use a different file name)
with pd.ExcelWriter('DC_MDOF_PUSHOVER_MULTI_STORY_RESULTS.xlsx',
                    mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    story_response.to_excel(writer, sheet_name='StoryShears', index=False)

print("Story shear plots created. Story shear data saved to 'StoryShears' sheet.")

#%% -----------------------------------------------------
# Plot story shear (base reaction) vs. inter-story drift for each story

# Get unique story numbers
stories = sorted(story_response['story'].unique())
n_stories_plot = len(stories)

# Create subplots: one per story (arranged vertically)
fig, axes = plt.subplots(n_stories_plot, 1, figsize=(8, 2.5 * n_stories_plot), sharex=True)
if n_stories_plot == 1:
    axes = [axes]  # ensure list for consistency

for ax, story_num in zip(axes, stories):
    data = story_response[story_response['story'] == story_num]
    ax.plot(data['drift'], data['story_shear'], 'b-o', markersize=3, linewidth=1.5)
    ax.set_ylabel(f'Story {story_num}\nShear Force')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title(f'Story {story_num} – Shear vs. Drift', fontsize=10)

axes[-1].set_xlabel('Inter-story Drift')
plt.tight_layout()
plt.show()

# Plot story shear (base reaction) vs. inter-story drift for all stories on one plot

# Get unique story numbers
stories = sorted(story_response['story'].unique())

plt.figure(figsize=(10, 6))

for story_num in stories:
    data = story_response[story_response['story'] == story_num]
    plt.plot(data['drift'], data['story_shear'],
             marker='o', markersize=3, linewidth=1.5,
             label=f'Story {story_num}')

plt.xlabel('Inter-story Drift')
plt.ylabel('Story Shear Force')
plt.title('Story Shear vs. Inter-story Drift (All Stories)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
#%% -----------------------------------------------------
# Plot maximum story shear (base reaction) per story

# Compute maximum absolute story shear for each story
max_story_shear = story_response.groupby('story')['story_shear'].apply(
    lambda x: x.abs().max()
).reset_index()
max_story_shear.columns = ['story', 'max_shear']

# Plot as a line chart with markers
plt.figure(figsize=(8, 6))
plt.plot(max_story_shear['max_shear'], max_story_shear['story'],
         marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
plt.ylabel('Story Number')
plt.xlabel('Maximum Story Shear (Base Reaction)')
plt.title('Maximum Story Shear per Story During Pushover')
plt.yticks(max_story_shear['story'])
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Optional: export to Excel (adds a new sheet)
with pd.ExcelWriter('DC_MDOF_PUSHOVER_MULTI_STORY_RESULTS.xlsx',
                    mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    max_story_shear.to_excel(writer, sheet_name='MaxStoryShears', index=False)

print("Maximum story shear plot created and saved to 'MaxStoryShears' sheet.")
#%% -----------------------------------------------------
# Story response quantities

# Sum element forces per step and story
story_shear_df = (
    df_elements
    .groupby(['step', 'story'])['force']
    .sum()
    .reset_index()
)

story_shear_df.rename(
    columns={'force': 'story_shear'},
    inplace=True
)

# Get inter-story drift
story_drift_df = (
    df_elements
    .groupby(['step', 'story'])['drift']
    .first()
    .reset_index()
)

# Merge
story_response = pd.merge(
    story_shear_df,
    story_drift_df,
    on=['step', 'story']
)

# Maximum absolute roof displacement
roof_max = df_global['roof_disp'].max()

# Approximate building height
# Here story height is assumed to be 1.0
story_height = 1.0

# Inter-story drift ratio
story_response['drift_ratio'] = (
    story_response['drift'].abs() / story_height
)

story_response['drift_ratio_percent'] = (
    story_response['drift_ratio'] * 100.0
)
#%% -----------------------------------------------------
story_height = 3.2


#%% -----------------------------------------------------
# Plot: Inter-story Drift Ratio vs Roof Displacement

plt.figure(figsize=(10, 6))

for story_num in sorted(story_response['story'].unique()):

    data = story_response[
        story_response['story'] == story_num
    ]

    plt.plot(
        data['step'],
        data['drift_ratio_percent'],
        linewidth=1.8,
        label=f'Story {story_num}'
    )

plt.xlabel('Analysis Step')
plt.ylabel('Inter-story Drift Ratio (%)')
plt.title(
    f'Inter-story Drift Ratio - {n_stories}-Story Building'
)

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)

plt.tight_layout()
plt.show()

#%% -----------------------------------------------------
# Inter-story Drift Ratio vs Roof Displacement

plt.figure(figsize=(10, 6))

for story_num in sorted(story_response['story'].unique()):

    data = story_response[
        story_response['story'] == story_num
    ]

    roof_disp = df_global.loc[
        df_global['step'].isin(data['step']),
        'roof_disp'
    ].values

    plt.plot(
        roof_disp,
        data['drift_ratio_percent'].values,
        linewidth=1.8,
        label=f'Story {story_num}'
    )

plt.xlabel('Roof Displacement')
plt.ylabel('Inter-story Drift Ratio (%)')
plt.title('Inter-story Drift Ratio vs Roof Displacement')

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc='upper left'
)

plt.tight_layout()
plt.show()

#%% -----------------------------------------------------
# Global effective stiffness

df_global['effective_stiffness'] = (
    df_global['base_shear']
    / df_global['roof_disp'].replace(0, np.nan)
)

#%% -----------------------------------------------------
# Plot: Effective Stiffness vs Roof Displacement

plt.figure(figsize=(8, 6))

plt.plot(
    df_global['roof_disp'],
    df_global['effective_stiffness'],
    linewidth=2
)

plt.xlabel('Roof Displacement')
plt.ylabel('Effective Stiffness')
plt.title(
    f'Global Effective Stiffness Degradation - '
    f'{n_stories}-Story Building'
)

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.semilogy()
plt.show()


#alpha = R[-1] / P_roof


#%% -----------------------------------------------------
# Plot: Load Factor vs Roof Displacement

plt.figure(figsize=(8, 6))

plt.plot(
    df_global['roof_disp'],
    df_global['load_factor'],
    linewidth=2
)

plt.xlabel('Roof Displacement')
plt.ylabel('Load Factor, α')
plt.title(
    f'Load Factor vs Roof Displacement - '
    f'{n_stories}-Story Building'
)

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#%% -----------------------------------------------------
# Maximum inter-story drift ratio per story

max_drift_ratio = (
    story_response
    .groupby('story')['drift_ratio_percent']
    .max()
    .reset_index()
)

max_drift_ratio.columns = [
    'story',
    'max_drift_ratio_percent'
]

# Plot
plt.figure(figsize=(8, 6))

plt.plot(
    max_drift_ratio['max_drift_ratio_percent'],
    max_drift_ratio['story'],
    marker='o',
    linewidth=2
)

plt.xlabel('Maximum Inter-story Drift Ratio (%)')
plt.ylabel('Story Number')

plt.title(
    'Maximum Inter-story Drift Ratio Profile'
)

plt.yticks(max_drift_ratio['story'])

plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

#%% -----------------------------------------------------
# Export additional results

with pd.ExcelWriter(
    'DC_MDOF_PUSHOVER_MULTI_STORY_RESULTS.xlsx',
    mode='a',
    engine='openpyxl',
    if_sheet_exists='replace'
) as writer:

    story_response.to_excel(
        writer,
        sheet_name='StoryResponse',
        index=False
    )

    df_global.to_excel(
        writer,
        sheet_name='Global',
        index=False
    )

    max_drift_ratio.to_excel(
        writer,
        sheet_name='MaxDriftRatio',
        index=False
    )

print("Additional response parameters exported successfully.")
