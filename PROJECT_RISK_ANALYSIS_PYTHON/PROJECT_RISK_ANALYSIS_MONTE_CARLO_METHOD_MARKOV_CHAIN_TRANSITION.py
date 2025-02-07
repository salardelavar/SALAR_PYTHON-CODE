
#########################################################################################
#                                   IN THE NAME OF ALLAH                                #
#      PROJECT RISK ANALYSIS USING MONTE CARLO SIMULATION: TIME AND COST MANAGEMENT     #
#      BASED ON UNCERTAINTY WITH BETA DISTRIBUTION AND MARKOV CHAIN TRANSITION MODELS.  #
#---------------------------------------------------------------------------------------# 
# THIS PYTHON CODE SIMULATES PROJECT RISK ANALYSIS USING MONTE CARLO SIMULATIONS        #
# BASED ON A BETA PROBABILITY DISTRIBUTION. THE PROJECT CONSISTS OF A SERIES OF         #
# ACTIVITIES WITH TIME AND COST ESTIMATES, ALONG WITH DEPENDENCIES (PREDECESSORS).      #
# HERE'S A BREAKDOWN:                                                                   #
#                                                                                       #
# 1. MONTE CARLO SIMULATIONS: FOR EACH ACTIVITY, TIME AND COST ARE SIMULATED MULTIPLE   # 
#    TIMES BASED ON BETA DISTRIBUTIONS, GENERATING A RANGE OF POSSIBLE OUTCOMES         #
#    FOR EACH.                                                                          #
#                                                                                       #
# 2. TIME QUANTILES CALCULATION: AFTER RUNNING THE SIMULATIONS, THE 10TH, 25TH, 50TH,   #
#    75TH, AND 90TH PERCENTILES (QUANTILES) FOR EACH ACTIVITY’S DURATION ARE CALCULATED.# 
#    THESE REPRESENT THE PROJECT COMPLETION TIME AT DIFFERENT CONFIDENCE LEVELS.        #
#                                                                                       #
# 3. GANTT CHART: FOR EACH QUANTILE (E.G., 10TH, 50TH), A GANTT CHART IS GENERATED TO   #
#    VISUALIZE THE PROJECT'S TIMELINE AND THE COMPLETION TIMES OF EACH ACTIVITY.        #
#                                                                                       #
# 4. TRANSITION MATRIX GENERATION: THE CODE ALSO SIMULATES A MARKOV CHAIN TO REPRESENT  #
#    TRANSITIONS BETWEEN ACTIVITIES. A TRANSITION MATRIX IS GENERATED BASED ON THE      #
#    TIME QUANTILES. ACTIVITIES THAT DEPEND ON OTHERS (SUCCESSORS) HAVE TRANSITION      #
#    PROBABILITIES DETERMINED BY THE SIMULATED TIMES FOR THOSE SUCCESSOR ACTIVITIES.    #
#    FOR ACTIVITIES WITH NO SUCCESSORS, SELF-LOOPS ARE ASSIGNED.                        #
#                                                                                       #
# 5. VISUALIZATION: THE CODE VISUALIZES THE TRANSITION MATRICES AS HEATMAPS AND ALSO    #
#    CREATES NETWORK GRAPHS TO SHOW THE DEPENDENCIES BETWEEN ACTIVITIES AT EACH         #
#    QUANTILE. THE EDGE WEIGHTS IN THE NETWORK GRAPH REPRESENT THE PROBABILITY OF       #
#    TRANSITIONING FROM ONE ACTIVITY TO ANOTHER.                                        #
#                                                                                       #
# 6. THRESHOLD FOR VISUALIZATION: A THRESHOLD IS APPLIED WHEN PLOTTING THE TRANSITION   #
#    NETWORK TO REDUCE CLUTTER, ONLY SHOWING TRANSITIONS WITH PROBABILITIES ABOVE THE   #
#    THRESHOLD.                                                                         #
#                                                                                       #
# THIS PROCESS HELPS ASSESS RISKS RELATED TO BOTH TIME AND COST AND VISUALIZES          #
# HOW UNCERTAINTY AFFECTS THE PROJECT'S OVERALL PROGRESS AND DEPENDENCIES.              #
#---------------------------------------------------------------------------------------#
#            THIS PROGRAM IS WRITTEN BY SALAR DELAVAR GHASHGHAEI (QASHQAI)              #
#                         EMAIL: SALAR.D.GHASHGHAEI@GMAIL.COM                           #
#########################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# Define the project activities and their dependencies
activities = {
    'A': {'cost': (1000, 1500), 'time': (5, 9), 'predecessors': []},
    'B': {'cost': (800, 1200), 'time': (4, 8), 'predecessors': []},
    'C': {'cost': (600, 1000), 'time': (3, 7), 'predecessors': ['A']},
    'D': {'cost': (400, 800), 'time': (2, 6), 'predecessors': ['A']},
    'E': {'cost': (500, 900), 'time': (3, 7), 'predecessors': ['B']},
    'F': {'cost': (300, 700), 'time': (2, 6), 'predecessors': ['B']},
    'G': {'cost': (700, 1100), 'time': (4, 8), 'predecessors': ['C', 'D']},
    'H': {'cost': (600, 1000), 'time': (3, 7), 'predecessors': ['E', 'F']},
    'I': {'cost': (800, 1200), 'time': (4, 8), 'predecessors': ['G', 'H']},
    'J': {'cost': (100, 800), 'time': (3, 10), 'predecessors': ['I']}
}

#------------------------------------------------------------------------------
# Beta PDF for Monte Carlo simulation
def beta_pdf(min_x, max_x):
    a, b = 2, 1
    return min_x + (max_x - min_x) * np.random.beta(a, b)

#------------------------------------------------------------------------------
# Function to plot Gantt chart for time quantile
def plot_gantt_time(quantile, COLOR):
    fig, ax = plt.subplots(figsize=(10, 6))
    start_time = {activity: 0 for activity in activities}
    
    # Sort activities in an approximate topological order based on number of predecessors.
    for activity in sorted(activities, key=lambda x: (len(activities[x]['predecessors']), x)):
        for predecessor in activities[activity]['predecessors']:
            start_time[activity] = max(start_time[activity], start_time[predecessor] + results[predecessor][quantile])
        ax.barh(activity, results[activity][quantile], left=start_time[activity], color=COLOR)
        TIME = start_time[activity] + results[activity][quantile]
        START = start_time[activity]
        FINISH = START + results[activity][quantile]
        ax.text(START, activity, f"S: {START:.2f}", va='center', ha='left', fontsize=8)
        ax.text(TIME, activity, f" T: {results[activity][quantile]:.2f}", va='center', ha='left', fontsize=8)
        ax.text(FINISH, activity, f"F: {FINISH:.2f}", va='center', ha='right', fontsize=8)
    
    ax.set_xlabel('Time')
    ax.set_title(f'Gantt Chart for {quantile} Quantile - Project Total Time: {TIME:.2f}')
    plt.grid()
    plt.show()
    return TIME

#------------------------------------------------------------------------------
# Number of simulations
n_sim = 100000

# Run simulations and store results for time
results = {activity: {'times': [], 'costs': []} for activity in activities}

for _ in range(n_sim):
    for activity in activities:
        min_time, max_time = activities[activity]['time']
        results[activity]['times'].append(beta_pdf(min_time, max_time))

# Calculate time quantiles for each activity
for activity in results:
    results[activity]['10th'] = np.quantile(results[activity]['times'], 0.10)
    results[activity]['25th'] = np.quantile(results[activity]['times'], 0.25)
    results[activity]['50th'] = np.quantile(results[activity]['times'], 0.50)
    results[activity]['75th'] = np.quantile(results[activity]['times'], 0.75)
    results[activity]['90th'] = np.quantile(results[activity]['times'], 0.90)

#------------------------------------------------------------------------------
# Plot Gantt charts for different quantiles
time_10 = plot_gantt_time('10th', 'lightgreen')
time_25 = plot_gantt_time('25th', 'lightblue')
time_50 = plot_gantt_time('50th', 'orange')
time_75 = plot_gantt_time('75th', 'red')
time_90 = plot_gantt_time('90th', 'purple')

print("-------------------- TIME --------------------------")
print(f"Project Total Time for 10th Quantile: {time_10:.2f}")
print(f"Project Total Time for 25th Quantile: {time_25:.2f}")
print(f"Project Total Time for 50th Quantile: {time_50:.2f}")
print(f"Project Total Time for 75th Quantile: {time_75:.2f}")
print(f"Project Total Time for 90th Quantile: {time_90:.2f}")

#------------------------------------------------------------------------------
# Now, instead of (or in addition to) generating a random transition matrix,
# we generate one based on the simulated activity times for a given quantile.
# The idea: for each activity (state), if it has successors, assign a weight for
# transitioning to each successor based on that successor's time (at the chosen quantile).
# For activities with no successors, assign a self-loop of probability 1.

# First, compute the successors for each activity based on the dependency structure.
successors = {act: [] for act in activities}
for act in activities:
    for other in activities:
        if act in activities[other]['predecessors']:
            successors[act].append(other)

# Define a function to generate a transition matrix for a given time quantile.
def generate_transition_matrix(quantile):
    activities_list = list(activities.keys())
    n = len(activities_list)
    T = np.zeros((n, n))
    
    for i, act in enumerate(activities_list):
        succs = successors[act]
        if succs:  # If there are successors, weight the transitions
            # Here we use the simulated time for the successor activity at the chosen quantile
            weights = [results[succ][quantile] for succ in succs]
            total_weight = sum(weights)
            # To avoid division by zero, check total_weight
            if total_weight == 0:
                # If all weights are zero, assign equal probabilities.
                for succ in succs:
                    j = activities_list.index(succ)
                    T[i, j] = 1 / len(succs)
            else:
                for j, succ in enumerate(succs):
                    idx = activities_list.index(succ)
                    T[i, idx] = weights[j] / total_weight
        else:
            # No successors: assign a self-loop (absorbing state)
            T[i, i] = 1.0
    return T, activities_list

# Generate transition matrices for the 10th and 25th time quantiles
T_10, activities_list = generate_transition_matrix('10th')
T_25, _ = generate_transition_matrix('25th')
T_50, _ = generate_transition_matrix('50th')
T_75, _ = generate_transition_matrix('75th')
T_90, _ = generate_transition_matrix('90th')

#------------------------------------------------------------------------------
# Visualize the transition matrix (heatmap) and network graph for each quantile

def plot_transition_matrix(T, activities_list, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(T, annot=True, cmap='coolwarm', xticklabels=activities_list, yticklabels=activities_list)
    plt.title(title)
    plt.show()

def plot_transition_network(T, activities_list, title, threshold=0.0):
    G = nx.DiGraph()
    n = len(activities_list)
    for i in range(n):
        for j in range(n):
            if T[i, j] > threshold:
                G.add_edge(activities_list[i], activities_list[j], weight=T[i, j])
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, 
            font_size=10, font_weight='bold', edge_color='gray')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    # Format edge labels to show probabilities
    formatted_edge_labels = {(u, v): f"{w:.2f}" for (u, v), w in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=formatted_edge_labels)
    plt.title(title)
    plt.show()

# Plot for 10th quantile
plot_transition_matrix(T_10, activities_list, "Transition Matrix Based on 10th Time Quantile")
plot_transition_network(T_10, activities_list, "Transition Network Based on 10th Time Quantile", threshold=0.01)

# Plot for 25th quantile
plot_transition_matrix(T_25, activities_list, "Transition Matrix Based on 25th Time Quantile")
plot_transition_network(T_25, activities_list, "Transition Network Based on 25th Time Quantile", threshold=0.01)

# Plot for 50th quantile
plot_transition_matrix(T_50, activities_list, "Transition Matrix Based on 50th Time Quantile")
plot_transition_network(T_50, activities_list, "Transition Network Based on 50th Time Quantile", threshold=0.01)

# Plot for 75th quantile
plot_transition_matrix(T_75, activities_list, "Transition Matrix Based on 75th Time Quantile")
plot_transition_network(T_75, activities_list, "Transition Network Based on 75th Time Quantile", threshold=0.01)

# Plot for 90th quantile
plot_transition_matrix(T_90, activities_list, "Transition Matrix Based on 90th Time Quantile")
plot_transition_network(T_90, activities_list, "Transition Network Based on 90th Time Quantile", threshold=0.01)

#------------------------------------------------------------------------------
