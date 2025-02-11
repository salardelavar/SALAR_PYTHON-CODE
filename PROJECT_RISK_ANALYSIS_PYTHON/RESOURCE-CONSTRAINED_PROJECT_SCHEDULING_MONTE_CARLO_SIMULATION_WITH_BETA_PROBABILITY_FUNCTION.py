"""
*****************************************************************
*                    >> IN THE NAME OF ALLAH <<                 *
*         RESOURCE-CONSTRAINED PROJECT SCHEDULING (RCPSP)       *
*       MONTE CARLO SIMULATION WITH BETA PROBABILITY FUNCTION   *
*---------------------------------------------------------------*
* This program is written by Salar Delavar Ghashghaei (Qashqai) *
*             E-mail:salar.d.ghashghaei@gmail.com               *
*****************************************************************
"""
"""
Project Definition:
The activities dictionary defines each activity with its uncertain time interval (min, mode, max), required resources, and predecessor relationships.

Beta PDF Sampling:
The beta_pdf function samples a value between the given minimum and maximum using a beta distribution with shape parameters 2 and 1.

Topological Sorting:
The topological_sort function orders the activities such that all predecessor constraints are respected.

Resource Check:
The check_resource_availability function ensures that adding an activity at a candidate start time does not exceed the resource capacity.

Project Scheduling:
The schedule_project function assigns start and finish times for each activity while satisfying both precedence and resource constraints using a Serial Schedule Generation Scheme.

Monte Carlo Simulation:
The simulation repeatedly samples activity durations, generates schedules, and collects the overall project makespan. Statistics (quantiles) for the makespan are then calculated and displayed.

Visualization:
A histogram of the makespan distribution is plotted, and a Gantt chart for a sample schedule (closest to the median makespan) is generated.
"""

import numpy as np
import matplotlib.pyplot as plt
import math

# -------------------------------------------------------------------------------------
# 1. Define the project activities with uncertain durations and resource requirements
# -------------------------------------------------------------------------------------

# For each activity, the time interval is given as (min, mode, max),
# along with its resource requirement and predecessor relationships.
activities = {
    'A': {'time': (3, 4, 6),  'resource': 2, 'predecessors': []},
    'B': {'time': (2, 3, 5),  'resource': 3, 'predecessors': []},
    'C': {'time': (4, 5, 7),  'resource': 2, 'predecessors': ['A']},
    'D': {'time': (1, 2, 3),  'resource': 4, 'predecessors': ['A']},
    'E': {'time': (3, 4, 7),  'resource': 2, 'predecessors': ['B']},
    'F': {'time': (2, 3, 6),  'resource': 3, 'predecessors': ['B']},
    'G': {'time': (4, 5, 8),  'resource': 3, 'predecessors': ['C', 'D']},
    'H': {'time': (3, 4, 7),  'resource': 2, 'predecessors': ['E', 'F']},
    'I': {'time': (4, 5, 8),  'resource': 3, 'predecessors': ['G', 'H']},
    'J': {'time': (3, 4, 10), 'resource': 1, 'predecessors': ['I']}
}

# Total resource capacity (e.g., number of resource units available at any time)
resource_capacity = 5

# ------------------------------------------------------
# 2. Beta PDF Sampling Function for Activity Durations
# ------------------------------------------------------

def beta_pdf(min_x, max_x):
    """
    Sample a value from a beta distribution scaled between min_x and max_x.
    The shape parameters 'a' and 'b' are set to 2 and 1, respectively.
    """
    a, b = 2, 1  # Shape parameters (modifiable)
    return min_x + (max_x - min_x) * np.random.beta(a, b)

# -----------------------------------------------------------------------
# 3. Topological Sorting of Activities (Based on Precedence Constraints)
# -----------------------------------------------------------------------

def topological_sort(activities):
    in_degree = {act: 0 for act in activities}
    for act, data in activities.items():
        for pred in data['predecessors']:
            in_degree[act] += 1
    order = []
    zero_in_degree = [act for act in activities if in_degree[act] == 0]
    while zero_in_degree:
        current = zero_in_degree.pop(0)
        order.append(current)
        for act, data in activities.items():
            if current in data['predecessors']:
                in_degree[act] -= 1
                if in_degree[act] == 0:
                    zero_in_degree.append(act)
    return order

# --------------------------------------------------------------
# 4. Check Resource Availability within a Given Time Interval
# --------------------------------------------------------------

def check_resource_availability(schedule, candidate_start, duration, demand, resource_capacity):
    """
    Check whether adding an activity with a specific resource demand during the 
    interval [candidate_start, candidate_start + duration) violates the resource capacity.
    """
    candidate_end = candidate_start + duration
    # Check using discrete time units (can be made more precise if needed)
    start_int = int(math.floor(candidate_start))
    end_int = int(math.ceil(candidate_end))
    for t in range(start_int, end_int):
        usage = 0
        for act, times in schedule.items():
            if times['start'] <= t < times['end']:
                usage += activities[act]['resource']
        if usage + demand > resource_capacity:
            return False
    return True

# -------------------------------------------------------------------------
# 5. Project Scheduling Using the Serial Schedule Generation Scheme (SSGS)
# -------------------------------------------------------------------------

def schedule_project(sampled_durations, activities, resource_capacity):
    """
    Schedule the project activities while respecting precedence constraints and resource limits.
    
    Parameters:
        sampled_durations (dict): A dictionary containing the sampled duration for each activity.
    
    Returns:
        schedule (dict): A dictionary with the start and end times for each activity.
        makespan (float): The overall project completion time.
    """
    order = topological_sort(activities)
    schedule = {}
    for act in order:
        duration = sampled_durations[act]
        # Determine the earliest start time based on the finish times of predecessor activities
        if activities[act]['predecessors']:
            earliest_start = max(schedule[pred]['end'] for pred in activities[act]['predecessors'])
        else:
            earliest_start = 0
        candidate_start = earliest_start
        # Find the first time when the resource constraint is satisfied
        while not check_resource_availability(schedule, candidate_start, duration, activities[act]['resource'], resource_capacity):
            candidate_start += 1
        schedule[act] = {'start': candidate_start, 'end': candidate_start + duration}
    makespan = max(times['end'] for times in schedule.values())
    return schedule, makespan

# --------------------------------------------------
# 6. Monte Carlo Simulation for RCPSP
# --------------------------------------------------

def monte_carlo_rcpsp(num_sim, activities, resource_capacity):
    makespans = []
    schedules = []
    for _ in range(num_sim):
        sampled_durations = {}
        # Sample a duration for each activity using the beta_pdf function
        for act, data in activities.items():
            min_time, mode_time, max_time = data['time']
            # Use beta_pdf and round up to the nearest integer
            sampled_duration = math.ceil(beta_pdf(min_time, max_time))
            sampled_durations[act] = sampled_duration
        schedule, makespan = schedule_project(sampled_durations, activities, resource_capacity)
        makespans.append(makespan)
        schedules.append(schedule)
    return makespans, schedules

# Set the number of simulation iterations (e.g., 10000 iterations)
num_sim = 10000
makespans, schedules = monte_carlo_rcpsp(num_sim, activities, resource_capacity)

# -------------------------------------------------------------------
# 7. Calculate Makespan Statistics (Overall Project Completion Time)
# -------------------------------------------------------------------

quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]
quantile_values = {q: np.quantile(makespans, q) for q in quantiles}

print("------ Monte Carlo Simulation Results for RCPSP ------")
for q in quantiles:
    print(f"Project Makespan at {int(q*100)}th Percentile: {quantile_values[q]:.2f}")

# --------------------------------------------------
# 8. Plot Histogram of the Makespan Distribution
# --------------------------------------------------

plt.figure(figsize=(8, 4))
plt.hist(makespans, bins=range(min(makespans), max(makespans) + 2), color='skyblue', edgecolor='black')
plt.xlabel('Project Makespan')
plt.ylabel('Frequency')
plt.title('Makespan Distribution in Monte Carlo Simulation (RCPSP)')
plt.grid(True)
plt.show()

# --------------------------------------------------------------
# 9. Select a Sample Schedule (Closest to the Median Makespan)
# --------------------------------------------------------------

median_makespan = quantile_values[0.50]
closest_index = min(range(len(makespans)), key=lambda i: abs(makespans[i] - median_makespan))
selected_schedule = schedules[closest_index]

# --------------------------------------------------
# 10. Plot a Gantt Chart for the Selected Schedule
# --------------------------------------------------

def plot_gantt(schedule, activities):
    order = topological_sort(activities)
    fig, ax = plt.subplots(figsize=(10, 6))
    yticks = []
    ytick_labels = []
    height = 0.8
    for i, act in enumerate(order):
        start = schedule[act]['start']
        end = schedule[act]['end']
        duration = end - start
        ax.broken_barh([(start, duration)], (i - height/2, height), facecolors=('tab:blue'))
        yticks.append(i)
        ytick_labels.append(act)
        ax.text(start + duration/2, i, f"{act}\n({start}, {end})", va='center', ha='center', color='white', fontsize=9)
    ax.set_xlabel('Time')
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    total_time = max(schedule[act]['end'] for act in schedule)
    ax.set_title(f'Selected Schedule Gantt Chart (Makespan: {total_time:.2f})')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

plot_gantt(selected_schedule, activities)
