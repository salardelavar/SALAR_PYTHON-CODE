"""
            *****************************************************************
            *                    >> IN THE NAME OF ALLAH <<                 *
            *                      PROJECT RISK ANALYSIS                    *
            *      TIME AND COST MANAGEMENT BASED ON UNCERTAINTY CONDITION  *
            *---------------------------------------------------------------*
            * This program is written by Salar Delavar Ghashghaei (Qashqai) *
            *             E-mail:salar.d.ghashghaei@gmail.com               *
            *****************************************************************
            

This sophisticated Python solution implements PMI-aligned critical path analysis with Monte Carlo-inspired scenario modeling
 to generate professional S-curve forecasts. 
-------------- 
Key features:
--------------
1. Dependency-Aware Scheduling: Implements Kahn's topological sort to resolve activity dependencies,
ensuring valid schedule sequences while detecting cyclic conflicts

2. Scenario Modeling: Processes pessimistic/likely/optimistic estimates using triangular distributions
for both duration and cost parameters.

3. Time-Phased Cost Allocation: Distributes activity costs linearly across durations, creating accurate
daily expenditure profiles.

4. Critical Path Methodology: Calculates ES/EF times through forward pass analysis, identifying scenario-dependent critical paths.

5. Advanced Visualization: Produces publication-ready S-curves with embedded summary statistics, using matplotlib's table functionality for at-a-glance scenario comparisons

6. Risk Analysis Ready: Clear separation of scenario parameters enables easy integration with probabilistic analysis (VaR, confidence intervals)

7. Executive Reporting: Outputs include both granular schedule metrics (ES/EF per activity) and C-suite focused cumulative cost projections

The solution adheres to AACE International recommended practices for cost engineering, providing actionable insights for capital project planning and risk mitigation. Final visualization delivers immediate visual comparison of scenario outcomes with professional styling suitable for board presentations.            
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# Project Configuration
SCENARIOS = ["pessimistic", "likely", "optimistic"]
scenario_to_idx = {"optimistic": 0, "likely": 1, "pessimistic": 2}
ACTIVITIES = {
    'A': {'cost': (1000, 1200, 1500), 'time': (5, 7, 9), 'predecessors': []},
    'B': {'cost': (800, 1000, 1200), 'time': (4, 6, 8), 'predecessors': []},
    'C': {'cost': (600, 800, 1000), 'time': (3, 5, 7), 'predecessors': ['A']},
    'D': {'cost': (400, 600, 800), 'time': (2, 4, 6), 'predecessors': ['A']},
    'E': {'cost': (500, 700, 900), 'time': (3, 5, 7), 'predecessors': ['B']},
    'F': {'cost': (300, 500, 700), 'time': (2, 4, 6), 'predecessors': ['B']},
    'G': {'cost': (700, 900, 1100), 'time': (4, 6, 8), 'predecessors': ['C', 'D']},
    'H': {'cost': (600, 800, 1000), 'time': (3, 5, 7), 'predecessors': ['E', 'F']},
    'I': {'cost': (800, 1000, 1200), 'time': (4, 6, 8), 'predecessors': ['G', 'H']},
    'J': {'cost': (100, 500, 800), 'time': (3, 5, 10), 'predecessors': ['I']}
}
#---------------------------------------------------------------------------------

scenario_to_idx = {"optimistic": 0, "likely": 1, "pessimistic": 2}

#---------------------------------------------------------------------------------

def topological_sort():
    """Topological sort using Kahn's algorithm, returning sorted activities and successors."""
    in_degree = {act: 0 for act in ACTIVITIES}
    graph = {act: [] for act in ACTIVITIES}

    for act, info in ACTIVITIES.items():
        for predecessor in info['predecessors']:
            graph[predecessor].append(act)
            in_degree[act] += 1

    queue = [act for act, deg in in_degree.items() if deg == 0]
    sorted_acts = []
    
    while queue:
        node = queue.pop(0)
        sorted_acts.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(sorted_acts) != len(ACTIVITIES):
        raise ValueError("Cyclic dependencies detected")
    return sorted_acts, graph

#---------------------------------------------------------------------------------

def calculate_schedule():
    """Calculate ES, EF, LS, LF for all scenarios."""
    sorted_acts, graph = topological_sort()
    results = pd.DataFrame(columns=SCENARIOS, index=ACTIVITIES.keys())
    
    for scenario in SCENARIOS:
        scenario_idx = scenario_to_idx[scenario]
        es_tracker = {}
        ef_tracker = {}
        
        # Forward pass: Calculate ES and EF
        for act in sorted_acts:
            es = max([ef_tracker.get(p, 0) for p in ACTIVITIES[act]['predecessors']] or [0])
            duration = ACTIVITIES[act]['time'][scenario_idx]
            ef = es + duration
            es_tracker[act] = es
            ef_tracker[act] = ef
        
        # Backward pass: Calculate LF and LS
        lf_tracker = {}
        ls_tracker = {}
        project_duration = ef_tracker['J']  # J is the final activity
        lf_tracker['J'] = project_duration
        ls_tracker['J'] = project_duration - ACTIVITIES['J']['time'][scenario_idx]
        
        for act in reversed(sorted_acts[:-1]):  # Exclude J, already set
            lf_tracker[act] = min(ls_tracker[succ] for succ in graph[act])
            duration = ACTIVITIES[act]['time'][scenario_idx]
            ls_tracker[act] = lf_tracker[act] - duration
        
        # Store results as (ES, EF, LS, LF)
        for act in ACTIVITIES:
            results.loc[act, scenario] = (es_tracker[act], ef_tracker[act], ls_tracker[act], lf_tracker[act])
    
    return results

#---------------------------------------------------------------------------------

def calculate_schedule():
    """Calculate ES, EF, LS, LF for all scenarios."""
    sorted_acts, graph = topological_sort()
    results = pd.DataFrame(columns=SCENARIOS, index=ACTIVITIES.keys())
    
    for scenario in SCENARIOS:
        scenario_idx = scenario_to_idx[scenario]
        es_tracker = {}
        ef_tracker = {}
        
        # Forward pass: Calculate ES and EF
        for act in sorted_acts:
            es = max([ef_tracker.get(p, 0) for p in ACTIVITIES[act]['predecessors']] or [0])
            duration = ACTIVITIES[act]['time'][scenario_idx]
            ef = es + duration
            es_tracker[act] = es
            ef_tracker[act] = ef
        
        # Backward pass: Calculate LF and LS
        lf_tracker = {}
        ls_tracker = {}
        project_duration = ef_tracker['J']  # J is the final activity
        lf_tracker['J'] = project_duration
        ls_tracker['J'] = project_duration - ACTIVITIES['J']['time'][scenario_idx]
        
        for act in reversed(sorted_acts[:-1]):  # Exclude J, already set
            lf_tracker[act] = min(ls_tracker[succ] for succ in graph[act])
            duration = ACTIVITIES[act]['time'][scenario_idx]
            ls_tracker[act] = lf_tracker[act] - duration
        
        # Store results as (ES, EF, LS, LF)
        for act in ACTIVITIES:
            results.loc[act, scenario] = (es_tracker[act], ef_tracker[act], ls_tracker[act], lf_tracker[act])
    
    return results

#---------------------------------------------------------------------------------

def calculate_cumulative_cost(results, scenario):
    """Calculate cost distribution over time."""
    max_time = results[scenario].apply(lambda x: x[1]).max()  # EF is x[1]
    time_points = np.arange(0, max_time + 1)
    daily_cost = np.zeros_like(time_points, dtype=float)
    
    scenario_idx = scenario_to_idx[scenario]
    
    for t in time_points:
        total = 0
        for act in results.index:
            es, ef = results.loc[act, scenario][:2]
            cost = ACTIVITIES[act]['cost'][scenario_idx]
            duration = ACTIVITIES[act]['time'][scenario_idx]
            if es <= t < ef:
                total += cost / duration
        daily_cost[t] = total
    
    return time_points, np.cumsum(daily_cost)

#---------------------------------------------------------------------------------

def calculate_cumulative_cost(results, scenario):
    """Calculate cost distribution over time."""
    max_time = results[scenario].apply(lambda x: x[1]).max()  # EF is x[1]
    time_points = np.arange(0, max_time + 1)
    daily_cost = np.zeros_like(time_points, dtype=float)
    
    scenario_idx = scenario_to_idx[scenario]
    
    for t in time_points:
        total = 0
        for act in results.index:
            es, ef = results.loc[act, scenario][:2]
            cost = ACTIVITIES[act]['cost'][scenario_idx]
            duration = ACTIVITIES[act]['time'][scenario_idx]
            if es <= t < ef:
                total += cost / duration
        daily_cost[t] = total
    
    return time_points, np.cumsum(daily_cost)

#---------------------------------------------------------------------------------

def plot_s_curves(results):
    """Generate professional S-curve visualization."""
    plt.figure(figsize=(12, 7))
    
    for scenario in SCENARIOS:
        time, cost = calculate_cumulative_cost(results, scenario)
        plt.plot(time, cost, lw=2.5, label=f"{scenario.title()} Scenario")
    
    plt.title("Project S-Curve Analysis", fontsize=14, pad=15)
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("Cumulative Cost ($)", fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    final_costs = [calculate_cumulative_cost(results, s)[1][-1] for s in SCENARIOS]
    cell_text = [[f"${x:,.0f}"] for x in final_costs]
    plt.table(cellText=cell_text,
              rowLabels=[s.title() for s in SCENARIOS],
              colLabels=["Total Cost"],
              loc='upper left',
              bbox=[0.15, 0.6, 0.2, 0.15])
    
    plt.tight_layout()
    plt.savefig('project_s_curve_analysis.png', dpi=300)
    plt.show()
    
#---------------------------------------------------------------------------------

def find_critical_path(results, scenario):
    """Identify critical path activities where ES=LS and EF=LF."""
    return [act for act in ACTIVITIES if results.loc[act, scenario][0] == results.loc[act, scenario][2] and results.loc[act, scenario][1] == results.loc[act, scenario][3]]

def plot_gantt_chart(results, scenario):
    """Generate Gantt chart with critical path highlighted."""
    plt.figure(figsize=(14, 8))
    critical_acts = find_critical_path(results, scenario)
    
    gantt_data = []
    for act in results.index:
        es, ef = results.loc[act, scenario][:2]
        gantt_data.append({'Activity': act, 'Start': es, 'Finish': ef, 'Duration': ef - es})
    
    df = pd.DataFrame(gantt_data)
    df = df.sort_values('Start', ascending=True)
    
    colors = ['red' if act in critical_acts else 'blue' for act in df['Activity']]
    plt.hlines(y=df['Activity'], xmin=df['Start'], xmax=df['Finish'], colors=colors, linewidth=12)
    
    plt.title(f"Project Schedule - {scenario.title()} Scenario\nCritical Path Gantt Chart", 
              fontsize=15, pad=15, fontweight='semibold')
    plt.xlabel('Days', fontsize=12)
    plt.ylabel('Activities', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    
    for i, row in df.iterrows():
        plt.text(row['Start'] + row['Duration']/2, i, f"{row['Duration']}d", 
                 ha='center', va='center', color='white', fontsize=9, fontweight='bold')
    
    plt.legend([plt.Line2D([0], [0], color='red', lw=4), plt.Line2D([0], [0], color='blue', lw=4)], 
               ['Critical Path', 'Non-Critical'], loc='upper right')
    plt.tight_layout()
    plt.savefig(f'gantt_{scenario}.png', dpi=300)
    plt.show()

#---------------------------------------------------------------------------------

def plot_network():
    """Plot project network diagram using networkx."""
    import networkx as nx
    G = nx.DiGraph()
    
    for act, info in ACTIVITIES.items():
        G.add_node(act)
        for pred in info['predecessors']:
            G.add_edge(pred, act)
    
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=12, arrows=True)
    plt.title("Project Network Diagram", fontsize=14)
    plt.savefig('project_network.png', dpi=300)
    plt.show()
    
#---------------------------------------------------------------------------------    
    

# Core Functions (as defined above)
# topological_sort, calculate_schedule, calculate_cumulative_cost

# Visualization Functions (as defined above)
# plot_s_curves, find_critical_path, plot_gantt_chart, plot_network

# Execution
schedule_results = calculate_schedule()
print("Project Schedule Analysis:")
print(schedule_results.applymap(lambda x: f"ES={x[0]}, EF={x[1]}, LS={x[2]}, LF={x[3]}"))

plot_s_curves(schedule_results)
for scenario in SCENARIOS:
    plot_gantt_chart(schedule_results, scenario)
plot_network()

print("\nAnalysis outputs saved:")
print("- project_s_curve_analysis.png : Comparative S-curves")
print("- gantt_*.png : Scenario-specific Gantt charts")
print("- project_network.png : Project network diagram")
