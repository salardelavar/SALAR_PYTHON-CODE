
"""
            *****************************************************************************
            *                         >> IN THE NAME OF ALLAH <<                        *
            *  FINANCIAL INCOME STATEMENT OPTIMIZATION BASED ON UNCERTAINTY CONDITION   *
            *        FINDING BEST FITTED PESSIMISTIC AND OPTIMISTIC REVENUE             *
            *        USING MONTE-CARLO METHOD WITH BETA PROBABILTY FUNCTION             *
            *            FINDING OPTIMUM REVENUE WITH USING GRAPH THEORY                *
            *---------------------------------------------------------------------------*
            *        This program is written by Salar Delavar Ghashghaei (Qashqai)      *
            *                     E-mail:salar.d.ghashghaei@gmail.com                   *
            *****************************************************************************
"""

import networkx as nx
import numpy as np
import time
import matplotlib.pyplot as plt
import plotly.express as px
#--------------------------------------------------------------------------------------
def BETA_PDF(min_x, max_x, a, b, n):
    return min_x + (max_x - min_x) * np.random.beta(a, b, size=n)
#--------------------------------------------------------------------------------------
def BUILD_INCOME_GRAPH(Xmin, Xmax, SIZE_PDF):
    # Constructs a financial dependency graph and computes Net Income Margin
    G = nx.DiGraph()

    # Generate financial components as Beta-distributed random variables
    Revenue = BETA_PDF(Xmin, Xmax, a=1, b=2, n=SIZE_PDF)
    Direct_Material_Cost = BETA_PDF(100, 200, a=2, b=1, n=SIZE_PDF)
    Direct_Labor_Cost = BETA_PDF(10, 20, a=2, b=1, n=SIZE_PDF)
    Overhead_Cost = BETA_PDF(10, 50, a=2, b=1, n=SIZE_PDF)
    Operating_Expenses = BETA_PDF(5, 10, a=1, b=2, n=SIZE_PDF)
    Interest_Expense = BETA_PDF(5, 10, a=2, b=1, n=SIZE_PDF)

    # Compute intermediate and final financial metrics
    Cost_of_Goods_Sold = Direct_Material_Cost + Direct_Labor_Cost + Overhead_Cost
    Gross_Profit = Revenue - Cost_of_Goods_Sold
    Operating_Income = Gross_Profit - Operating_Expenses
    Income_Before_Tax = Operating_Income - Interest_Expense
    Income_Tax_Expense = Income_Before_Tax * 0.3
    Net_Income = Income_Before_Tax - Income_Tax_Expense
    Net_Income_Margin = 100 * Net_Income / Revenue

    # Add nodes and edges representing financial dependencies
    G.add_edges_from([
        ("Revenue", "Gross Profit"),
        ("Direct Material Cost", "Cost of Goods Sold"),
        ("Direct Labor Cost", "Cost of Goods Sold"),
        ("Overhead Cost", "Cost of Goods Sold"),
        ("Cost of Goods Sold", "Gross Profit"),
        ("Gross Profit", "Operating Income"),
        ("Operating Expenses", "Operating Income"),
        ("Operating Income", "Income Before Tax"),
        ("Interest Expense", "Income Before Tax"),
        ("Income Before Tax", "Income Tax Expense"),
        ("Income Tax Expense", "Net Income"),
        ("Income Before Tax", "Net Income")
    ])

    return G, Net_Income_Margin
#--------------------------------------------------------------------------------------
def OPTIMIZE_REVENUE(NIM_MIN, NIM_MAX, SIZE_PDF, SIZE_MESH, MIIG, MAIG):
    # Search for the optimal (Xmin, Xmax) to achieve desired Net Income Margin (NIM)

    Xmin_grid = np.linspace(0, MIIG, SIZE_MESH)  # Minimum revenue search space
    Xmax_grid = np.linspace(MIIG, MAIG, SIZE_MESH)  # Maximum revenue search space

    best_pair = (None, None)
    min_diff = np.inf
    it = 0
    STOP = 0
    starttime = time.process_time()

    # **Iterate over different revenue values to find the optimal range**
    for Xmin in Xmin_grid:
        for Xmax in Xmax_grid:
            G, Z = BUILD_INCOME_GRAPH(Xmin, Xmax, SIZE_PDF)

            actual_Q1 = np.quantile(Z, 0.25)
            actual_Q3 = np.quantile(Z, 0.75)
            diff = np.abs(actual_Q1 - NIM_MIN) + np.abs(actual_Q3 - NIM_MAX)

            it += 1
            if actual_Q1 >= NIM_MIN and actual_Q3 <= NIM_MAX:
                print(f"\n\t\t Feasible Solution Found in {it} Iterations - diff: {diff:.4f}")
                print(f"\t\t Best Pair: Xmin = {Xmin:.2f}, Xmax = {Xmax:.2f}")
                print(f"\t\t Q1: {actual_Q1:.2f}, Q3: {actual_Q3:.2f}")
                totaltime = time.process_time() - starttime
                print(f"\t\t Total Iteration Time (s): {totaltime:.2f} \n\n")
                STOP = 1
                break
        if STOP == 1:
            break

    # **Handle case when no feasible solution is found**
    if it == (SIZE_MESH * SIZE_MESH):
        STOP = 0
        Xmin = 0
        Xmax = 0
        print("\n No Feasible Solution Found - Try adjusting Xmin/Xmax or increasing SIZE_PDF/SIZE_MESH")
        print(f"\n Minimum difference found: {diff:.2f}")
        Q25_Z = np.quantile(Z, 0.25)
        Q75_Z = np.quantile(Z, 0.75)
        print(f" Net Income Margin (Q25): {Q25_Z:.2f}, Net Income Margin (Q75): {Q75_Z:.2f}")

    return STOP, Xmin, Xmax
#--------------------------------------------------------------------------------------
# Execute the optimization algorithm
NIM_MIN, NIM_MAX = 1, 100  # Desired Net Income Margin range
SIZE_PDF = 1000  # Number of samples
SIZE_MESH = 50  # Grid size for searching revenue values
MIIG, MAIG, SIZE_PDF = 200, 500, 1000  # Range of revenue to explore

OPTIMIZE_REVENUE(NIM_MIN, NIM_MAX, SIZE_PDF, SIZE_MESH, MIIG, MAIG)
#--------------------------------------------------------------------------------------

# Build Financial Dependency Graph & Compute Net Income Margin
G, NIM = BUILD_INCOME_GRAPH(MIIG, MAIG, SIZE_PDF)

# Graph Visualization
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G)  # Define graph layout
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", edge_color="gray", font_size=10)
plt.title("Graph Representation of Income Statement")
plt.show()

# Net Income Margin Distribution Visualization
fig = px.histogram(x=NIM, marginal="box", color_discrete_sequence=['cyan'])
fig.update_layout(title="Net Income Margin Distribution", xaxis_title="Net Income Margin (%)", yaxis_title="Frequency")
fig.show()

#--------------------------------------------------------------------------------------


