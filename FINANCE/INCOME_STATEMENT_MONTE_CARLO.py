###############################################
#            IN THE NAME OF ALLAH             #
#  FORECASTING NET INCOME MARGIN USING        #
# MONTE CARLO METHOD WITH EXCEL AND PYTHON    #
# THIS PROGRAM IS WRITTEN BY:                 #
#             SALAR DELAVAR QASHQAI           #
#          SALAR.D.GHASHGHAEI@GMAIL.COM       #
###############################################

import matplotlib.pyplot as plt
import numpy as np

def HISTOGRAM_BOXPLOT_MATPLOTLIB(DATA, XLABEL='X', TITLE='A', COLOR='cyan'):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot histogram
    ax1.hist(DATA, bins='auto', color=COLOR, edgecolor='black')
    ax1.set_xlabel(XLABEL)
    ax1.set_ylabel('Frequency')
    ax1.set_title('Histogram')
    
    # Plot boxplot
    boxplot = ax2.boxplot(DATA, vert=False, patch_artist=True, boxprops=dict(facecolor=COLOR))
    ax2.set_xlabel(XLABEL)
    ax2.set_title('Boxplot')
    
    # Calculate quantiles (10th, 25th, 50th, 75th, 90th)
    quantiles = np.quantile(DATA, [0.1, 0.25, 0.5, 0.75, 0.9])
    
    # Annotate quantiles on the boxplot
    for i, q in enumerate(quantiles):
        ax2.text(q, 1.1, f'Q{i+1}: {q:.2f}', ha='center', va='bottom', color='purple', fontsize=10)
    
    # Calculate mean and standard deviation
    mean = np.mean(DATA)
    std = np.std(DATA)
    
    # Plot mean and mean ± std as dot points
    ax2.scatter([mean - std, mean, mean + std], [1, 1, 1], color='red', s=100, label='Mean ± Std', zorder=5)
    ax2.text(mean - std, 1.2, f'Mean - Std: {mean - std:.2f}', ha='center', va='bottom', color='red', fontsize=10)
    ax2.text(mean, 1.2, f'Mean: {mean:.2f}', ha='center', va='bottom', color='red', fontsize=10)
    ax2.text(mean + std, 1.2, f'Mean + Std: {mean + std:.2f}', ha='center', va='bottom', color='red', fontsize=10)
    
    # Add legend
    ax2.legend(loc='upper right')
    
    # Set the main title for the figure
    fig.suptitle(TITLE, fontsize=16)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

"""
def HISTOGRAM_BOXPLOT_PLOTLY( DATA, XLABEL='X', TITLE='A', COLOR='cyan'):
    # Plotting histogram and boxplot
    import plotly.express as px
    fig = px.histogram(x=DATA, marginal="box", color_discrete_sequence=[COLOR])
    fig.update_layout(title=TITLE, xaxis_title=XLABEL, yaxis_title="Frequency")
    fig.show()
    #fig = px.ecdf(irr, title=TITLE)
    #fig.show()
"""
    
def BETA_PDF(MIN_X, MAX_X, a, b, n):
    import numpy as np
    return MIN_X + (MAX_X - MIN_X) * np.random.beta(a, b, size=n)

NUM = 100000
Revenue = BETA_PDF(200, 400, a=1, b=2, n=NUM)

Direct_Material_Cost = BETA_PDF(100, 200, a=2, b=1, n=NUM)
Direct_Labor_Cost = BETA_PDF(10, 20, a=2, b=1, n=NUM)
Overhead_Cost = BETA_PDF(10, 50, a=2, b=1, n=NUM)
cost_of_goods_sold = Direct_Material_Cost + Direct_Labor_Cost + Overhead_Cost
# Calculate gross profit
gross_profit = Revenue - cost_of_goods_sold
# Calculate operating income
operating_expenses = BETA_PDF(5, 10, a=1, b=2, n=NUM)
operating_income = gross_profit - operating_expenses
# Calculate income before tax
interest_expense = BETA_PDF(5, 10, a=2, b=1, n=NUM)
income_before_tax = operating_income - interest_expense
# Calculate net income
income_tax_expense = income_before_tax * 0.3
net_income = income_before_tax - income_tax_expense
net_income_margin = 100 * net_income / Revenue

#----------------------------------------------------
"""
HISTOGRAM_BOXPLOT_PLOTLY(Revenue, XLABEL='REVENUE',TITLE='REVENUE', COLOR='lime')
HISTOGRAM_BOXPLOT_PLOTLY(Direct_Material_Cost, XLABEL='Direct Material Cost',TITLE='Direct Material Cost', COLOR='orange')
HISTOGRAM_BOXPLOT_PLOTLY(cost_of_goods_sold, XLABEL='COST OF GOODS',TITLE='COST OF GOODS', COLOR='cyan')
HISTOGRAM_BOXPLOT_PLOTLY(net_income_margin, XLABEL='NET INCOME MARGIN (%)',TITLE='NET INCOME MARGIN', COLOR='purple')
"""
HISTOGRAM_BOXPLOT_MATPLOTLIB(Revenue, XLABEL='REVENUE',TITLE='REVENUE', COLOR='lime')
HISTOGRAM_BOXPLOT_MATPLOTLIB(Direct_Material_Cost, XLABEL='Direct Material Cost',TITLE='Direct Material Cost', COLOR='orange')
HISTOGRAM_BOXPLOT_MATPLOTLIB(cost_of_goods_sold, XLABEL='COST OF GOODS',TITLE='COST OF GOODS', COLOR='cyan')
HISTOGRAM_BOXPLOT_MATPLOTLIB(net_income_margin, XLABEL='NET INCOME MARGIN (%)',TITLE='NET INCOME MARGIN', COLOR='purple')

#----------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# Define a range for Revenue
revenue_range = np.linspace(1000, 10000, 400)

# Define the net income margins
net_income_margins = [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]

# Create a meshgrid for plotting
R, N = np.meshgrid(revenue_range, net_income_margins)
COGS = R - (N / 100) * R

# Plot the contour
plt.figure(figsize=(10, 6))
contour = plt.contour(R, COGS, N, levels=net_income_margins)
plt.clabel(contour, inline=True, fontsize=8)
plt.title('Contour Plot of COGS vs. Revenue for Different Net Income Margins')
plt.xlabel('Revenue')
plt.ylabel('Cost of Goods Sold (COGS)')
#plt.semilogy()
plt.show()