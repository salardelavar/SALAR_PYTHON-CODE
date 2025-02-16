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
def HISROGRAM_BOXPLOT_MATPLOTLIB(X, HISTO_COLOR, LABEL):
    import numpy as np
    import matplotlib.pyplot as plt
    X = np.array(X)
    print("-------------------------")
    from scipy.stats import skew, kurtosis
    MINIMUM = np.min(X)
    MAXIMUM = np.max(X)
    #MODE = max(set(X), key=list(X).count)
    MEDIAN = np.quantile(X, .50)#q2
    MEAN = np.mean(X)
    STD = np.std(X)
    q1 = np.quantile(X, .25)
    q3 = np.quantile(X, .75)
    SKEW = skew(X)
    KURT = kurtosis(X)
    #SKEW = (MEAN - MODE) / STD
    #KURT = (np.mean((X - MEAN)**4) / STD**4)
    # Estimate confidence intervals of the output variable
    lower_bound = np.quantile(X, .05)
    upper_bound = np.quantile(X, .95)
    print("Box-Chart Datas: ")
    print(f'Minimum: {MINIMUM:.6e}')
    print(f'First quartile: {q1:.6e}')
    #print(f'Mode: {MODE:.6e}')
    print(f'Median: {MEDIAN:.6e}')
    print(f'Mean: {MEAN:.6e}')
    print(f'Std: {STD:.6e}')
    print(f'Third quartile: {q3:.6e}')
    print(f'Maximum: {MAXIMUM :.6e}')
    print(f'Skewness: {skew(X) :.6e}')
    print(f'kurtosis: {kurtosis(X) :.6e}')
    print(f"90% Confidence Interval: ({lower_bound:.6e}, {upper_bound:.6e})")
    print("-------------------------")

    plt.figure(figsize=(10,6))
    # Plot histogram of data
    count, bins, ignored = plt.hist(X, bins=100, color=HISTO_COLOR, density=True, align='mid')#, edgecolor="black"
    
    # Plot lognormal PDF
    x = np.linspace(min(bins), max(bins), 1000)
    pdf = (np.exp(-(x - MEAN)**2 / (2 * STD**2)) / (STD * np.sqrt(2 * np.pi)))
    plt.plot(x, pdf, linewidth=2, color='r', label="Normal PDF")
    
    # Plot vertical lines for risk measures
    plt.axvline(q1, color="black", linestyle="--", label=f"Quantile 0.25: {q1:.6e}")
    plt.axvline(MEDIAN, color="green", linestyle="--", label=f"Median: {MEDIAN:.6e}")
    plt.axvline(q3, color="black", linestyle="--", label=f"Quantile 0.75: {q3:.6e}")
    #plt.axvline(MODE, color="purple", linestyle="--", label=f"Mode: {MODE:.6e}")
    plt.axvline(MEAN, color="red", linestyle="--", label=f"Mean: {MEAN:.6e}")
    plt.axvline(MEAN-STD, color="blue", linestyle="--", label=f"Mean-Std: {MEAN-STD:.6e}")
    plt.axvline(MEAN+STD, color="blue", linestyle="--", label=f"Mean+Std: {MEAN+STD:.6e}")
    plt.xlabel(LABEL)
    plt.ylabel("Frequency")
    prob = np.sum(X > 0) / len(X)
    plt.title(f"Histogram - Probability of Positive {LABEL} is {100*prob:.2f} %")
    plt.legend()
    #plt.grid()
    plt.show()

    #Plot boxplot with outliers
    plt.figure(figsize=(10,6))
    plt.boxplot(X, vert=0)
    # Write the quartile data on the chart
    plt.text(q1, 1.05, f" Q1: {q1:.6e}")
    plt.text(MEDIAN, 1.1, f" Q2: {MEDIAN:.6e}")
    plt.text(q3, 1.05, f" Q3: {q3:.6e}")
    #plt.text(MODE, 1.15, f" Mode: {MODE:.6e}")
    
    #plt.text(MEAN, 0.9, f" Mean: {MEAN:.6e}")
    #plt.text(MEAN-STD, 0.9, f" Mean-Std: {MEAN-STD:.6e}")
    #plt.text(MEAN+STD, 0.9, f" Mean+Std: {MEAN+STD:.6e}")
    plt.scatter(MEAN, 1, color="red", marker="+", s=200, label=f"Mean: {MEAN:.6e}")
    plt.scatter(MEAN-STD, 1, color="green", marker="X", s=200, label=f"Mean-Std: {MEAN-STD:.6e}")
    plt.scatter(MEAN+STD, 1, color="blue", marker="*", s=200, label=f"Mean+Std:  {MEAN+STD:.6e}")
    plt.xlabel(LABEL)
    plt.ylabel("Data")
    plt.title(f"Boxplot of {LABEL}")
    plt.legend()
    plt.grid()
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
HISROGRAM_BOXPLOT_MATPLOTLIB(Revenue, HISTO_COLOR='lime', LABEL='REVENUE')
HISROGRAM_BOXPLOT_MATPLOTLIB(Direct_Material_Cost, HISTO_COLOR='orange', LABEL='Direct Material Cost')
HISROGRAM_BOXPLOT_MATPLOTLIB(cost_of_goods_sold, HISTO_COLOR='cyan', LABEL='COST OF GOODS')
HISROGRAM_BOXPLOT_MATPLOTLIB(net_income_margin, HISTO_COLOR='purple', LABEL='NET INCOME MARGIN (%)')

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