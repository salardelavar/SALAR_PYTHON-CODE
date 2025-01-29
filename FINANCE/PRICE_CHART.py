import random
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------

def Generate_Prices(start_price, Volatility, Direction_Bias, num_days):
    import random
    prices = [start_price]
    changes = [0]
    for i in range(num_days-1):
        Var = random.uniform(-1, 1)*random.uniform(0, 1)*Volatility
        price = prices[-1] *(1 + Var + Direction_Bias)
        prices.append(price)
        changes.append(prices[-1] / prices[i-1] - 1)
    return prices, changes

start_price = 100
Volatility = 0.05
Direction_Bias = 0.0001
num_days = 1000
prices, changes = Generate_Prices(start_price, Volatility, Direction_Bias, num_days)


# Plot the prices and the indicators
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(prices)), prices, label="Prices", color="black")
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Price Chart")
plt.legend()
#plt.legend(loc=(1.01, 0))
plt.semilogy()
plt.show()

#------------------------------------------------------------------------------

def Generate_Prices_Priods(prices, priods):
    t = [];p = [];changes = [];last_price = prices[0];
    for i in range(0, len(prices), priods):
        t.append(i) # use append instead of assignment
        p.append(prices[i]) # use append instead of assignment
        changes.append(prices[i] / last_price - 1)
        last_price = prices[i]
        #print(t[-1],p[-1]) # print the last elements of the lists
    return  t, p, changes

times_7, prices_7, changes_7 = Generate_Prices_Priods(prices, 7)
times_30, prices_30, changes_30  = Generate_Prices_Priods(prices, 30)
times_90, prices_90, changes_90  = Generate_Prices_Priods(prices, 90)
times_220, prices_220, changes_220  = Generate_Prices_Priods(prices, 220)
# Plot the prices and the indicators
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(prices)), prices, label="Prices 1 Time", color="black")
plt.plot(times_7, prices_7, label="Prices 7 Times")
plt.plot(times_30, prices_30, label="Prices 30 Times")
plt.plot(times_90, prices_90, label="Prices 90 Times")
plt.plot(times_220, prices_220, label="Prices 220 Times")
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Price with Different Times")
plt.legend()
#plt.legend(loc=(1.01, 0))
plt.semilogy()
plt.show()


plt.figure(figsize=(10,6))
# Plot the boxplot with patch_artist=True and boxprops
plt.boxplot(changes, positions=[1], vert=0)
plt.boxplot(changes_7, positions=[2], vert=0)
plt.boxplot(changes_30, positions=[3], vert=0)
plt.boxplot(changes_90, positions=[4], vert=0)
plt.boxplot(changes_220, positions=[5], vert=0)
plt.xlabel("change")
plt.ylabel("sample")
plt.title("Box plot of sample for changes in different periods")
plt.grid()
plt.legend(['Change 1 Time', 'Change 7 Time','Change 30 Time','Change 90 Time','Change 220 Time'])
plt.show()


#------------------------------------------------------------------------------

LL =np.arange(len(prices))# Length of Prices

def RISK_REWARD_RATIO(prices):
    stop_loss = np.min(prices)
    take_profit = np.max(prices)
    entry_price = prices[-1]
    risk = entry_price - stop_loss
    reward = take_profit - entry_price
    return reward / risk

a = RISK_REWARD_RATIO(prices)
print(f'RISK-REWARD RATIO: {a:.2f}')

# Plot the prices and the indicators
plt.figure(figsize=(10, 6))
plt.plot(LL, prices, label="Prices", color="black")

plt.axhline(y=np.min(prices), linestyle="--", color="blue", label="RISK")
plt.text(0, np.min(prices), f" RISK: {np.min(prices):.2f}") # write the equation of the line on the plot

plt.axhline(y=np.max(prices), linestyle="--", color="red", label="REWARD")
plt.text(0, np.max(prices), f" REWARD: {np.max(prices):.2f}") # write the equation of the line on the plot

plt.axhline(y=prices[-1], linestyle="--", color="green", label="ENTRY PRICE")
plt.text(0, prices[-1], f" ENTRY PRICE: {prices[-1]:.2f}") # write the equation of the line on the plot


plt.xlabel("Time")
plt.ylabel("Price")
plt.title(f'RISK-REWARD RATIO: {a:.2f}')
#plt.legend(loc=(1.01, 0))
plt.semilogy()
plt.show()

#------------------------------------------------------------------------------
###########################################################################
################ Calculate Moving Average

def Moving_Average(prices, periods):
    import pandas as pd
    import numpy as np
    # Use pandas.Series.rolling() method to calculate moving averages
    X = pd.Series(prices).rolling(window=periods).mean()
    X = np.array(X)
    return X


MA_20 = Moving_Average(prices, 20)
MA_50 = Moving_Average(prices, 50)
MA_100 = Moving_Average(prices, 100)
MA_200 = Moving_Average(prices, 200)
MA_400 = Moving_Average(prices, 400)

# Plot the prices and the moving averages
plt.figure(figsize=(10, 6))
plt.plot(LL, prices, label="Prices", color="black")
plt.plot(LL, MA_20, label=f"MA 20 : {MA_20[-1]:.2f}")
plt.plot(LL, MA_50, label=f'MA 50 : {MA_50[-1]:.2f}')
plt.plot(LL, MA_100, label=f'MA 100 : {MA_100[-1]:.2f}')
plt.plot(LL, MA_200, label=f'MA 200 : {MA_200[-1]:.2f}')
plt.plot(LL, MA_400, label=f'MA 400 : {MA_400[-1]:.2f}')
plt.title("Price Moving Average Line")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.semilogy()
plt.show()
#------------------------------------------------------------------------------
###########################################################################
################ Calculate Quantile Moving Average

def Quantile_Indictor(prices, periods, ratio):
    import pandas as pd
    import numpy as np
    # Use pandas.Series.rolling() method to calculate quantile indictor
    X =  pd.Series(prices).rolling(window=periods).quantile(ratio)
    X = np.array(X)
    return X

QI_20 = Quantile_Indictor(prices, 20, .5)
QI_50 = Quantile_Indictor(prices, 50, .5)
QI_100 = Quantile_Indictor(prices, 100, .5)
QI_200 = Quantile_Indictor(prices, 200, .5)
QI_400 = Quantile_Indictor(prices, 400, .5)

# Plot the prices and the moving averages
plt.figure(figsize=(10, 6))
plt.plot(LL, prices, label="Prices", color="black")
plt.plot(LL, QI_20, label=f"QMA 20 : {QI_20[-1]:.2f}")
plt.plot(LL, QI_50, label=f'QMA 50 : {QI_50[-1]:.2f}')
plt.plot(LL, QI_100, label=f'QMA 100 : {QI_100[-1]:.2f}')
plt.plot(LL, QI_200, label=f'QMA 200 : {QI_200[-1]:.2f}')
plt.plot(LL, QI_400, label=f'QMA 400 : {QI_400[-1]:.2f}')
plt.title("Price Quantile Line")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.semilogy()
plt.show()
#------------------------------------------------------------------------------

###########################################################################
################ Calculate Linear Regression Channel Indictor

def LRC_INDICATOR(prices, period):
    import numpy as np
    import matplotlib.pyplot as plt
    # calculate the x and y values for the linear regression
    x = np.arange(period)
    y = prices[:period]
    # calculate the slope and intercept of the best fit line
    slope, intercept = np.polyfit(x, y, 1)
    # calculate the LRC values for the window
    lrc = slope * x + intercept
    # initialize the LRC list with the first window values
    lrc_list = list(lrc)
    # loop through the remaining prices and update the LRC values
    for i in range(period, len(prices)):
        # shift the x and y values by one
        x = x + 1
        y = prices[i-period+1:i+1]
        # recalculate the slope and intercept
        slope, intercept = np.polyfit(x, y, 1)
        # append the latest LRC value to the list
        lrc_list.append(slope * x[-1] + intercept)
    # return the LRC list
    return lrc_list

# call the function with the generated prices and a period
LRC_20 = LRC_INDICATOR(prices, 20)
LRC_50 = LRC_INDICATOR(prices, 50)
LRC_100 = LRC_INDICATOR(prices, 100)
LRC_200 = LRC_INDICATOR(prices, 200)
LRC_400 = LRC_INDICATOR(prices, 400)

# plot the prices and the LRC indicator
plt.figure(figsize=(10, 6))
plt.plot(LL, prices, label="Prices", color="black")
plt.plot(LL, LRC_20, label=f"LRC 20 : {LRC_20[-1]:.2f}")
plt.plot(LL, LRC_50, label=f'LRC 50 : {LRC_50[-1]:.2f}')
plt.plot(LL, LRC_100, label=f'LRC 100 : {LRC_100[-1]:.2f}')
plt.plot(LL, LRC_200, label=f'LRC 200 : {LRC_200[-1]:.2f}')
plt.plot(LL, LRC_400, label=f'LRC 400 : {LRC_400[-1]:.2f}')
plt.title("Linear Regression Channel Indictor")
plt.legend()
plt.semilogy()
plt.show()

#------------------------------------------------------------------------------

###########################################################################
################ Calculate Zig Zag Indictor line

def Zig_Zag(prices, threshold):
    # Initialize the zig zag points list with the first price
    zz = [prices[0]]
    # Initialize the trend direction
    trend = None
    # Loop through the prices from index 1 to the end
    for i in range(1, len(prices)):
        # Calculate the percentage change from the last zig zag point
        change = (prices[i] - zz[-1]) / zz[-1]
        # If the trend direction is not set yet
        if trend is None:
            # If the change is greater than the threshold, set the trend to up and append the price as a zig zag point
            if change > threshold:
                trend = "up"
                zz.append(prices[i])
            # If the change is less than the negative threshold, set the trend to down and append the price as a zig zag point
            elif change < -threshold:
                trend = "down"
                zz.append(prices[i])
        # If the trend direction is up
        elif trend == "up":
            # If the change is less than the negative threshold, change the trend to down and append the price as a zig zag point
            if change < -threshold:
                trend = "down"
                zz.append(prices[i])
            # If the change is greater than zero and the price is higher than the last zig zag point, update the last zig zag point to the current price
            elif change > 0 and prices[i] > zz[-1]:
                zz[-1] = prices[i]
        # If the trend direction is down
        elif trend == "down":
            # If the change is greater than the threshold, change the trend to up and append the price as a zig zag point
            if change > threshold:
                trend = "up"
                zz.append(prices[i])
            # If the change is less than zero and the price is lower than the last zig zag point, update the last zig zag point to the current price
            elif change < 0 and prices[i] < zz[-1]:
                zz[-1] = prices[i]
    return zz


# Calculate the zig zag points with a 5% threshold
zz = Zig_Zag(prices, 0.05)

# Create a list of numbers from 0 to the length of prices
time = list(range(len(prices)))
zz_time = [i for i in range(len(prices)) if prices[i] in zz]

# Plot the prices and the Zig Zag
plt.figure(figsize=(10, 6))
plt.plot(LL, prices, label="Prices", color="black")
plt.plot(zz_time, zz, label="Zig Zag", marker="o", color="red")
plt.title("Price Zig Zag Line")
plt.xlabel("Time")
plt.ylabel("Price")
plt.semilogy()
plt.legend()
plt.show()

#------------------------------------------------------------------------------

###########################################################################
### FIND MAXIMUM AND MINIMUM PIVOTS ON CHARTS
def Find_Pivots(prices, window):
    # Find the local maxima and minima of the prices within a given window
    peaks = []
    troughs = []
    for i in range(window, len(prices) - window):
        if prices[i] == np.max(prices[i-window:i+window+1]):
            peaks.append((i, prices[i]))
        elif prices[i] == np.min(prices[i-window:i+window+1]):
            troughs.append((i, prices[i]))

    # Connect the peaks and troughs with lines and find the intersections
    intersections = []
    for i in range(len(peaks) - 1):
        for j in range(len(troughs) - 1):
            # Calculate the slopes and intercepts of the lines
            m1 = (peaks[i+1][1] - peaks[i][1]) / (peaks[i+1][0] - peaks[i][0])
            b1 = peaks[i][1] - m1 * peaks[i][0]
            m2 = (troughs[j+1][1] - troughs[j][1]) / (troughs[j+1][0] - troughs[j][0])
            b2 = troughs[j][1] - m2 * troughs[j][0]

            # Check if the lines intersect within the window
            if m1 != m2:
                x = (b2 - b1) / (m1 - m2)
                y = m1 * x + b1
                if x > peaks[i][0] and x < peaks[i+1][0] and x > troughs[j][0] and x < troughs[j+1][0]:
                    intersections.append((x, y))

    # Plot the prices, peaks, troughs and intersections
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(prices)), prices, color='black')
    #plt.scatter([p[0] for p in peaks], [p[1] for p in peaks], color='red', label='Max. Pivots')
    #plt.scatter([t[0] for t in troughs], [t[1] for t in troughs], color='green', label='Min. Pivots')
    #plt.scatter([i[0] for i in intersections], [i[1] for i in intersections], color='blue')
    plt.plot([p[0] for p in peaks], [p[1] for p in peaks], color='red', marker='o', label='Max. Pivots')
    plt.plot([t[0] for t in troughs], [t[1] for t in troughs], color='green', marker='o', label='Min. Pivots')
    plt.plot([i[0] for i in intersections], [i[1] for i in intersections], marker='o', color='blue')
    plt.title("Minimum and Maximum Pivots")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.semilogy()
    plt.legend()
    plt.show()

Find_Pivots(prices, 50)

###########################################################################

def Plot_Trend_Line(prices):
    import matplotlib.pyplot as plt
    # Find the peak and trough indices of the prices
    peaks = [i for i in range(1, len(prices)-1) if prices[i] > prices[i-1] and prices[i] > prices[i+1]]
    troughs = [i for i in range(1, len(prices)-1) if prices[i] < prices[i-1] and prices[i] < prices[i+1]]
    # Find the slope and intercept of the trend lines
    slope = (prices[peaks[-1]] - prices[peaks[0]]) / (peaks[-1] - peaks[0])
    intercept = prices[peaks[0]] - slope * peaks[0]
    # Plot the prices and the trend lines
    plt.figure(figsize=(10,6))
    plt.plot(np.arange(len(prices)), prices, label="Prices", color="black")
    plt.plot([slope * i + intercept for i in range(len(prices))], label="Trend line", color="red")
    plt.legend()
    plt.title("Trading line")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.show()

Plot_Trend_Line(prices)

###########################################################################

def REGRESSION_TREND(prices, ORDER):
    import numpy as np
    import matplotlib.pyplot as plt
    days = np.arange(1, len(prices)+1)
    #plot the prices
    plt.figure(figsize=(10,6))
    plt.plot(days, prices,color="black")
    #fit a linear trendline
    z = np.polyfit(days, prices, ORDER)
    p = np.poly1d(z)
    #plot the trendline
    plt.plot(days, p(days), color="red")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title(f"Price Trend Ploynominal Regression Line with Order {ORDER}")
    plt.semilogy()
    #show the plot
    plt.show()

REGRESSION_TREND(prices, 3)

#------------------------------------------------------------------------------

###########################################################################

def Price_Fibonacci_Indicator(prices, fibo_ratios):
    # Define the Fibonacci ratios
    # A function to calculate the price Fibonacci indicator
    # based on the number of days in the series
    # based on the highest and lowest price in the series
    high = np.max(prices); low = np.min(prices); L = high - low;
    levels = []
    for i in range(0, len(fibo_ratios)):
        # Calculate the Fibonacci projection levels
        level = low + L * fibo_ratios[i]
        levels.append(level)
    return levels

def Time_Fibonacci_Indicator(num_days, fibo_ratios):
    # Define the Fibonacci ratios
    # A function to calculate the time Fibonacci indicator
    # based on the number of days in the series
    # based on the highest and lowest time in the series
    L = num_days
    levels = []
    for i in range(0,  len(fibo_ratios)):
        # Calculate the Fibonacci projection levels
        level = L * fibo_ratios[i]
        levels.append(level)
    return levels

fibo_ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
# Calculate the price and time Fibonacci indicators
price_levels = Price_Fibonacci_Indicator(prices, fibo_ratios)
time_levels = Time_Fibonacci_Indicator(len(prices), fibo_ratios)


# Plot the prices and the indicators
plt.figure(figsize=(10, 6))
plt.plot(prices, label="Prices", color="black")

for i, level in enumerate(price_levels):
    plt.axhline(y=level, linestyle="--", color="blue", label=f"Fib Ratio {fibo_ratios[i]}")
    plt.text(0, level, f"{fibo_ratios[i]:.3f}  -  {level:.2f}") # write the equation of the line on the plot

for i, level in enumerate(time_levels):
    plt.axvline(x=level, linestyle="--", color="red", label=f"Fib Ratio {fibo_ratios[i]}")
    #plt.text(level, 0, f"{fibo_ratios[i]:.3f}  -  {level:.2f}") # write the equation of the line on the plot

plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Fibonacci Sequence")
plt.legend(loc=(1.01, 0))
plt.semilogy()
plt.show()


#------------------------------------------------------------------------------

###########################################################################
##############  rsi indictor

def RSI_INDICATOR(prices, periods, ma_period, k):
    import numpy as np
    # Calculate the price changes
    changes = np.diff(prices)

    # Separate the positive and negative changes
    gains = np.where(changes > 0, changes, 0)
    losses = np.where(changes < 0, -changes, 0)

    # Calculate the exponential moving averages of gains and losses
    avg_gain = np.empty(len(prices))
    avg_loss = np.empty(len(prices))

    avg_gain[:periods] = np.mean(gains[:periods-1])
    avg_loss[:periods] = np.mean(losses[:periods-1])

    for i in range(periods, len(prices)):
        avg_gain[i] = (avg_gain[i-1]*(periods-1) + gains[i-1])/periods
        avg_loss[i] = (avg_loss[i-1]*(periods-1) + losses[i-1])/periods


    # Calculate the RSI
    rsi = 100 * (avg_gain / (avg_gain + avg_loss))
    rsi_ma = np.empty(len(prices))
    rsi_std = np.empty(len(prices))
    lower_band = np.empty(len(prices))
    upper_band = np.empty(len(prices))
    ratio = np.empty(len(prices))
    rsi_ma[:periods] = np.mean(rsi[:periods-1])
    rsi_std[:periods] = np.std(rsi[:periods-1])
    for i in range(periods, len(prices)):# moving average of rsi
        rsi_ma[i] = ( rsi_ma[i-1]*(periods-1) + rsi[i-1])/periods
        rsi_std[i] = np.std(rsi[i-periods+1:i+1])

    lower_band = rsi_ma - k * rsi_std
    upper_band = rsi_ma + k * rsi_std

    for i in range(periods, len(prices)):
        ratio[i] = (rsi[i] - lower_band[i]) / (upper_band[i] - lower_band[i])

    return rsi, rsi_ma, lower_band, upper_band, ratio

##############  cci indictor

def CCI_INDICATOR(prices, period):
    import numpy as np
    prices = np.array(prices).reshape(-1, 1)
    # Calculate the typical price
    tp = np.mean(prices, axis=1)
    # Calculate the moving average of the typical price
    ma = np.convolve(tp, np.ones(period)/period, mode='valid')
    # Calculate the mean deviation
    md = np.zeros_like(ma)
    for i in range(period, len(tp)):
        md[i-period] = np.mean(np.abs(tp[i-period:i] - ma[i-period]))
    # Calculate the CCI
    cci = (tp[period-1:] - ma) / (0.015 * md)
    return cci


##############  bollinger bands indictor

def Bollinger_Bands(prices, periods, k):
    import numpy as np
    # Calculate the simple moving average
    sma = np.convolve(prices, np.ones(periods)/periods, mode='valid')

    # Calculate the standard deviation
    std = np.std(prices[:periods-1])

    # Initialize the upper and lower bands
    ma_band = np.empty(len(prices))
    upper_band = np.empty(len(prices))
    lower_band = np.empty(len(prices))

    ma_band[:periods-1] = sma[0]
    upper_band[:periods-1] = sma[0] + k*std
    lower_band[:periods-1] = sma[0] - k*std
    ratio = np.zeros(len(prices))
    # Update the bands for each period
    for i in range(periods-1, len(prices)):
        std = np.std(prices[i-periods+1:i+1])
        ma_band[i] = sma[i-periods+1]
        upper_band[i] = sma[i-periods+1] + k*std
        lower_band[i] = sma[i-periods+1] - k*std
        ratio[i] = (prices[i] - lower_band[i]) / (upper_band[i] - lower_band[i])
        
    return ma_band, upper_band, lower_band, ratio

############## MACD indictor

def MACD(prices, short_period, long_period, signal_period):
    import numpy as np
    import pandas as pd # added this line
    short_ema = pd.Series(prices).ewm(span=short_period, adjust=False).mean() # changed this line
    long_ema = pd.Series(prices).ewm(span=long_period, adjust=False).mean() # changed this line
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_line = np.array(macd_line)# change array pandas to numpy
    signal_line = np.array(signal_line)# change array pandas to numpy
    return macd_line, signal_line

############## Momentum indictor
"""
def Momentum(changes, period):
    momentum = []
    for i in range(len(changes)):
        if i < period:
            momentum.append(0)
        else:
            momentum.append(sum(changes[i-period+1:i+1]))
    momentum = np.array(momentum)
    return momentum
"""
def Momentum(prices, period):
    momentum = []
    for i in range(len(prices)):
        if i < period:
            momentum.append(prices[1] - prices[0])
        else:
            momentum.append(prices[i] - prices[i-period])
    momentum = np.array(momentum)
    return momentum


##############  roc indictor

def ROC_INDICATOR(prices, period):
    import numpy as np
    roc = np.zeros(len(prices))
    for i in range(period, len(prices)):
        roc[i] = (prices[i] - prices[i-period]) / prices[i-period]
    return roc

##############  z-score Indicator
# Calculate z-score function for period
def Z_Score_Indicator(x, period):
    # Ensure x is a numpy array
    x = np.array(x)
    # Initialize an empty list to store z-scores
    z_scores = []

    # Loop over the array to calculate the rolling mean and standard deviation
    for i in range(period - 1, len(x)):
        # Define the rolling window
        window = x[i - period + 1:i + 1]
        # Calculate mean and standard deviation of the window
        mean = np.mean(window)
        std = np.std(window)
        # Calculate the z-score and append to the list
        # Check if std is not zero to avoid division by zero
        if std != 0:
            z_score = (x[i] - mean) / std
            z_scores.append(z_score)
        else:
            # If std is zero, append 0 as the z-score
            z_scores.append(0)

    # Convert the list of z-scores to a numpy array
    z_scores = np.array(z_scores)
    return z_scores




# Calculate the  RSI and bollinger bands and Roc and MACD and Momentum
rsi, rsi_ma, rsi_lower_band, rsi_upper_band, rsi_ratio = RSI_INDICATOR(prices, 14, 20, 2)
rsi_low = np.quantile(rsi, 0.025)
rsi_mid = np.quantile(rsi, 0.50)
rsi_high = np.quantile(rsi, 0.975)
ma_band_RSI, upper_band_RSI, lower_band_RSI, ratio_RSI = Bollinger_Bands(rsi_ratio, 20, 2)

cci = CCI_INDICATOR(prices, 20)
cci_low = np.quantile(cci, 0.025)
cci_mid = np.quantile(cci, 0.50)
cci_high = np.quantile(cci, 0.975)

ma_band, upper_band, lower_band, ratio = Bollinger_Bands(prices, 20, 2)

ma_band02, upper_band02, lower_band02, ratio02 = Bollinger_Bands(ratio, 20, 2)

roc = 100*ROC_INDICATOR(prices, 20)
roc_low = np.quantile(roc, 0.025)
roc_mid = np.quantile(roc, 0.50)
roc_high = np.quantile(roc, 0.975)

macd_line, signal_line = MACD(prices, 12, 26, 9)
macd = macd_line;
macd_low = np.quantile(macd, 0.025)
macd_mid = np.quantile(macd, .50)
macd_high = np.quantile(macd, 0.975)

mom = Momentum(prices, 10)
mom_low = np.quantile(mom, 0.025)
mom_mid = np.quantile(mom, 0.50)
mom_high = np.quantile(mom, 0.975)

z_score = Z_Score_Indicator(prices, period = 20)
z_score_low = np.quantile(z_score, 0.025)
z_score_mid = np.quantile(z_score, 0.50)
z_score_high = np.quantile(z_score, 0.975)


# Plot the datas
plt.figure(figsize=(16,18))
plt.subplot(9,1,1)
plt.plot(100*ratio, label=f'BB%: {100*ratio[-1]:.2f}', color='black')
plt.axhline(100, color='red', linestyle='--')

plt.axhline(0, color='green', linestyle='--')
plt.legend(loc="lower left")
plt.title('Bollinger Bands')


plt.subplot(9,1,2)
plt.plot(100*ratio02, label=f'BB%: {100*ratio02[-1]:.2f}', color='black')
plt.axhline(100, color='red', linestyle='--')

plt.axhline(0, color='green', linestyle='--')
plt.legend(loc="lower left")
plt.title('Bollinger Bands of Bollinger Bands')
"""
plt.plot(prices, label='Price', color='black')
plt.plot(ma_band, label='ma', color='red')
plt.plot(upper_band, label='Upper band', color='orange')
plt.plot(lower_band, label='Lower band', color='orange')
"""

plt.subplot(9,1,3)
plt.plot(rsi, label=f'RSI: {rsi[-1]:.2f}', color='black')
plt.plot(rsi_ma, label=f'RSI MA: {rsi_ma[-1]:.2f}', color='red')
plt.plot(rsi_lower_band, label=f'RSI MA-STD: {rsi_lower_band[-1]:.2f}', color='lime')
plt.plot(rsi_upper_band, label=f'RSI MA+STD: {rsi_upper_band[-1]:.2f}', color='lime')
plt.axhline(rsi_low, color='green', linestyle='--')
plt.axhline(rsi_mid, color='cyan', linestyle='--')
plt.axhline(rsi_high, color='red', linestyle='--')
plt.text(0, rsi_low, f"LOW: {rsi_low:.2f}") # write the equation of the line on the plot
plt.text(0, rsi_mid, f"MID: {rsi_mid:.2f}") # write the equation of the line on the plot
plt.text(0, rsi_high, f"HIGH: {rsi_high:.2f}") # write the equation of the line on the plot
plt.title('Relative Strength Index')
plt.legend(loc="lower left")


plt.subplot(9,1,4)
plt.plot(100*ratio_RSI, label=f'RSI BB%: {100*ratio_RSI[-1]:.2f}', color='black')
plt.axhline(100, color='red', linestyle='--')

plt.axhline(0, color='green', linestyle='--')
plt.legend(loc="lower left")
plt.title('Bollinger Bands of RSI')


plt.subplot(9,1,5)
plt.plot(roc, color="black", label=f"ROC: {roc[-1]:.2f}")
plt.axhline(roc_low, color='green', linestyle='--')
plt.axhline(roc_mid, color='cyan', linestyle='--')
plt.axhline(roc_high, color='red', linestyle='--')
plt.text(0, roc_low, f"LOW: {roc_low:.2f}") # write the equation of the line on the plot
plt.text(0, roc_mid, f"MID: {roc_mid:.2f}") # write the equation of the line on the plot
plt.text(0, roc_high, f"HIGH: {roc_high:.2f}") # write the equation of the line on the plot
plt.title("Roc")
plt.legend(loc="lower left")

plt.subplot(9,1,6)
plt.plot(cci, color="black", label=f"CCI: {cci[len(cci)-2]:.2f}")
plt.axhline(cci_low, color='green', linestyle='--')
plt.axhline(cci_mid, color='cyan', linestyle='--')
plt.axhline(cci_high, color='red', linestyle='--')
plt.text(0, cci_low, f"LOW: {cci_low:.2f}") # write the equation of the line on the plot
plt.text(0, cci_mid, f"MID: {cci_mid:.2f}") # write the equation of the line on the plot
plt.text(0, cci_high, f"HIGH: {cci_high:.2f}") # write the equation of the line on the plot
plt.title("CCI")
plt.legend(loc="lower left")

plt.subplot(9,1,7)
plt.plot(macd_line, color="black")
plt.plot(signal_line, color="red")
plt.axhline(macd_low, color='green', linestyle='--')
plt.axhline(macd_mid, color='cyan', linestyle='--')
plt.axhline(macd_high, color='red', linestyle='--')
plt.text(0, macd_low, f"LOW: {macd_low:.2f}") # write the equation of the line on the plot
plt.text(0, macd_mid, f"MID: {macd_mid:.2f}") # write the equation of the line on the plot
plt.text(0, macd_high, f"HIGH: {macd_high:.2f}") # write the equation of the line on the plot
plt.title(f"MACD")
plt.legend(loc="lower left")

plt.subplot(9,1,8)
plt.plot(mom, color="black", label=f"Momentum: {mom[-1]:.2f}")
plt.axhline(mom_low, color='green', linestyle='--')
plt.axhline(mom_mid, color='cyan', linestyle='--')
plt.axhline(mom_high, color='red', linestyle='--')
plt.text(0, mom_low, f"LOW: {mom_low:.2f}") # write the equation of the line on the plot
plt.text(0, mom_mid, f"MID: {mom_mid:.2f}") # write the equation of the line on the plot
plt.text(0, mom_high, f"HIGH: {mom_high:.2f}") # write the equation of the line on the plot
plt.title("Momentum")
plt.legend(loc="lower left")

plt.subplot(9,1,9)
plt.plot(np.arange(len(z_score)), z_score, color="black", label=f"z score: {z_score[-1]:.2f}")
plt.axhline(z_score_low, color='green', linestyle='--')
plt.axhline(z_score_mid, color='cyan', linestyle='--')
plt.axhline(z_score_high, color='red', linestyle='--')
plt.text(0, z_score_low, f"LOW: {z_score_low:.2f}") # write the equation of the line on the plot
plt.text(0, z_score_mid, f"MID: {z_score_mid:.2f}") # write the equation of the line on the plot
plt.text(0, z_score_high, f"HIGH: {z_score_high:.2f}") # write the equation of the line on the plot
plt.title("z-score")
plt.legend(loc="lower left")

plt.show()

#------------------------------------------------------------------------------

def Calculate_Volatility(prices, changes, periods):
    # prices: a list of prices generated by Generate_Prices
    # changes: a list of percentage changes generated by Generate_Prices
    # periods: a list of integers representing the number of days for each period
    # returns: a list of volatilities for each period

    volatilities = [] # initialize an empty list to store the volatilities
    start_index = 0 # initialize the starting index of each period
    for period in periods: # loop through each period
        end_index = start_index + period# calculate the ending index of each period
        if end_index > len(prices): # check if the ending index exceeds the length of prices
            break # stop the loop if so
        period_prices = prices[start_index:end_index] # slice the prices for the current period
        period_changes = changes[start_index:end_index] # slice the changes for the current period
        mean_change = sum(period_changes) / len(period_changes) # calculate the mean percentage change for the current period
        #period_changes = 1 + np.array(changes[start_index:end_index]) # slice the changes for the current period
        #mean_change = (np.prod(period_changes) ** (1/len(period_changes))) - 1 # calculate the mean percentage change for the current period
        squared_deviations = [(change - mean_change)**2 for change in period_changes] # calculate the squared deviations from the mean change for each day
        variance = sum(squared_deviations) / len(squared_deviations) # calculate the variance of percentage changes for the current period
        volatility = np.sqrt(variance) * np.sqrt(252) # calculate the annualized volatility for the current period
        volatilities.append(volatility) # append the volatility to the list
        start_index = end_index # update the starting index for the next period
    return volatilities # return the list of volatilities


periods = np.array([5, 10, 20, 50, 100, 200, 400])
periods_r = ["5", "10", "20", "50", "100", "200", "400"]


# calculate volatilities
VOL = Calculate_Volatility(prices, changes, periods)
q1 = np.quantile(VOL, 0.25)
q2 = np.quantile(VOL, 0.5)
q3 = np.quantile(VOL, 0.75)
#plot the original Period and the Volatility
plt.figure(figsize=(10, 6))
plt.bar(periods_r, VOL, color="yellow", edgecolor="black")
plt.axhline(q1, color='green', linestyle='--')
plt.axhline(q2, color='cyan', linestyle='--')
plt.axhline(q3, color='red', linestyle='--')
plt.text(0, q1, f" Quantile 0.25: {q1:.2f}") # write the equation of the line on the plot
plt.text(0, q2, f" Quantile 0.50: {q2:.2f}") # write the equation of the line on the plot
plt.text(0, q3, f" Quantile 0.75: {q3:.2f}") # write the equation of the line on the plot
plt.xlabel("Time")
plt.ylabel("Annualized Volatility")
plt.title("Annualized Volatility of Price change in Different Periods")
#plt.semilogy()
plt.show()

# print the results
for i in range(len(periods)):
    print(f"Period {periods[i]} Volatility {100*VOL[i]:.2f} %")

print('\n')

#------------------------------------------------------------------------------

###########################################################################
##############       CALCULATE BETA AND CORRELATION COEFFICIENT
prices_index, changes_index = Generate_Prices(1000, .06, .00001, 1000)

def ALPHA(prices, prices_index):
    # write alpha coefficient function between (prices, prices_index)
    import numpy as np
    # calculate the returns of prices and prices_index
    returns = [prices[i+1] / prices[i] - 1 for i in range(len(prices) - 1)]
    returns_index = [prices_index[i+1] / prices_index[i] - 1 for i in range(len(prices_index) - 1)]
    # calculate the beta of prices with respect to prices_index
    beta = np.cov(returns_index, returns)[0][1] / np.var(returns_index)
    # calculate the expected return of prices given beta and market return
    expected_return = np.mean(returns_index) * beta
    # calculate the alpha coefficient as the difference between actual and expected return
    return np.mean(returns) - expected_return


def BETA(A, B):
    import numpy as np
    # calculate beta function (slope of A and B)
    return np.cov(A, B)[0][1] / np.var(B)

beta = BETA(prices_index, prices)

def CORRELATION(A, B):
    import numpy as np
    # calculate correlation function (correlation of A and B)
    return np.corrcoef(A, B)[0][1]

# calculate correlation function (correlation of prices_index and prices)
correlation = CORRELATION(prices_index, prices)

# print the alpha coefficient
print("--------------------------------------")
print(f"        Alpha coefficient: {ALPHA(prices, prices_index):.4f}")
print(f"         Beta coefficient: {BETA(prices_index, prices):.4f}")
print(f"  Correlation coefficient: {CORRELATION(prices_index, prices):.4f}")
print("--------------------------------------")

# plot prices and prices_index in 2 y axis separately
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()
ax1.plot(prices_index, color="blue", label="Index")
ax2.plot(prices, color="black", label="Prices")
ax1.set_xlabel("Time")
ax1.set_ylabel("Index")
ax2.set_ylabel("Price")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
ax1.semilogy(); ax2.semilogy();
plt.title(f"Prices and Index Chart - Beta: {beta:.4f} , Correlation: {correlation:.4f}")
plt.show()


def LINEAR_REGRESSION_NUMPY(X, Y, XL='xlabel', YL='ylabel', TI='title', SX=1, SY=1):
    # Import libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    # Create a linear regression model
    X = np.array(X);Y = np.array(Y);
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y) # reshape X to a 2D array

    # Predict the next X income (2023)
    #next_X = np.array([np.min(X)+NUM_SIZES]) # create a 1D array with the next X
    prediction = model.predict(next_X.reshape(-1, 1)) # reshape next_X to a 2D array and make prediction
    # Calculate and display the coefficient and R^2
    coefficient = model.coef_[0] # get the coefficient from the model
    y_predicted = model.predict(X.reshape(-1, 1))
    correlation = np.corrcoef(Y, y_predicted)[0, 1]
    r_squared = (correlation ** 2) # calculate the R^2 score
    print(f"The predicted Y for {next_X[0]} is {prediction[0]:.2f}")

    # Plot the data and the regression line
    plt.figure(figsize=(10,6))
    plt.scatter(X, Y, label=YL) # plot the net income data as scatter points
    plt.plot(X, y_predicted, color="red", label=f"y = {coefficient:.4f}x + {model.intercept_:.4f} -  R^2 = {r_squared:.4f}") # plot the regression line
    plt.plot(next_X, prediction, marker="x", color="green", label=f"Prediction {prediction[0]: .4f}") # plot the prediction as a cross point
    plt.xlabel(XL) # add x-axis label
    plt.ylabel(YL) # add y-axis label
    plt.title(TI) # add title
    if SX == 1:
        plt.semilogx()
    if SY == 1:
        plt.semilogy()
    plt.legend() # add legend

LINEAR_REGRESSION_02(changes_index, changes, XL="Index returns", YL="Price returns", TI="Index and Price Relation")
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------





