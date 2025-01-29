# Import libraries
import random
import numpy as np
import matplotlib.pyplot as plt

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

# Plot the actual and predicted prices
plt.figure(figsize=(12, 8))
plt.plot(prices, label=f'Last Price:  {prices[-1]:.2f}', color='black')
plt.xlabel('Day')
plt.ylabel('Price')
plt.title(f'Price Charts')
plt.legend()
plt.semilogy()
plt.grid()
plt.show()


# Define parameters
#S = 100 # Stock price
S = prices[-1] # Stock price
K = 105 # Strike price
C = 5 # Call option premium
P = 7 # Put option premium

# Define range of stock prices at expiration
ST = np.arange(np.min(prices), np.max(prices), 1)

# Define payoff functions for each strategy
def long_call_payoff(ST, K, C):
    return np.maximum(ST - K, 0) - C

def long_put_payoff(ST, K, P):
    return np.maximum(K - ST, 0) - P

def short_call_payoff(ST, K, C):
    return -long_call_payoff(ST, K, C)

def short_put_payoff(ST, K, P):
    return -long_put_payoff(ST, K, P)

def covered_call_payoff(ST, S, K, C):
    return ST - S + long_call_payoff(ST, K, C)

def protective_put_payoff(ST, S, K, P):
    return ST - S + long_put_payoff(ST, K, P)

# Plot payoff diagrams for each strategy
plt.figure(figsize=(10, 8))
plt.subplot(3, 2, 1)
plt.plot(ST, long_call_payoff(ST, K, C))
plt.title('Long Call')
plt.xlabel('Stock Price')
plt.ylabel('Payoff')
plt.axhline(0, color='black', ls='--')
plt.axvline(K, color='red', ls='--')

plt.subplot(3, 2, 2)
plt.plot(ST, long_put_payoff(ST, K, P))
plt.title('Long Put')
plt.xlabel('Stock Price')
plt.ylabel('Payoff')
plt.axhline(0, color='black', ls='--')
plt.axvline(K, color='red', ls='--')

plt.subplot(3, 2, 3)
plt.plot(ST, short_call_payoff(ST, K, C))
plt.title('Short Call')
plt.xlabel('Stock Price')
plt.ylabel('Payoff')
plt.axhline(0, color='black', ls='--')
plt.axvline(K, color='red', ls='--')

plt.subplot(3, 2, 4)
plt.plot(ST, short_put_payoff(ST, K, P))
plt.title('Short Put')
plt.xlabel('Stock Price')
plt.ylabel('Payoff')
plt.axhline(0 ,color='black', ls='--')
plt.axvline(K ,color='red', ls='--')

plt.subplot(3, 2, 5)
plt.plot(ST, covered_call_payoff(ST, S, K, C))
plt.title('Covered Call')
plt.xlabel('Stock Price')
plt.ylabel('Payoff')
plt.axhline(0, color='black', ls='--')
plt.axvline(K, color='red', ls='--')

plt.subplot(3, 2, 6)
plt.plot(ST, protective_put_payoff(ST, S, K ,P))
plt.title('Protective Put')
plt.xlabel('Stock Price')
plt.ylabel('Payoff')
plt.axhline(0 ,color='black', ls='--')
plt.axvline(K ,color='red', ls='--')

plt.tight_layout()
plt.show()

"""
write Married Put function,
Bull Call Spread function,
Bear Put Spread function,
Protective Collar function,
Long Straddle function,
Long Call Butterfly Spread function,
Iron Condor function,
Iron Butterfly function
and plot them
"""

#######################################################


# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
#s0 = 100 # initial stock price
k1 = 90 # lower strike price of long put
k2 = 95 # lower strike price of short put
k3 = 105 # higher strike price of short call
k4 = 110 # higher strike price of long call
premium1 = 2 # premium of long put
premium2 = 4 # premium of short put
premium3 = 4 # premium of short call
premium4 = 2 # premium of long call

# Define stock price range at expiration
sT = np.arange(np.min(prices), np.max(prices), 1)

# Calculate payoffs of each option
payoff_long_put = np.maximum(k1 - sT, 0) - premium1
payoff_short_put = -np.maximum(k2 - sT, 0) + premium2
payoff_short_call = -np.maximum(sT - k3, 0) + premium3
payoff_long_call = np.maximum(sT - k4, 0) - premium4

# Calculate payoff of Iron Condor
payoff_iron_condor = payoff_long_put + payoff_short_put + payoff_short_call + payoff_long_call

# Plot payoff diagram
plt.figure(figsize=(10,6))
plt.plot(sT, payoff_iron_condor, label='Iron Condor')
plt.plot(sT, payoff_long_put, '--', label='Long Put')
plt.plot(sT, payoff_short_put, '--', label='Short Put')
plt.plot(sT, payoff_short_call, '--', label='Short Call')
plt.plot(sT, payoff_long_call, '--', label='Long Call')
plt.axhline(y=0, color='k')
plt.xlabel('Stock Price at Expiration')
plt.ylabel('Payoff')
plt.legend()
plt.show()

#######################################################

# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
S = np.arange(np.min(prices), np.max(prices), 1) # Stock price range
K1 = 100 # Strike price of lower call
K2 = 120 # Strike price of higher call
C1 = 10 # Premium of lower call
C2 = 5 # Premium of higher call

# Calculate payoff and profit/loss
payoff = np.maximum(S - K1, 0) - np.maximum(S - K2, 0) # Payoff of bull call spread
profit = payoff - (C1 - C2) # Profit/loss of bull call spread

# Plot the graph
plt.figure(figsize=(10, 6))
plt.plot(S, payoff, '--', label='Payoff')
plt.plot(S, profit, label='Profit/Loss')
plt.axhline(0, color='black', lw=0.5)
plt.axvline(K1 + C1 - C2 , color='red', ls='--', label='Breakeven Point')
plt.xlabel('Stock Price')
plt.ylabel('Payoff / Profit/Loss')
plt.title('Bull Call Spread')
plt.legend()
plt.show()


#######################################################

# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
S = np.arange(np.min(prices), np.max(prices), 1) # Stock price range
K1 = 100 # Strike price of lower call
K2 = 120 # Strike price of higher call
K3 = 80 # Strike price of put
C1 = 10 # Premium of lower call
C2 = 5 # Premium of higher call
P = 8 # Premium of put

# Calculate payoff and profit/loss for Butterfly Spread
payoff_BS = np.maximum(S - K1, 0) - np.maximum(S - K2, 0) # Payoff of bull call spread
profit_BS = payoff_BS - (C1 - C2) # Profit/loss of bull call spread

# Calculate payoff and profit/loss for Married Put
payoff_MP = np.maximum(K3 - S, 0) + S # Payoff of married put
profit_MP = payoff_MP - (S + P) # Profit/loss of married put

# Plot the graphs
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(S, payoff_BS, '--', label='Payoff')
plt.plot(S, profit_BS, label='Profit/Loss')
plt.axhline(0, color='black', lw=0.5)
plt.axvline(K1 + C1 - C2 , color='red', ls='--', label='Breakeven Point')
plt.xlabel('Stock Price')
plt.ylabel('Payoff / Profit/Loss')
plt.title('Butterfly Spread')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(S, payoff_MP, '--', label='Payoff')
plt.plot(S, profit_MP, label='Profit/Loss')
plt.axhline(0, color='black', lw=0.5)
plt.axvline(S[0] + P , color='red', ls='--', label='Breakeven Point')
plt.xlabel('Stock Price')
plt.ylabel('Payoff / Profit/Loss')
plt.title('Married Put')
plt.legend()

plt.show()


#######################################################

# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
S = np.arange(np.min(prices), np.max(prices), 1) # Stock price range
K1 = 100 # Strike price of lower call
K2 = 120 # Strike price of higher call
K3 = 80 # Strike price of put
C1 = 10 # Premium of lower call
C2 = 5 # Premium of higher call
C3 = 8 # Premium of put
shares = 100 # Shares per lot

# Calculate payoff and profit/loss for Long Strangle
payoff_LS = np.maximum(S - K2, 0) + np.maximum(K3 - S, 0) # Payoff of long strangle
profit_LS = payoff_LS - (C2 + C3) * shares # Profit/loss of long strangle

# Calculate payoff and profit/loss for Long Call Butterfly Spread
payoff_LC = np.maximum(S - K1, 0) - np.maximum(S - K2, 0) # Payoff of long call spread
profit_LC = payoff_LC - (C1 - C2) * shares # Profit/loss of long call spread

# Plot the graphs
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(S, payoff_LS, '--', label='Payoff')
plt.plot(S, profit_LS, label='Profit/Loss')
plt.axhline(0, color='black', lw=0.5)
plt.xlabel('Stock Price')
plt.ylabel('Payoff / Profit/Loss')
plt.title('Long Strangle')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(S, payoff_LC, '--', label='Payoff')
plt.plot(S, profit_LC, label='Profit/Loss')
plt.axhline(0, color='black', lw=0.5)
plt.axvline(K1 + C1 - C2 , color='red', ls='--', label='Breakeven Point')
plt.xlabel('Stock Price')
plt.ylabel('Payoff / Profit/Loss')
plt.title('Long Call Butterfly Spread')
plt.legend()

plt.show()

