import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Function to generate prices
def Generate_Prices(start_price, Volatility, Direction_Bias, num_days):
    prices = [start_price]
    changes = [0]
    for i in range(num_days-1):
        Var = random.uniform(-1, 1) * random.uniform(0, 1) * Volatility
        price = prices[-1] * (1 + Var + Direction_Bias)
        prices.append(price)
        changes.append(price / prices[-2] - 1)
    return prices, changes

# Function to generate prices using Monte Carlo
def Generate_Prices_Monte_Carlo(start_price, Volatility, Direction_Bias, num_days, num_simulations):
    all_simulations = []
    for _ in range(num_simulations):
        prices = [start_price]
        for i in range(num_days - 1):
            Var = random.uniform(-1, 1) * random.uniform(0, 1) * Volatility
            price = prices[-1] * (1 + Var + Direction_Bias)
            prices.append(price)
        all_simulations.append(prices)
    return np.array(all_simulations)

# Generate historical data
start_price = 100
Volatility = 0.05
Direction_Bias = 0.0001
num_days = 1000
prices, changes = Generate_Prices(start_price, Volatility, Direction_Bias, num_days)

# Generate multiple price simulations
start_price_MC = prices[-1]
Volatility_MC = 0.05
Direction_Bias_MC = 0.0001
num_days_MC = 100
num_simulations = 100
simulated_prices = Generate_Prices_Monte_Carlo(start_price_MC, Volatility_MC, Direction_Bias_MC, num_days_MC, num_simulations)

# Calculate average predicted prices
avg_predicted_prices = np.mean(simulated_prices, axis=0)

# Prepare data for ANN
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(np.array(prices).reshape(-1, 1))

# Create training data
train_data = scaled_prices[:int(len(scaled_prices) * 0.8)]
x_train = []
y_train = []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the ANN model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Prepare test data
test_data = scaled_prices[int(len(scaled_prices) * 0.8) - 60:]
x_test = []
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predict prices using ANN
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot results
plt.figure(figsize=(14, 7))
for i in range(num_simulations):
    plt.plot(range(len(prices), len(prices) + len(simulated_prices[i])), simulated_prices[i], color='grey', alpha=0.5)
plt.plot(range(len(prices), len(prices) + len(avg_predicted_prices)), avg_predicted_prices, label='Average Predicted Prices', color='red', linewidth=2)
plt.plot(range(len(prices)), prices, label='Historical Prices', color='black')
plt.plot(range(int(len(prices) * 0.8), int(len(prices) * 0.8) + len(predicted_prices)), predicted_prices, label='ANN Predicted Prices', color='blue', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Price Prediction Using Monte Carlo Method and ANN')
plt.legend()
plt.semilogy()
plt.show()