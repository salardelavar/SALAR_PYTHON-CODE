"""
FINANCIAL ALPHA AND BETA MOVING AVERAGE
This program is written by Salar Delavar Ghashaghaei (Qashqai)
نویسنده: سالار دلاورقشقایی
"""
### ALPHA AND BETA MOVING AVERAGE 
def TSETMC_HISTORY_PRICE(NAME, START_DATE, END_DATE, SAVE_DATA_PRICE, SAVE_DATA_PRICE_SHEET):
    # : دریافت سابقه قیمت
    import finpy_tse as fpy
    DATA = fpy.Get_Price_History(
        stock=NAME,
        start_date=START_DATE,
        end_date=END_DATE,
        ignore_date=False,
        adjust_price=True,
        show_weekday=False,
        double_date=False,)
    return DATA
# -------------------------------------------
def TSETMC_HISTORY_HAMVAZN(START_DATE, END_DATE, SAVE_DATA_PRICE, SAVE_DATA_PRICE_SHEET):
    # : دریافت سابقه شاخص هم وزن
    import finpy_tse as fpy
    DATA = fpy.Get_EWI_History(
        start_date=START_DATE,
        end_date=END_DATE,
        ignore_date=False,
        just_adj_close=True,
        show_weekday=False,
        double_date=False)
    return DATA
# -------------------------------------------
def TSETMC_HISTORY_KOL(START_DATE, END_DATE, SAVE_DATA_PRICE, SAVE_DATA_PRICE_SHEET):
    # : دریافت سابقه شاخص کل
    import finpy_tse as fpy
    DATA = fpy.Get_CWI_History(
        start_date=START_DATE,
        end_date=END_DATE,
        ignore_date=False,
        just_adj_close=False,
        show_weekday=False,
        double_date=False)
    return DATA
# -------------------------------------------
stock = 'نوری'
index = 'طلا'
START_DATE='1398-07-01'
END_DATE='1403-12-08'
SAVE_DATA_PRICE = "C://TSE_SALAR/TSE_PRICE.xlsx"
SAVE_DATA_PRICE_SHEET = "قیمت"

STOCK = TSETMC_HISTORY_PRICE(stock, START_DATE, END_DATE, SAVE_DATA_PRICE, SAVE_DATA_PRICE_SHEET)
INDEX01 = TSETMC_HISTORY_PRICE(index, START_DATE, END_DATE, SAVE_DATA_PRICE, SAVE_DATA_PRICE_SHEET)
INDEX02 = TSETMC_HISTORY_HAMVAZN(START_DATE, END_DATE, SAVE_DATA_PRICE, SAVE_DATA_PRICE_SHEET) #هم وزن
INDEX = TSETMC_HISTORY_KOL(START_DATE, END_DATE, SAVE_DATA_PRICE, SAVE_DATA_PRICE_SHEET) #کل

### BETA MOVING AVERAGE 
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Define the y data as the STOCK price change
y_data = STOCK["Adj Close"].pct_change()

# Define the x data as the INDEX change
x_data = INDEX["Adj Close"].pct_change()

merged_df = pd.merge(x_data, y_data, on='J-Date')
#print(merged_df)

x_data = merged_df['Adj Close_x']
y_data = merged_df['Adj Close_y']

PERIOD = 50
fig, ax = plt.subplots(figsize=(10, 6))
beta_symbol = x_data.rolling(PERIOD).cov(y_data) / y_data.rolling(PERIOD).var()

# Plot the beta moving average for each symbol
ax.axhline(beta_symbol.quantile(0.05), color="green", linestyle="--", label=f"Q 5: {beta_symbol.quantile(0.05):.4f}")
ax.axhline(beta_symbol.quantile(0.5), color="orange", linestyle="--", label=f"Q 50: {beta_symbol.quantile(0.5):.4f}")
ax.axhline(beta_symbol.quantile(0.95), color="red", linestyle="--", label=f"Q 95: {beta_symbol.quantile(0.95):.4f}")
ax.plot(beta_symbol, label=f"{stock}: {beta_symbol[-1]:.3e}", color='black')
ax.set_title(f"Beta Moving Average for {stock}")
ax.set_xlabel("Date")
ax.set_ylabel("Beta")
#ax.grid()
ax.legend()
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
beta_symbol = x_data.rolling(PERIOD).cov(y_data) / y_data.rolling(PERIOD).var()
alpha_symbol = y_data.rolling(PERIOD).mean() - beta_symbol * x_data.rolling(PERIOD).mean()
Y = alpha_symbol + alpha_symbol * y_data
# Plot the beta moving average for each symbol
ax.axhline(Y.quantile(0.05), color="green", linestyle="--", label=f"Q 5: {Y.quantile(0.05):.4f}")
ax.axhline(Y.quantile(0.5), color="orange", linestyle="--", label=f"Q 50: {Y.quantile(0.5):.4f}")
ax.axhline(Y.quantile(0.95), color="red", linestyle="--", label=f"Q 95: {Y.quantile(0.95):.4f}")
ax.plot(Y, label=f"{stock}:  {alpha_symbol[-1]:.4e} + {beta_symbol[-1]:.4e} * X = {Y[-1]:.3e}", color='black')  
ax.set_title(f"(alpha + beta * X) Moving Average for {stock}")
ax.set_xlabel("Date")
ax.set_ylabel("alpha + beta * X")
#ax.grid()
ax.legend()
plt.show()
print(f'{stock}\t\t alpha + beta * X : {alpha_symbol[-1]:.4e} + {beta_symbol[-1]:.4e} * X')

# ----------------------------------------------------------------------------------
"""
## PREDICTED PRICE WITH LSTM ALOGRITHM

import numpy as np
import matplotlib.pyplot as plt

def PREDICT_LSTM(x, y, look_back, ITERATION):
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    # Prepare data for LSTM
    trainX, trainY = [], []
    for i in range(len(x) - look_back):
        trainX.append(x[i:i + look_back])
        trainY.append(y[i + look_back])

    trainX, trainY = np.array(trainX), np.array(trainY)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(4, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=ITERATION, batch_size=1, verbose=2)

    # Predict the next 'y' value
    next_x = np.array(x[-look_back:]).reshape(1, look_back, 1)
    predicted_y = model.predict(next_x)
    return predicted_y


x = np.arange(1 , len(STOCK["Adj Close"]) + 1)
y = STOCK["Adj Close"]
predicted_y = PREDICT_LSTM(x, y, look_back=50, ITERATION = 200)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x, y, color='black', label='Original data')
plt.scatter(len(x), predicted_y, color='red', marker='o', label='Predicted next y')
plt.xlabel('Data points')
plt.ylabel('y values')
plt.legend()
plt.grid()
plt.show()
"""
