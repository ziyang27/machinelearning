import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

model = load_model('stock_price_predictor.h5')

st.header('Stock Price Predictor')

stock = st.text_input('Enter Stock Ticker', 'GOOG')
start = '2012-01-01'
end = '2024-10-01'

data = yf.download(stock, start=start, end=end)

st.subheader(f'Stock Data for {stock}')
st.write(data)

data_train = pd.DataFrame(data['Close'][:int(0.8*len(data))])
data_test = pd.DataFrame(data['Close'][int(0.8*len(data)):])

scaler = MinMaxScaler(feature_range=(0, 1))

past_100_days = data_train.tail(100)
data_test = pd.concat((past_100_days, data_test), ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test)

st.subheader('Price vs Moving Average 50')
ma_50 = data['Close'].rolling(window=50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(data['Close'], label='Close Price', color='red')
plt.plot(ma_50, label='50 Day Moving Average', color='blue')
plt.title(f'{stock} Stock Price vs Moving Average 50')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig1)

st.subheader('Price vs Moving Average 50 vs Moving Average 100')
ma_100 = data['Close'].rolling(window=100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(data['Close'], label='Close Price', color='red')
plt.plot(ma_50, label='50 Day Moving Average', color='blue')
plt.plot(ma_100, label='100 Day Moving Average', color='green')
plt.title(f'{stock} Stock Price vs Moving Average 50 vs Moving Average 100')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

st.subheader('Price vs Moving Average 100 vs Moving Average 200')
ma_200 = data['Close'].rolling(window=100).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(data['Close'], label='Close Price', color='red')
plt.plot(ma_200, label='200 Day Moving Average', color='blue')
plt.plot(ma_100, label='100 Day Moving Average', color='green')
plt.title(f'{stock} Stock Price vs Moving Average 100 vs Moving Average 200')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)

x = []
y = []

for i in range(100, data_test_scaled.shape[0]):
    x.append(data_test_scaled[i-100:i])
    y.append(data_test_scaled[i, 0])    

x, y = np.array(x), np.array(y) 

predicted_stock_price = model.predict(x)
predicted_stock_price = predicted_stock_price / scaler.scale_
y = y / scaler.scale_

st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8, 6))
plt.plot(predicted_stock_price, label='Predicted Price', color='red')
plt.plot(y, label='Original Price', color='blue')
plt.title(f'{stock} Original Price vs Predicted Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)

