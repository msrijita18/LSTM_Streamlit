import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader as dr
import datetime as dt
from keras.models import load_model
import streamlit as st

ticker = 'AAPL'
start = dt.datetime(2010, 1, 1)
end = dt.datetime(2019, 12, 31)

st.title("Stock Trend Prediction")

user_input = st.text_input('Enter stock ticker', 'AAPL')
df = dr.DataReader(user_input, 'yahoo', start, end)

st.subheader('Data from 2010 - 2019')
st.write(df.describe())

st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 200MA')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma200)
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


train = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
test = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

train_array = scaler.fit_transform(train) 


# x_train = []
# y_train = []

# for i in range(100, train_array.shape[0]):
#     x_train.append(train_array[i-100:i])
#     y_train.append(train_array[i, 0])


# x_train, y_train = np.array(x_train), np.array(y_train)

model = load_model('keras_model.h5')

past_100_days = train.tail(100)
final_df = past_100_days.append(test, ignore_index=True)
input_data = scaler.fit_transform(final_df) 


x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])


x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
