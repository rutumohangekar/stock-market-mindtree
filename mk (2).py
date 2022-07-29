from multiprocessing.sharedctypes import Value
from turtle import title
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import datetime
from keras.models import load_model
import streamlit as st

st.title('Stock Market Predictions')


start = st.date_input('Start Date',value=datetime.datetime(2012, 7, 6))

end = st.date_input('End Date',value=datetime.datetime(2022, 7, 6))

user_input = st.text_input('Enter stock name','MINDTREE.NS')
stock = pdr.get_data_yahoo('MINDTREE.NS', start,end)

#describe data 
st.subheader('data form 2012 to 2022')
st.write(stock.describe())

#visulation
st.subheader('closing price vs time')
fig = plt.figure(figsize=(12,6))
import plotly.express as px
fig = px.line(stock.Close)
st.plotly_chart(fig)

st.subheader('closing price vs time with 100ma')
ma100=stock.Close.rolling(100).mean()
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=stock.index, y=stock['Close'], name="Close"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=ma100.index, y=ma100, name="ma100"),
    secondary_y=True,
)
st.plotly_chart(fig)


st.subheader('closing price vs time with 100ma and 200ma')
ma200=stock.Close.rolling(200).mean()
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=stock.index, y=stock['Close'], name="Close"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=ma100.index, y=ma100, name="ma100"),
    secondary_y=True,
)

fig.add_trace(
    go.Scatter(x=ma200.index, y=ma200, name="ma200"),
    secondary_y=True,
)
st.plotly_chart(fig)


# spliting the data into train test

train=pd.DataFrame(stock['Close'][0:int(len(stock)*0.70)])
test=pd.DataFrame(stock['Close'][int(len(stock)*0.70):int(len(stock))])
print(train.shape)
print(test.shape)

# scaling  the data

from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler(feature_range=(0,1))


train_array=scaler.fit_transform(train)

# spliting the data into x train y train
x_train =[]
y_train =[]

for i in range (100,train_array.shape[0]):
    x_train.append(train_array[i-100: i])
    y_train.append(train_array[i, 0])
    
x_train,y_train = np.array(x_train),np.array(y_train)

#load the model
model= load_model('keras_model.sdk')

#testing part
past_100_days = train.tail(100)
final_stock = past_100_days.append(test,ignore_index=True)
input_data=scaler.fit_transform(final_stock)

# spliting data into x test and y test
x_test =[]
y_test =[]

for i in range (100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])
    
x_test,y_test = np.array(x_test),np.array(y_test)

#making predications
y_predicted = model.predict(x_test)


scalerk= scaler.scale_
scale_factor = 1/scalerk[0]
y_predicted = y_predicted * scale_factor
y_test=y_test * scale_factor


# final graph
st.subheader('prediction vs orginal')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label = 'Org Price')
plt.plot(y_predicted,'r',label='Pre Price')
plt.xlabel('Time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig2)



