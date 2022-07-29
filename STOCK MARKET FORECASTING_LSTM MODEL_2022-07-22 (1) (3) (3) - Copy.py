#!/usr/bin/env python
# coding: utf-8

# # Stock Market Forecasting: MINDTREE.NS

# In[1]:


# Import Labraries
# !pip install pandas_datareader


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import datetime


# In[3]:


stock = pdr.get_data_yahoo('MINDTREE.NS', 
                          start=datetime.datetime(2012, 7, 6), 
                          end=datetime.datetime(2022, 7, 6))
stock


# # Description of each attribute
# #### Describe the attribute of the data set given below. Attribute values are in floating point expect to date and volume.
# 
# #### Date: — Trading date of the stock.
# 
# #### Open: — This price of stock’s opening price which means the very beginning price of particular trading day, but which is not be the same price of precious’s day ending price.
# 
# #### High: — This is the highest price of the stock on a particular trading day.
# 
# #### Low: — This is the lowest stock price during trade day.
# 
# #### Close: — This is the closing price of the stock during trade-in particular day.
# 
# #### Volume: — This is the number of stocks traded on a particular day.
# 
# #### Adj Close: — This is the ending or closing price of the stock which was changed to contain any corporations’ actions and distribution that is occurred during trade time of the day.

# In[4]:


# EDA PART


# In[5]:


stock.head()


# In[6]:


stock.tail()


# In[7]:


stock=stock.reset_index()
stock


# In[8]:


# DROP THE DATE AND ADJ CLOSE COLUMN.


# In[9]:


stock= stock.drop(['Date','Adj Close'],axis=1)
stock


# In[10]:


#Visulation PART


# In[11]:


import plotly.express as px
fig2 = px.line(stock.Close)
fig2.show()


# # Moving average of 100 days

# In[12]:


ma100=stock.Close.rolling(100).mean()
ma100


# In[13]:


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
fig.show()


# In[14]:


# We have plotted 100 days moving average on close column
#we can see the superimpose image of Close column and moving average of 100days


# # Moving average of 200 days 

# In[15]:



ma200=stock.Close.rolling(200).mean()
ma200


# In[16]:


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
fig.show()


# ### Remark:- When green line crosses red line sudden downfall & upfall happens and it has observed in above plot

# In[17]:


stock.shape


# # Spliting 'Close' column into train and test

# In[18]:


train=pd.DataFrame(stock['Close'][0:int(len(stock)*0.70)])
test=pd.DataFrame(stock['Close'][int(len(stock)*0.70):int(len(stock))])
print(train.shape)
print(test.shape)


# ## Scaling the data

# In[19]:


from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler(feature_range=(0,1))


# ## Transform the data

# In[20]:


train_array=scaler.fit_transform(train)
train_array


# In[21]:


train_array.shape


# ## Creating Data with Timesteps
#  
# ### LSTMs expect our data to be in a specific format, usually a 3D array. 
# ### We start by creating data in 100 timesteps and converting it into an array using NumPy. 
# ### Next, we convert the data into a 3D dimension array with X_train samples, 100 timestamps, and one feature at each step.

# In[22]:


#Creating a data structure with 100 time-steps and 1 output

x_train =[]
y_train =[]

for i in range (100,train_array.shape[0]):
    x_train.append(train_array[i-100: i])
    y_train.append(train_array[i, 0])
    
x_train,y_train = np.array(x_train),np.array(y_train)


# In[23]:


x_train.shape


# In[24]:


#ML model 
# !pip install keras
# !pip install tensorflow


# In[25]:


from keras.layers import Dense,Dropout,LSTM
from keras.models import Sequential 


# # Building the LSTM
#  
# ### In order to build the LSTM, we need to import a couple of modules from Keras:
# 
# ### 1. Sequential for initializing the neural network
# ### 2. Dense for adding a densely connected neural network layer
# ### 3. LSTM for adding the Long Short-Term Memory layer
# ### 4. Dropout for adding dropout layers that prevent overfitting
# 
# ### When defining the Dropout layers, we specify 0.2, meaning that 20% of the layers will be dropped. 
# ### Thereafter, we add the Dense layer that specifies the output of 1 unit.
# 
# Now, it’s time to build the model. We will build the LSTM with different sets of neurons and 3 hidden layers. Finally, we will assign 1 neuron in the output layer for predicting the normalized stock price. We will use the MSE loss function and the Adam  optimizer.

# In[26]:


model=Sequential()
model.add(LSTM(units=50,activation='relu',return_sequences = True,
              input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60,activation='relu',return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units=80,activation='relu',return_sequences = True))
model.add(Dropout(0.4))

model.add(LSTM(units=120,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))


# In[27]:


model.summary()


# ### After this, we compile our model using the adam optimizer and set the loss as the mean_squarred_error. 
# ### This will compute the mean of the squared errors.
# ### Next, we will fit the model to run on 50 epochs

# In[28]:


model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,epochs = 50)


# In[29]:


# model.save('keras_model.sdk')


# In[30]:


test.head()


# In[31]:


train.tail(100)


# In[32]:


#in this model we have take previous 100 days values for fisrt value of test data set.


# In[33]:


past_100_days = train.tail(100)


# In[34]:


final_stock = past_100_days.append(test,ignore_index=True)


# In[35]:


final_stock.head()


# In[36]:


#we have scaled this data 


# In[37]:


input_data=scaler.fit_transform(final_stock)
input_data


# In[38]:


input_data.shape


# In[39]:


# we split data x nd y test data


# In[40]:


x_test =[]
y_test =[]

for i in range (100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])
    
x_test,y_test = np.array(x_test),np.array(y_test)


# In[41]:


print(x_test.shape)
print(y_test.shape)


# In[42]:


#making predications

y_predicted = model.predict(x_test)


# In[43]:


y_predicted.shape


# In[44]:


y_test


# In[45]:


y_predicted


# In[46]:


#we have measured scaler scale


# In[47]:


scaler.scale_


# In[48]:


#we inversely transformed that scale factor


# In[49]:


scale_factor = 1/0.00023141
y_predicted = y_predicted * scale_factor
y_test=y_test * scale_factor


# In[50]:


# we have ploted the garph of orginal vs predicated.


# In[51]:


plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label = 'Org Price')
plt.plot(y_predicted,'r',label='Pre Price')
plt.xlabel('Time')
plt.ylabel('price')
plt.legend()
plt.show()


# In[52]:


# we have calculated the rmse for the model.


# In[53]:


#Calulate RMSE of test data 
import math 
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_test,y_predicted))


# In[54]:


# this is the prediction for next day .


# In[55]:


#get the quote
MIND = pdr.get_data_yahoo('MINDTREE.NS', 
                          start=datetime.datetime(2012, 7, 6), 
                          end=datetime.datetime(2022, 7, 6))
#create new dataframe
new_stock = MIND.filter(['Close'])
#get the last 100 days close price value and covert the data frame into array
last_100_days = new_stock[-60:].values
#scale the values between 0 to 1
last_100_days_scaled = scaler.transform(last_100_days)
#create empty list
x_test = []
#append the past 100 days 
x_test.append(last_100_days_scaled)
#convert the x test data into numpy array
x_test = np.array(x_test)
#reshape the data 
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1], 1))
#get the predicted scaled price 
pred_price = model.predict(x_test)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)


# In[56]:


# this is the actual value for next day
# prediction for a signal day 


# In[57]:


MIND = pdr.get_data_yahoo('MINDTREE.NS', 
                          start=datetime.datetime(2022, 7, 7), 
                          end=datetime.datetime(2022, 7, 7))
print(MIND['Close'])


# In[58]:


# Prediction for 10 days


# In[59]:


MIND = pdr.get_data_yahoo('MINDTREE.NS', 
                          start=datetime.datetime(2022, 7, 7), 
                          end=datetime.datetime(2022, 7, 22))
print(MIND['Close'])


# In[ ]:





# In[ ]:





# In[ ]:




