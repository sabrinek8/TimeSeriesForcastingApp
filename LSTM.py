import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn import metrics
from sklearn.metrics import mean_squared_error


#the start and end date
start_date = dt.datetime(2018,4,1)
end_date = dt.datetime(2023,4,1)

#loading from yahoo finance
data = yf.download("TSLA",start_date, end_date)

pd.set_option('display.max_rows', 4)
pd.set_option('display.max_columns',5)
#print(data)

# Setting 80 percent data for training
training_data_len = math.ceil(len(data) * .8)

#Splitting the dataset
train_data = data[:training_data_len].iloc[:,:1] 
test_data = data[training_data_len:].iloc[:,:1]
#print(train_data.shape, test_data.shape)

# Selecting Open Price values
dataset_train = train_data.Open.values 
# Reshaping 1D to 2D array
dataset_train = np.reshape(dataset_train, (-1,1)) 
#print(dataset_train.shape)


scaler = MinMaxScaler(feature_range=(0,1))
# scaling dataset
scaled_train = scaler.fit_transform(dataset_train)
#print(scaled_train[:7])
# Selecting Open Price values
dataset_test = test_data.Open.values 
# Reshaping 1D to 2D array
dataset_test = np.reshape(dataset_test, (-1,1))  
# Normalizing values between 0 and 1
scaled_test = scaler.fit_transform(dataset_test)  
#print(*scaled_test[:5])

X_train = []
y_train = []
for i in range(50, len(scaled_train)):
    X_train.append(scaled_train[i-50:i, 0])
    y_train.append(scaled_train[i, 0])


X_test = []
y_test = []
for i in range(50, len(scaled_test)):
    X_test.append(scaled_test[i-50:i, 0])
    y_test.append(scaled_test[i, 0])
    # The data is converted to Numpy array
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
y_train = np.reshape(y_train, (y_train.shape[0],1))
#print("X_train :",X_train.shape,"y_train :",y_train.shape)


# The data is converted to numpy array
X_test, y_test = np.array(X_test), np.array(y_test)

#Reshaping
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
y_test = np.reshape(y_test, (y_test.shape[0],1))
print("X_test :",X_test.shape,"y_test :",y_test.shape)

#Initialising the model
regressorLSTM = Sequential()

#Adding LSTM layers
regressorLSTM.add(LSTM(50, 
					return_sequences = True, 
					input_shape = (X_train.shape[1],1)))
regressorLSTM.add(LSTM(50, 
					return_sequences = False))
regressorLSTM.add(Dense(25))

#Adding the output layer
regressorLSTM.add(Dense(1))

#Compiling the model
regressorLSTM.compile(optimizer = 'adam',
					loss = 'mean_squared_error',
					metrics = ["accuracy"])

#Fitting the model
regressorLSTM.fit(X_train, 
				y_train, 
				batch_size = 1, 
				epochs =12)

regressorLSTM.summary()


# predictions with X_test data

y_LSTM = regressorLSTM.predict(X_test)

# scaling back from 0-1 to original
y_LSTM_O = scaler.inverse_transform(y_LSTM) 

plt.figure(figsize =(18,12))
#Plot for LSTM predictions
plt.plot(train_data.index[150:], train_data.Open[150:], label = "train_data", color = "b")
plt.plot(test_data.index, test_data.Open, label = "test_data", color = "g")
plt.plot(test_data.index[50:], y_LSTM_O, label = "y_LSTM", color = "orange")
plt.legend()
plt.xlabel("Days")
plt.ylabel("Open price")
plt.show()
