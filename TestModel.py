import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plot

historical_data = pd.read_csv('data/AAPL.csv')

data = historical_data.filter(['Close'])

dataset = data.values
# Calculate size of training data (80%)
training_data_len = math.ceil( len(dataset) *.8)

# Scale the all of the data to be values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data = scaled_data[0:training_data_len, : ]
#Split the data into x_train and y_train data sets
x_train=[]
y_train = []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])


x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


#Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))


model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model on x and y datasets
model.fit(x_train, y_train, batch_size=1, epochs=1)

#Test data set
test_data = scaled_data[training_data_len - 60: , : ]

x_test = []
y_test =  dataset[training_data_len : , : ]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

# Predicting values using LSTM
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)#Undo scaling


# Calculating the value of RMSE
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
rmse

# Calulate the predictions
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

# Plot prices predicted by model
plot.figure(figsize=(16,8))
plot.title('Model')
plot.xlabel('Date', fontsize=18)
plot.ylabel('Close Price USD ($)', fontsize=18)
plot.plot(train['Close'])
plot.plot(valid[['Close', 'Predictions']])
plot.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plot.show()