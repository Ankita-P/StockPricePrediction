import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from statsmodels.tsa.arima_model import ARIMA

import constant_variables
import graph_util

dataset = pd.read_csv('data/'+constant_variables.STOCK_SYMBL+'.csv')
dataset['Date'] = pd.to_datetime(dataset.Date,format='%Y-%m-%d')
dataset.index = dataset['Date']

#Filter data from date:
dataset = dataset[(dataset['Date'] > constant_variables.START_DATE) & (dataset['Date'] < constant_variables.END_DATE)]

graph_util.plot_graph_single_dataset(dataset["Close"], "Microsoft's Stock Price", "Year", "Close Price USD ($)")

#Consider Close prices for training
company_close_price = dataset[["Close"]]

company_close_price["Prediction"] = company_close_price[["Close"]]

#company_close_price["Prediction"] = company_close_price[["Close"]].shift(-test_data_size)
#x = np.array(company_close_price.drop(["Prediction"], 1))[:-constant_variables.PREDICTION_DAYS]
#y = np.array(company_close_price["Prediction"])[:-constant_variables.PREDICTION_DAYS]

# x = company_close_price.drop(["Prediction"],1)
# y = company_close_price["Prediction"]

#% of the data is test Data
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=constant_variables.TEST_DATA_SIZE)

test_data_size = int(len(dataset) * constant_variables.TEST_DATA_SIZE)
train_data_size = len(dataset) - test_data_size;

#Split on sequence of tsrain and test data.
x = company_close_price[:train_data_size]
y = company_close_price[train_data_size:]

x_train = x.drop('Close', axis=1)
y_train = x['Close']
x_test = y.drop('Close', axis=1)
y_test = y['Close']

future_prices = company_close_price.drop(["Prediction"], 1)[:-len(x_test)]
future_prices = future_prices.tail(len(x_test))
future_prices = np.array(future_prices)
print(future_prices)

#Linear regression
linear = LinearRegression().fit(x_train, y_train)
linear_prediction = linear.predict(future_prices)
print("Linear regression Prediction =",linear_prediction)
graph_util.plot_graph_original_predicted(dataset[-len(x_test):], "Apple's Stock Price Prediction Model - [Linear Regression Model]",
                                         "Years", "Close Price", company_close_price["Close"], linear_prediction)


#K - nearest neighbour Confidence
knn = KNeighborsRegressor().fit(x_train, y_train)
knn_prediction = knn.predict(future_prices)
print("K-nearest neighbour Prediction =",knn_prediction)
graph_util.plot_graph_original_predicted(dataset[-len(x_test):], "Apple's Stock Price Prediction Model - [Knn Model]",
                                         "Years", "Close Price", company_close_price["Close"], knn_prediction)


#LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
data = dataset.filter(["Close"])
data.reset_index()
scaled_data = scaler.fit_transform(data)

training_data_len = len(x_train)

train_data = scaled_data[0:training_data_len]
x_train=[]
y_train = []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

#Train the model on x and y datasets
model.fit(x_train, y_train, batch_size=1, epochs=1)

#Test data set
test_data = scaled_data[training_data_len - 60:]

x_test = []
y_test =  dataset[training_data_len:]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

# Predicting values using LSTM
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)#Undo scaling

graph_util.plot_graph_original_predicted(dataset[-len(x_test):], "Apple's Stock Price Prediction Model - [LSTM]",
                                         "Years", "Close Price", company_close_price["Close"], predictions)


#ARIMA,

train_data = company_close_price['Close'][:training_data_len].values
testing_data = company_close_price['Close'][training_data_len:].values

history_observations = [x for x in train_data]
model_predictions = []
N_test_observations = len(testing_data)

# ARIMA model parameters set as p=4, d=1, q=0
count = 0
for time_point in range(N_test_observations):
    model = ARIMA(history_observations, order=( 5, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    model_predictions.append(yhat)
    true_test_value = testing_data[time_point]
    history_observations.append(true_test_value)
    count += 1
    print("........",count)

graph_util.plot_graph_original_predicted(dataset[-len(testing_data):], "Apple's Stock Price Prediction Model - [ARIMA]",
                                         "Years", "Close Price", company_close_price["Close"], model_predictions)