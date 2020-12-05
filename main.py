import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

import constant_variables
import graph_util

dataset = pd.read_csv('data/'+constant_variables.STOCK_SYMBL+'.csv')
dataset['Date'] = pd.to_datetime(dataset.Date,format='%Y-%m-%d')
dataset.index = dataset['Date']

graph_util.plot_graph_single_dataset(dataset["Close"], "Microsoft's Stock Price", "Year", "Close Price USD ($)")

#Filter data from date:
dataset = dataset[(dataset['Date'] > constant_variables.START_DATE) & (dataset['Date'] < constant_variables.END_DATE)]

#Consider Close prices for training
company_close_price = dataset[["Close"]]

test_data_size = int(len(dataset) * constant_variables.TEST_DATA_SIZE)
train_data_size = len(dataset) - test_data_size;

company_close_price["Prediction"] = company_close_price[["Close"]]

#company_close_price["Prediction"] = company_close_price[["Close"]].shift(-test_data_size)
#x = np.array(company_close_price.drop(["Prediction"], 1))[:-constant_variables.PREDICTION_DAYS]
#y = np.array(company_close_price["Prediction"])[:-constant_variables.PREDICTION_DAYS]

x = company_close_price.drop(["Prediction"],1)
y = company_close_price["Prediction"]

#25% of the data is test Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=constant_variables.TEST_DATA_SIZE)

#x = company_close_price[:train_data_size]
#y = company_close_price[train_data_size:]

#x_train = x.drop('Close', axis=1)
#y_train = x['Close']
#x_test = y.drop('Close', axis=1)
#y_test = y['Close']

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

