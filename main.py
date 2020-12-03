import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import constant_variables
import graph_util

dataset = pd.read_csv('data/MSFT.csv')
dataset['Date'] = pd.to_datetime(dataset.Date,format='%Y-%m-%d')
dataset.index = dataset['Date']

graph_util.plot_graph_single_dataset(dataset["Close"], "Microsoft's Stock Price", "Year", "Close Price USD ($)")

#Filter data from date:
dataset = dataset[(dataset['Date'] > constant_variables.START_DATE) & (dataset['Date'] < constant_variables.END_DATE)]

#Consider Close prices for training
company_close_price = dataset[["Close"]]

company_close_price["Prediction"] = company_close_price[["Close"]].shift(-constant_variables.PREDICTION_DAYS)

x = np.array(company_close_price.drop(["Prediction"], 1))[:-constant_variables.PREDICTION_DAYS]

y = np.array(company_close_price["Prediction"])[:-constant_variables.PREDICTION_DAYS]

#25% of the data is test Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=constant_variables.TEST_DATA_SIZE)

linear = LinearRegression().fit(x_train, y_train)

future_prices = company_close_price.drop(["Prediction"], 1)[:-constant_variables.PREDICTION_DAYS]
future_prices = future_prices.tail(constant_variables.PREDICTION_DAYS)
future_prices = np.array(future_prices)
print(future_prices)

linear_prediction = linear.predict(future_prices)
print("Linear regression Prediction =",linear_prediction)




