import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

dataset = pd.read_csv('data/MSFT.csv')
print(dataset.head())

print("trainging days =",dataset.shape)
dataset['Date'] = pd.to_datetime(dataset.Date,format='%Y-%m-%d')
dataset.index = dataset['Date']

plt.figure(figsize=(10, 4))
plt.title("Microsoft's Stock Price")
plt.xlabel("Year")
plt.ylabel("Close Price USD ($)")
plt.plot(dataset["Close"])
plt.show()

