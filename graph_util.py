import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def plot_graph_single_dataset(dataset, title, xlabel, ylabel):
    # Plot the graph of original Prices
    plt.figure(figsize=(10, 4))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(dataset)
    plt.show()

def plot_graph_original_predicted(dataset, title, xlabel, ylabel):
    # Plot the graph of original Prices
    valid = dataset[x.shape[0]:]
    plt.figure(figsize=(10, 6))
    plt.title("Apple's Stock Price Prediction Model(Linear Regression Model)")
    plt.xlabel("Years")
    plt.ylabel("Close Price USD ($)")
    plt.plot(company_close_price["Close"])
    plt.plot(valid[["Close", "Predictions"]])
    plt.legend(["Original", "Valid", "Predictions"])
    plt.show()