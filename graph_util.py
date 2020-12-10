import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

import constant_variables

register_matplotlib_converters()


def plot_graph_single_dataset(dataset, title, xlabel, ylabel):
    # Plot the graph of original Prices
    #plt.ion()
    plt.figure(figsize=(10, 4))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(dataset)
    plt.savefig("output/graph/"+title+" - "+constant_variables.START_DATE+" -- "+constant_variables.END_DATE+".png")

def plot_graph_original_predicted(dataset, title, xlabel, ylabel, company_close_price, predictions):
    # Plot the graph of original Prices
    valid = dataset
    valid["Predictions"] = predictions
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(company_close_price)
    plt.plot(valid[["Close", "Predictions"]])
    plt.legend(["Original_train", "Valid_test", "Predictions"])
    plt.savefig("output/graph/" + title + " - " + constant_variables.START_DATE + " -- " + constant_variables.END_DATE + ".png")