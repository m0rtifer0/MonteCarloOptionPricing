# utils/visualization.py

import matplotlib.pyplot as plt

def plot_price_vs_parameter(x, y, parameter_name, title="Option Price vs Parameter"):
    """
    Plots the option price as a function of a given parameter.

    Args:
        x (array-like): Values of the parameter.
        y (array-like): Corresponding option prices.
        parameter_name (str): Label for the x-axis.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o')
    plt.xlabel(parameter_name)
    plt.ylabel("Option Price")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_greek_vs_spot(spot_prices, greek_values, greek_name):
    """
    Plots the value of a Greek (Delta, Gamma, etc.) against spot prices.

    Args:
        spot_prices (array-like): Spot prices.
        greek_values (array-like): Corresponding Greek values.
        greek_name (str): Name of the Greek being plotted.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(spot_prices, greek_values, marker='x', linestyle='--')
    plt.xlabel("Spot Price")
    plt.ylabel(greek_name)
    plt.title(f"{greek_name} vs Spot Price")
    plt.grid(True)
    plt.tight_layout()
    plt.show()