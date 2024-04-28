import logging as log

import numpy
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau
import pandas as pd
import numpy as np

from src.config.config import *


class Visual:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def dist(variables: dict, output: str):
        """
        Generate distribution.
        """
        var: str = variables[0]

        database_api_host = get("DATABASE_API_HOST")
        s = f"{database_api_host}/api/v1/repos/normalized"

        response = requests.get(s)
        response.raise_for_status()

        repos = response.json()

        data = []

        for repo in repos:
            data.append(repo[var])

        draw_dist(data, var, output)

    @staticmethod
    def plot(variables: dict, correlation: str, output: str):
        if len(variables) != 2:
            log.error("Please provide exactly two variables.")
            raise ValueError("Please provide exactly two variables.")
        x_title: str = variables[0]
        x_values: list = []
        y_title: str = variables[1]
        y_values: list = []

        database_api_host = get("DATABASE_API_HOST")
        s = f"{database_api_host}/api/v1/repos/normalized"

        response = requests.get(s)
        response.raise_for_status()

        repos = response.json()

        for repo in repos:
            if x_title in repo:
                x_values.append(repo[x_title])
            if y_title in repo:
                y_values.append(repo[y_title])

        draw_plot(correlation, x_title, y_title, x_values, y_values, output)

    @staticmethod
    def heatmap(variables: dict, correlation: str, output: str):
        print(variables)
        print(correlation)
        print(output)

        database_api_host = get("DATABASE_API_HOST")
        s = f"{database_api_host}/api/v1/repos/normalized"

        response = requests.get(s)
        response.raise_for_status()

        repos = response.json()

        data: dict = {}

        for repo in repos:
            if all(var in repo for var in variables):
                for key in variables:
                    if key in data:
                        data[key].append(repo[key])
                    else:
                        data[key] = [repo[key]]

        draw_heatmap(data, correlation, output)


def draw_dist(data, name, output):
    """
    Generates and saves a histogram with KDE for the provided data.

    Args:
    data (pandas.Series): Data series to plot.
    name (str): Name for the plot and part of the output file name.
    output_path (str): Full path including the filename to save the output plot file.

    """
    sns.set(style="whitegrid", context="notebook")

    plt.figure()

    bw_adj = 1
    linewidth = 1

    # Plotting the distribution
    sns.kdeplot(data, bw_adjust=bw_adj, linewidth=linewidth)
    sns.histplot(data)

    plt.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.75)

    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.5)

    plt.xlabel("Value", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.title(f"Distribution of {name}", fontsize=12)
    plt.xlim(0, 1.0)

    plt.savefig(output)
    plt.close()


def draw_plot(method: str, x_title: str, y_title: str, x_input: list, y_input: list, output: str):
    x = numpy.ravel(x_input)
    y = numpy.ravel(y_input)

    coefficient = None
    p = None

    if method == "pearson":
        coefficient, p = pearsonr(x, y)
        log.debug(f"Pearson's Correlation Coefficient: {coefficient} and Probability Value: {p}")

    if method == "spearman":
        coefficient, p = spearmanr(x, y)
        log.debug(f"Spearman's Correlation Coefficient: {coefficient} and Probability Value: {p}")

    if method == "kendall":
        coefficient, p = kendalltau(x, y)
        log.debug(f"Kendall's Correlation Coefficient: {coefficient} and Probability Value: {p}")

    # Calculate coefficients, and function for the line of best fit.
    data = numpy.polyfit(x, y, 1)
    polynomial = numpy.poly1d(data)

    # Plot the Values
    plt.scatter(x, y, color='black', s=0.5, alpha=1, marker="o", linewidth=0, label="Data Points", zorder=1, edgecolors=None, facecolors=None, antialiased=True, rasterized=None, norm=None, vmin=None, vmax=None, data=None)

    plt.plot(x, polynomial(x), color='black', linewidth=0.1, label="Linear Regression", zorder=1, antialiased=True, rasterized=True, data=None)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.autoscale(enable=True, axis='both', tight=None)

    legend_coefficient = plt.legend(["r = " + str(coefficient)], loc='upper right', bbox_to_anchor=(1.0, 1.0))

    if method == "pearson":
        legend_coefficient.set_title("Pearson's Correlation Coefficient")

    if method == "spearman":
        legend_coefficient.set_title("Spearman's Correlation Coefficient")

    if method == "kendall":
        legend_coefficient.set_title("Kendall's Correlation Coefficient")

    legend_coefficient.get_title().set_fontsize('small')
    legend_coefficient.get_title().set_fontweight('bold')

    legend_p = plt.legend(["p = " + str(p)], loc='upper right', bbox_to_anchor=(1.0, 0.85))
    legend_p.set_title("Probability Value")
    legend_p.get_title().set_fontsize('small')
    legend_p.get_title().set_fontweight('bold')

    plt.gca().add_artist(legend_coefficient)
    plt.gca().add_artist(legend_p)

    plt.grid(True, which='major', axis='both', linestyle='-', linewidth=0.05, color='grey', alpha=0.75)

    plt.axis([min(x), max(x), min(y), max(y)])

    plt.savefig(output, dpi=300, bbox_inches='tight', pad_inches=0.1)


def correlation_matrix(df, method: str):
    n = len(df.columns)
    corr_matrix = pd.DataFrame(np.zeros((n, n)), columns=df.columns, index=df.columns)
    for i in range(n):
        for j in range(n):
            if method == 'kendall':
                corr, _ = kendalltau(df.iloc[:, i], df.iloc[:, j])
            elif method == 'pearson':
                corr, _ = pearsonr(df.iloc[:, i], df.iloc[:, j])
            elif method == 'spearman':
                corr, _ = spearmanr(df.iloc[:, i], df.iloc[:, j])
            else:
                raise ValueError("Invalid correlation type. Use 'kendall', 'pearson', or 'spearman'.")
            corr_matrix.iat[i, j] = corr
    return corr_matrix


def draw_heatmap(data: dict, correlation: str, output: str):
    df = pd.DataFrame(data)

    corr_matrix = correlation_matrix(df, correlation)

    plt.figure(figsize=(10, 8))

    fig, ax = plt.subplots(figsize=(21, 17))

    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax,
                annot_kws={"size": 16},
                xticklabels=list(map(str, df.columns)),
                yticklabels=list(map(str, df.columns)))

    ax.set_title("Correlation Matrix: " + correlation, fontsize=20, loc='right', x=1.3, y=1.05)

    plt.savefig(output)