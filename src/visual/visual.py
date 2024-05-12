import logging as log

import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.api as sm

from collections import defaultdict
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.linear_model import LinearRegression

from src.config.config import *


class Visual:
    def __init__(self, config):
        self.config = config

    @staticmethod
    def dist(variables: dict, output: str):
        """
        Draw a distribution plot for the provided variable.
        """

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

        draw_dist(data, output)

    @staticmethod
    def plot(variables: dict, correlation: str, output: str):
        """
        Draw a scatter plot for the provided variables.
        """
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

        draw_correlation_plot(correlation, x_title, y_title, x_values, y_values, output)

    @staticmethod
    def heatmap(variables: dict, correlation: str, output: str):
        """
        Draw a heatmap for the provided variables.
        """

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

    @staticmethod
    def regression(method: str, dependent: list, independent: list):
        """
        Execute Regression using specified method.
        Args:
            method (str): The regression method to use.
            dependent (list): List of dependent variable names.
            independent (list): List of independent variable names.
        """
    
        database_api_host = get("DATABASE_API_HOST")
        url = f"{database_api_host}/api/v1/repos/normalized"
    
        response = requests.get(url)
        response.raise_for_status()
    
        repos = response.json()
    
        depen = defaultdict(list)
        indep = defaultdict(list)
    
        for repo in repos:
            for dep in dependent:
                if dep in repo:
                    depen[dep].append(repo[dep])
            for ind in independent:
                if ind in repo:
                    indep[ind].append(repo[ind])

        draw_regression_plot(method, depen, indep)

    @staticmethod
    def cluster(method: str, variables: list, output: str):
        """
        Execute clustering based on the specified method and save the results.
    
        Args:
            method (str): Clustering method, e.g., 'hierarchical'.
            variables (list): List of variables to include in the clustering.
            output (str): Path to save the output plot or data file.
        """
        database_api_host = get("DATABASE_API_HOST")
        s = f"{database_api_host}/api/v1/repos/normalized"
    
        response = requests.get(s)
        response.raise_for_status()
    
        repos = response.json()
        
        # Variables -> Observations
        data = {var: [] for var in variables}
        for repo in repos:
            for var in variables:
                if var in repo:
                    data[var].append(repo[var])
    
        df = pd.DataFrame(data)
        df_transposed = df.transpose()
    
        if method == "hierarchical":
            Z = linkage(df_transposed, method='ward', metric='euclidean')
    
            plt.figure(figsize=(15, 10))  # Increased figure size for better clarity
            dendrogram(
                Z,
                orientation='left',
                labels=variables,
                leaf_font_size=10,
            )
    
            plt.title("Hierarchical Clustering Dendrogram of Variables")
            plt.xlabel("Variables")
            plt.ylabel("Distance")
            plt.tight_layout()  # Adjust layout to fit everything nicely
            plt.savefig(output)
            plt.close()


def draw_regression_plot(method: str, depen, indep):
    """
    Draw a regression plot based on the specified method and data.
    Args:
        method (str): 'linear' or 'quantile' to specify the regression method.
        depen (dict): A dictionary where the values are lists of dependent variable data.
        indep (dict): A dictionary where the values are lists of independent variables data.
    """
    x = np.column_stack(list(indep.values()))

    for key, values in depen.items():
        y = np.array(values)

        if method == "linear":
            model = LinearRegression().fit(x, y)
            r_squared = model.score(x, y)
            intercept = model.intercept_
            slopes = model.coef_

            print(f"Linear Regression Summary for {key}")
            print(f"    R^2 = {r_squared}")
            print(f"    Intercept = {intercept}")
            print(f"    Slopes = {slopes}")

        elif method == "quantile":
            quantile = 0.5
            try:
                model = sm.QuantReg(y, sm.add_constant(x)).fit(q=quantile)
                print(f"Quantile Regression Summary for {key}")
                print(model.summary())
                
            except Exception as e:
                print(f"An error occurred during model fitting for {key}:")
                print(str(e))


def draw_dist(data, output):
    """
    Generates and saves a histogram with KDE for each provided data series in a single plot.

    Args:
        data (dict): Dictionary with keys as the variable names and values as lists of values to plot.
        output (str): Path including the filename to save the output plot file.
    """
    sns.set_theme(style="whitegrid", context="notebook")

    plt.figure(figsize=(10, 6))

    # Iterate over each data series to plot
    for name, values in data.items():
        sns.kdeplot(values, bw_adjust=2, label=name, clip=(min(values), max(values)))

    plt.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.75)
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.5)

    plt.xlabel("Value", fontsize=10)
    plt.ylabel("Frequency", fontsize=10)
    plt.title("Distributions of Dataset Variables", fontsize=12)
    plt.xlim(0, 0.1)
    plt.legend()

    plt.savefig(output, bbox_inches='tight')
    plt.close()


def draw_correlation_plot(method: str, x_title: str, y_title: str, x_input: list, y_input: list, output: str):
    """
    Draw a scatter plot for the provided variables.
    """
    x = np.ravel(x_input)
    y = np.ravel(y_input)

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
    data = np.polyfit(x, y, 1)
    polynomial = np.poly1d(data)

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
    """
    Calculate the correlation matrix for the given dataframe.
    """
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
    """
    Draw a heatmap for the provided variables.
    """
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
