# def multiple_linear_regression(independent_vars, dependent_var, column_names):
#     # Convert lists to a DataFrame
#     data = pd.DataFrame(np.column_stack(independent_vars), columns=column_names[:-1])
#     data[column_names[-1]] = dependent_var
#
#     # Define your independent (X) and dependent (y) variables
#     X = data[column_names[:-1]]
#     y = data[column_names[-1]]
#
#     # Split your data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
#     # Create a linear regression model and fit it to the training data
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#
#     # Make predictions using the testing data
#     y_pred = model.predict(X_test)
#
#     # Evaluate the model
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#
#     print("Mean Squared Error: ", mse)
#     print("R-squared: ", r2)
#
#     return model


# def visualize_multiple_distributions(data_frame):
#     # Plot histogram and KDE
#     plt.figure()
#
#     # for f in data_frame:
#     sns.histplot(data_frame['watchers'], kde=True)
#
#     sns.set(style="whitegrid", context="notebook")
#
#     # Add labels and title to the plot
#     plt.xlabel("Value", fontsize=10)
#     plt.ylabel("Frequency", fontsize=10)
#     plt.title(f"Distributions", fontsize=12)
#
#     # Create "out" folder if it doesn't exist
#     if not os.path.exists("out"):
#         os.makedirs("out")
#
#     # Save the plot to the "out" folder
#     file_name = f"{datetime.now().isoformat()}_distributions.png"
#     plt.savefig(f"out/{file_name}")


# def visualize_distribution(column_name, data_list):
#     # Create a DataFrame from the input list
#     df = pd.DataFrame(data_list, columns=[column_name])
#
#     # Plot histogram and KDE
#     plt.figure()
#     sns.histplot(df[column_name], kde=True)
#     sns.set(style="whitegrid", context="notebook")
#
#     # Add labels and title to the plot
#     plt.xlabel("Value", fontsize=10)
#     plt.ylabel("Frequency", fontsize=10)
#     plt.title(f"Distribution of {column_name}", fontsize=12)
#
#     # Create "out" folder if it doesn't exist
#     if not os.path.exists("out"):
#         os.makedirs("out")
#
#     # Save the plot to the "out" folder
#     file_name = f"{datetime.now().isoformat()}_{column_name}_distribution.png"
#     plt.savefig(f"out/{file_name}")


# def normalize_values(data):
#     ids, values = zip(*data)
#
#     values_array = np.array(values).reshape(-1, 1)
#
#     # Apply MinMaxScaler
#     scaler = MinMaxScaler()
#     normalized_values_array = scaler.fit_transform(values_array)
#
#     # Combine the IDs and normalized values back into the desired format
#     normalized_data = [(id, value[0]) for id, value in zip(ids, normalized_values_array)]
#
#     return normalized_data


# def calc_corr_matrix(df, method):
#     n = len(df.columns)
#     corr_matrix = pd.DataFrame(np.zeros((n, n)))
#     for i in range(n):
#         for j in range(n):
#             if method == 'kendall':
#                 corr, _ = kendalltau(df.iloc[:, i], df.iloc[:, j])
#             elif method == 'pearson':
#                 corr, _ = pearsonr(df.iloc[:, i], df.iloc[:, j])
#             elif method == 'spearman':
#                 corr, _ = spearmanr(df.iloc[:, i], df.iloc[:, j])
#             else:
#                 raise ValueError("Invalid correlation type. Use 'kendall', 'pearson', or 'spearman'.")
#             corr_matrix.iat[i, j] = corr
#     return corr_matrix


# def correlation_heatmap(lists, corr_type):
#     # Extract names and values from the input lists
#     names, values = zip(*lists)
#
#     # Create a DataFrame from the extracted values
#     df = pd.DataFrame(list(zip(*values)), columns=names)
#
#     # Calculate the correlation matrix based on the specified correlation type
#     corr_matrix = calc_corr_matrix(df, corr_type)
#
#     # Create a heatmap from the correlation matrix
#     plt.figure(figsize=(10, 8))
#
#     fig, ax = plt.subplots(figsize=(21, 17))
#
#     sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax,
#                 annot_kws={"size": 16},
#                 xticklabels=list(map(str, names)),
#                 yticklabels=list(map(str, names)))
#
#     ax.set_title("Correlation Matrix: " + corr_type, fontsize=20, loc='right', x=1.3, y=1.05)
#
#     # Save the heatmap to the specified format
#     if not os.path.exists("out"):
#         os.makedirs("out")
#     file_name = f"{datetime.now().isoformat()}_{corr_type}_heatmap.png"
#     plt.savefig(f"out/{file_name}")
#
#     # Return the correlation matrix
#     return corr_matrix


# def categorize_correlations(lists, corr_type):
#     categories = {
#         "Very Weak": [],
#         "Weak": [],
#         "Moderate": [],
#         "Strong": [],
#         "Very Strong": []
#     }
#
#     # Create a DataFrame from the lists
#     data = pd.DataFrame(dict(lists))
#
#     # Compute the correlation matrix
#     corr_matrix = data.corr()
#
#     variable_names = corr_matrix.columns
#
#     for i in range(corr_matrix.shape[0]):
#         for j in range(i + 1, corr_matrix.shape[1]):
#             corr = corr_matrix.iloc[i, j]
#             if corr <= -0.7 or corr >= 0.7:
#                 categories["Very Strong"].append(((variable_names[i], variable_names[j]), corr))
#             elif corr <= -0.5 or corr >= 0.5:
#                 categories["Strong"].append(((variable_names[i], variable_names[j]), corr))
#             elif corr <= -0.3 or corr >= 0.3:
#                 categories["Moderate"].append(((variable_names[i], variable_names[j]), corr))
#             elif corr <= -0.1 or corr >= 0.1:
#                 categories["Weak"].append(((variable_names[i], variable_names[j]), corr))
#             else:
#                 categories["Very Weak"].append(((variable_names[i], variable_names[j]), corr))
#
#     if not os.path.exists("out"):
#         os.makedirs("out")
#
#     with open("out" + "/" + datetime.now().isoformat() + "_" + corr_type + "_" + "categories.txt", # 'w') as f:
#         for category, values in categories.items():
#             f.write(f"{category}:\n")
#             for value in values:
#                 f.write(f"- {value}\n")
#             f.write("\n")
#
#     return categories


# def cluster_correlation_matrix(corr_matrix, corr_type, n_clusters=2, # linkage_method='ward'):
#     """
#     Clusters a correlation matrix using hierarchical clustering.
#
#     Parameters:
#     corr_matrix (pandas.DataFrame): a correlation matrix as a pandas DataFrame
#     n_clusters (int): the number of clusters to create
#     linkage_method (str): the linkage method to use for clustering (default: 'ward')
#
#     Returns:
#     labels (numpy.array): an array of labels indicating which cluster each variable belongs to
#     """
#
#     # Convert correlation matrix to distance matrix
#     dist_matrix = np.sqrt(1 - np.abs(corr_matrix))
#
#     # Perform hierarchical clustering with specified linkage method
#     Z = linkage(squareform(dist_matrix), method=linkage_method)
#
#     # Assign labels to clusters
#     labels = fcluster(Z, t=n_clusters, criterion='maxclust')
#
#     if not os.path.exists("out"):
#         os.makedirs("out")
#
#     with open("out" + "/" + datetime.now().isoformat() + "_" + corr_type + "_" + # "cluster_output.txt", "w") as f:
#         for i in range(1, n_clusters + 1):
#             f.write(f"Cluster {i}:\n")
#             cluster_vars = [col for col, label in zip(corr_matrix.columns, labels) if label == i]
#             for var in cluster_vars:
#                 f.write(f"- {var}\n")
#            f.write("\n")

#    return labels


# from datetime import datetime
# from scipy.stats import pearsonr, spearmanr, kendalltau
# from scipy.spatial.distance import squareform
# from scipy.cluster.hierarchy import linkage, dendrogram
#
# import pandas as pd
# import numpy
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import os
#
# mpl.use('agg')
#
# def plot(analysis_method, x_values, x_name, y_values, y_name):
#     x = numpy.ravel(x_values)
#     y = numpy.ravel(y_values)
#
#     coefficient = None
#     p = None
#
#     if analysis_method == "pearson":
#         coefficient, p = pearsonr(x, y)
#
#         print("Pearson's Correlation Coefficient: {} and Probability Value: {}".format(# coefficient, p))
#
#     if analysis_method == "spearman":
#         coefficient, p = spearmanr(x, y)
#
#         print("Spearman's Correlation Coefficient: {} and Probability Value: {}".format(# coefficient, p))
#
#     if analysis_method == "kendall":
#         coefficient, p = kendalltau(x, y)
#
#         print("Kendall's Correlation Coefficient: {} and Probability Value: {}".format(# coefficient, p))
#
#     # Calculate coefficients, and function for the line of best fit.
#     data = numpy.polyfit(x, y, 1)
#     polynomial = numpy.poly1d(data)
#
#     # Plot the Values
#     plt.scatter(x, y, color='black', s=0.5, alpha=1, marker="o", linewidth=0, label="Data Points", # zorder=1, edgecolors=None, facecolors=None, antialiased=True, rasterized=None, norm=None, vmin=None, # vmax=None, data=None)
#     plt.plot(x, polynomial(x), color='black', linewidth=0.1, label="Linear Regression", zorder=1, # antialiased=True, rasterized=True, data=None)
#     plt.xlabel(x_name)
#     plt.ylabel(y_name)
#     plt.autoscale(enable=True, axis='both', tight=None)
#
#     legend_coefficient = plt.legend(["r = " + str(coefficient)], loc='upper right', bbox_to_anchor=(# 1.0, 1.0))
#
#     if analysis_method == "pearson":
#         legend_coefficient.set_title("Pearson's Correlation Coefficient")
#
#     if analysis_method == "spearman":
#         legend_coefficient.set_title("Spearman's Correlation Coefficient")
#
#     legend_coefficient.get_title().set_fontsize('small')
#     legend_coefficient.get_title().set_fontweight('bold')
#
#     legend_p = plt.legend(["p = " + str(p)], loc='upper right', bbox_to_anchor=(1.0, 0.85))
#     legend_p.set_title("Probability Value")
#     legend_p.get_title().set_fontsize('small')
#     legend_p.get_title().set_fontweight('bold')
#
#     plt.gca().add_artist(legend_coefficient)
#     plt.gca().add_artist(legend_p)
#
#     plt.grid(True, which='major', axis='both', linestyle='-', linewidth=0.05, color='grey', # alpha=0.75)
#
#     now = datetime.now()
#     iso_date_string = now.isoformat()
#
#     ## Limiting the x -axis to make the plot more readable.
#     plt.axis([min(x), max(x), min(y), max(y)])
#
#     if not os.path.exists("out"):
#         os.makedirs("out")
#
#     plt.savefig("out/" + iso_date_string + "_" + analysis_method + "_" + x_name + "_to_" + y_name + # '.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
#
#
# def visualize_categories(categories, corr_type, chart_type):
#     if chart_type == "bar":
#         x_labels = list(categories.keys())
#         y_values = [len(categories[k]) for k in x_labels]
#         fig, ax = plt.subplots(figsize=(8, 6))
#         ax.bar(x_labels, y_values, color=['#cccccc', '#cccccc', '#cccccc', '#cccccc', '#cccccc'])
#         ax.set_title('Correlation Categories')
#         ax.set_xlabel('Category')
#         ax.set_ylabel('Number of Correlations')
#         plt.xticks(rotation=45)
#         plt.tight_layout()
#         plt.savefig('correlation_categories.png')
#
#         if not os.path.exists("out"):
#             os.makedirs("out")
#
#         # Save the plot to a file
#         plt.savefig('out' + '/' + datetime.now().isoformat() + "_" + corr_type + "_" + # 'correlation_categories' + "_" + chart_type + '.png')
#
#     if chart_type == "bubble":
#         fig, ax = plt.subplots()
#         ax.set_xlabel('Variable 1')
#         ax.set_ylabel('Variable 2')
#         ax.set_title('Correlation Coefficient Bubble Chart')
#
#         for category, pairs in categories.items():
#             for pair in pairs:
#                 x, y = pair[0]
#                 size = abs(pair[1]) * 100
#                 color = 'red' if pair[1] < 0 else 'green'
#                 ax.scatter(x, y, s=size, c=color, alpha=0.5, label=category)
#
#         ax.legend(loc='best', title='Category')
#         fig.tight_layout()
#
#         if not os.path.exists("out"):
#             os.makedirs("out")
#
#         plt.savefig('out' + '/' + datetime.now().isoformat() + "_" + corr_type + "_" + # 'correlation_categories' + "_" + chart_type + '.png')
#
#
# def visualize_dendrogram(lists, corr_type, linkage_method='ward'):
#     """
#     Visualizes a dendrogram using hierarchical clustering.
#
#     Parameters:
#     lists (list): a list of tuples containing the column names and corresponding values
#     corr_type (str): the type of correlation used (e.g., "spearman", default: "spearman")
#     linkage_method (str): the linkage method to use for clustering (default: 'ward')
#
#     Returns:
#     None
#     """
#
#     # Create a DataFrame from the lists
#     data = pd.DataFrame(dict(lists))
#
#     # Calculate the correlation matrix
#     if corr_type == "spearman":
#         corr_matrix = data.corr(method='spearman')
#
#     if corr_type == "pearson":
#         corr_matrix = data.corr(method='pearson')
#
#     if corr_type == "kendall":
#         corr_matrix = data.corr(method='kendall')
#
#     # Force the correlation matrix to be symmetric
#     corr_matrix = (corr_matrix + corr_matrix.T) / 2
#
#     # Convert correlation matrix to distance matrix
#     dist_matrix = numpy.sqrt(1 - numpy.abs(corr_matrix))
#
#     # Convert the distance matrix DataFrame to a NumPy array
#     dist_matrix = dist_matrix.to_numpy()
#
#     # Set the diagonal of the distance matrix to zero
#     numpy.fill_diagonal(dist_matrix, 0)
#
#     # Perform hierarchical clustering
#     Z = linkage(squareform(dist_matrix), method='ward')
#
#     # Extract the variable names from the lists
#     variable_names = [tup[0] for tup in lists]
#
#     # Plot dendrogram with variable names
#    plt.figure(figsize=(10, 8))
#    dendrogram(Z, labels=variable_names, orientation='right', leaf_font_size=6)
#
#    # Set plot parameters
#    plt.xlabel('Distance')
#
#    if not os.path.exists("out"):
#        os.makedirs("out")
#
#    plt.savefig('out' + '/' + datetime.now().isoformat() + "_" + corr_type + "_" + 'dendogram.png')
