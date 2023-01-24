from datetime import datetime

import csv
import pandas as pd
import seaborn as sns
import analysis.analysis as analysis
import matplotlib.pyplot as plt
import numpy as np

def main():

    # --------------------------------------------------------- #
    # 1. Read Dataset From CSV File
    # --------------------------------------------------------- #

    open_issues = []
    closed_issues = []
    commits = []
    self_written_loc = []
    library_loc = []
    creation_date = []
    stargazers = []
    latest_release = []
    forks = []
    open_pulls = []
    closed_pulls = []
    releases = []
    network_events = []
    subscribers = []
    contributors = []
    watchers = []
    
    with open('data/data.csv', newline='') as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            open_issues.append(row.get('open_issues'))
            closed_issues.append(row.get('closed_issues'))
            commits.append(row.get('commits'))
            self_written_loc.append(row.get('self_written_loc'))
            library_loc.append(row.get('library_loc'))
            creation_date.append(row.get('creation_date'))
            stargazers.append(row.get('stargazers'))
            latest_release.append(row.get('latest_release'))
            forks.append(row.get('forks'))
            open_pulls.append(row.get('open_pulls'))
            closed_pulls.append(row.get('closed_pulls'))
            releases.append(row.get('releases'))
            network_events.append(row.get('network_events'))
            subscribers.append(row.get('subscribers'))
            contributors.append(row.get('contributors'))
            watchers.append(row.get('watchers'))

    # library_loc_proportion = []
    # self_written_loc_proportion = []
    # total_loc = []

    # for i in range(len(library_loc)):
    #     total = self_written_loc[i] + library_loc[i]
    #     total_loc.append(total)
    #     library_loc_proportion.append(library_loc[i] / total)
    #     self_written_loc_proportion.append(self_written_loc[i] / total)

    # TODO

    data = pd.DataFrame({
        'stargazers': stargazers,
        'forks': forks,
        'subscribers': subscribers,
        'watchers': watchers,
        'open_issues': open_issues,
        'closed_issues': closed_issues,
        'commits': commits,
        'open_pulls': open_pulls,
        'closed_pulls': closed_pulls,
        'network_events': network_events,
        'contributors': contributors,
        'creation_date': creation_date,
        'latest_release': latest_release,
        'releases': releases,
    })

    stargazers = [float(x) for x in stargazers]
    forks = [float(x) for x in forks]
    subscribers = [float(x) for x in subscribers]
    watchers = [float(x) for x in watchers]
    open_issues = [float(x) for x in open_issues]
    closed_issues = [float(x) for x in closed_issues]
    commits = [float(x) for x in commits]
    open_pulls = [float(x) for x in open_pulls]
    closed_pulls = [float(x) for x in closed_pulls]
    network_events = [float(x) for x in network_events]
    contributors = [float(x) for x in contributors]
    creation_date = [float(x) for x in creation_date]
    latest_release = [float(x) for x in latest_release]
    releases = [float(x) for x in releases]

    bw_adj = 1
    linewidth = 1

    # Plotting the distribution
    sns.kdeplot(stargazers, bw_adjust=bw_adj, linewidth=linewidth)
    sns.kdeplot(forks, bw_adjust=bw_adj, linewidth=linewidth)
    sns.kdeplot(subscribers, bw_adjust=bw_adj, linewidth=linewidth)
    sns.kdeplot(watchers, bw_adjust=bw_adj, linewidth=linewidth)
    sns.kdeplot(open_issues, bw_adjust=bw_adj, linewidth=linewidth)
    sns.kdeplot(closed_issues, bw_adjust=bw_adj, linewidth=linewidth)
    sns.kdeplot(commits, bw_adjust=bw_adj, linewidth=linewidth)
    sns.kdeplot(open_pulls, bw_adjust=bw_adj, linewidth=linewidth)
    sns.kdeplot(closed_pulls, bw_adjust=bw_adj, linewidth=linewidth)
    sns.kdeplot(network_events, bw_adjust=bw_adj, linewidth=linewidth)
    sns.kdeplot(contributors, bw_adjust=bw_adj, linewidth=linewidth)
    sns.kdeplot(creation_date, bw_adjust=bw_adj, linewidth=linewidth)
    sns.kdeplot(latest_release, bw_adjust=bw_adj, linewidth=linewidth)
    sns.kdeplot(releases, bw_adjust=bw_adj, linewidth=linewidth)

    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.75)

    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.5)

    # Save the plot to the "out" folder
    file_name = f"{datetime.now().isoformat()}_distributions.png"
    plt.savefig(f"out/{file_name}")
    
    # analysis.visualize_multiple_distributions(data)
    # analysis.visualize_distribution("watchers", watchers)

    # Calculate composite scores for each group of metrics
    # popularity_score = data.iloc[:, :4].sum(axis=1)
    # activity_score = data.iloc[:, 4:11].sum(axis=1)
    # maturity_score = data.iloc[:, 11:].sum(axis=1)

    # Combine the composite scores into a single DataFrame
    # composite_scores = pd.DataFrame({
    #     'popularity_score': popularity_score,
    #     'activity_score': activity_score,
    #     'maturity_score': maturity_score,
    #     'library_loc_proportion': library_loc_proportion
    # })

    # correlation_matrix, p_value_matrix = spearmanr(composite_scores)

    # Extract the correlation coefficients and p-values for each independent variable
    # correlations = correlation_matrix[-1, :-1]
    # p_values = p_value_matrix[-1, :-1]

    # print("Coefficients: ", correlations)
    # print("Probability Values:", p_values)

    # variable_names = ['Popularity', 'Activity', 'Maturity']

    # Create bar plot for correlation coefficients
    # plt.figure(figsize=(8, 4))
    # plt.bar(variable_names, correlations)
    # plt.ylabel("Spearman's Rank Correlation Coefficient")

    # plt.axhline(y=0.00, color='black', linestyle='--', linewidth=0.5)

    # Save the correlation coefficients plot
    # if not os.path.exists("out"):
    #     os.makedirs("out")
    # file_name = f"{datetime.now().isoformat()}_spearman_correlation_coefficients.png"
    # plt.savefig(f"out/{file_name}")
    # plt.close()

    # print("Variable Names: ", variable_names)
    # print("P-Values: ", p_values)

    # # Create bar plot for p-values

    # plt.figure(figsize=(8, 4))
    # plt.bar(variable_names, p_values)
    # plt.axhline(y=significance_level, color='r', linestyle='--')
    # plt.ylabel('Probability Values')
    # plt.yscale('log')

    # # Create custom legend
    # legend_elements = [
    #     Line2D([0], [0], color='r', linestyle='--', lw=2, label=f"Significance level ({significance_level})")]
    # plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 0.95))

    # # Save the p-values plot
    # if not os.path.exists("out"):
    #     os.makedirs("out")
    # file_name = f"{datetime.now().isoformat()}_spearman_p_values_log_scale.png"
    # plt.savefig(f"out/{file_name}")
    # plt.close()

    # Close the Cursor and Connection


if __name__ == "__main__":
    main()
