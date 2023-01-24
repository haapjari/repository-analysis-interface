from src.alys.analysis import *
from src.mdls.repository import *
from src.utils.utils import *
from src.plot.plot import *


def create_commandline_interface(normalized_dataset):
  # Boilerplate Command Line Interface
    while True:
        print("Welcome to Glass Draw")
        print("Choose an option (or 'q' to quit): ")

        print("1. Show correlation matrix heatmap")
        print("2. Show correlation matrix categories")
        print("3. Cluster the correlation matrix")
        print("4. Perform correlation analysis")

        user_input = input("> ")

        if user_input.lower() == "q":
            break

        try:
            choice = int(user_input)
        except ValueError:
            print("Invalid input. Please enter a number (1-4) or 'q' to quit.")
            continue

        if choice == 1:
            print("Choose a correlation type:")
            print("1. Pearson correlation (measures linear relationships)")
            print("2. Spearman correlation (measures monotonic relationships)")
            print("3. Kendall correlation (measures ordinal relationships)")
            print("4. Back to main menu")

            user_input = input("> ")

            if user_input.lower() == "q":
                break

            try:
                corr_choice = int(user_input)
            except ValueError:
                print("Invalid input. Please enter a number (1-4) or 'q' to quit.")
                continue

            if corr_choice == 1:
                heatmap(normalized_dataset, "pearson", ["repository_name", "repository_url"])
            elif corr_choice == 2:
                heatmap(normalized_dataset, "spearman", ["repository_name", "repository_url"])
            elif corr_choice == 3:
                heatmap(normalized_dataset, "kendall", ["repository_name", "repository_url"])
            elif corr_choice == 4:
                continue
            else:
                print("Invalid input. Please enter a number (1-4) or 'q' to quit.")
                continue

        elif choice == 2:
            print("Choose a correlation type:")
            print("1. Pearson correlation (measures linear relationships)")
            print("2. Spearman correlation (measures monotonic relationships)")
            print("3. Kendall correlation (measures ordinal relationships)")
            print("4. Back to main menu")

            user_input = input("> ")

            if user_input.lower() == "q":
                break

            try:
                corr_choice = int(user_input)
            except ValueError:
                print("Invalid input. Please enter a number (1-4) or 'q' to quit.")
                continue

            if corr_choice == 1:
                matrix, names = correlation_matrix(normalized_dataset, "pearson", ["repository_name", "repository_url"])
                categorize_correlations(matrix, names, "pearson")
            elif corr_choice == 2:
                matrix, names = correlation_matrix(normalized_dataset, "spearman", ["repository_name", "repository_url"])
                categorize_correlations(matrix, names, "spearman")
            elif corr_choice == 3:
                matrix, names = correlation_matrix(normalized_dataset, "kendall", ["repository_name", "repository_url"])
                categorize_correlations(matrix, names, "kendall")
            elif corr_choice == 4:
                continue
            else:
                print("Invalid input. Please enter a number (1-4) or 'q' to quit.")
                continue

        elif choice == 3:
            print("Choose a correlation type:")
            print("1. Pearson correlation (measures linear relationships)")
            print("2. Spearman correlation (measures monotonic relationships)")
            print("3. Kendall correlation (measures ordinal relationships)")
            print("4. Back to main menu")

            user_input = input("> ")

            if user_input.lower() == "q":
                break

            try:
                corr_choice = int(user_input)
            except ValueError:
                print("Invalid input. Please enter a number (1-4) or 'q' to quit.")
                continue

            if corr_choice == 1:
                matrix, names = correlation_matrix(normalized_dataset, "pearson", ["repository_name", "repository_url"])
                labels = cluster_correlation_matrix(matrix, "pearson", n_clusters=3)
                visualize_dendrogram(matrix, "pearson")
            elif corr_choice == 2:
                matrix, names = correlation_matrix(normalized_dataset, "spearman", ["repository_name", "repository_url"])
                labels = cluster_correlation_matrix(matrix, "spearman", n_clusters=3)
                visualize_dendrogram(matrix, "spearman")
            elif corr_choice == 3:
                matrix, names = correlation_matrix(normalized_dataset, "kendall", ["repository_name", "repository_url"])
                labels = cluster_correlation_matrix(matrix, "kendall", n_clusters=3)
                visualize_dendrogram(matrix, "kendall")
            elif corr_choice == 4:
                continue
            else:
                print("Invalid input. Please enter a number (1-4) or 'q' to quit.")
                continue

        elif choice == 4:
            print("Choose a correlation type:")
            print("1. Pearson correlation (measures linear relationships)")
            print("2. Spearman correlation (measures monotonic relationships)")
            print("3. Kendall correlation (measures ordinal relationships)")
            print("4. Back to main menu")

            user_input = input("> ")

            if user_input.lower() == "q":
                break
           
            valid_columns = ["open_issue_count", "closed_issue_count", "commit_count", 
                             "open_closed_ratio", "stargazer_count", "creation_date", 
                             "latest_release", "original_codebase_size", 
                             "library_codebase_size", "library_to_original_ratio", 
                             "quality_measure", "total_codebase_size", "total_issue_count",
                             "maturity_score", "activity_score"]

            print(f"Valid columns: {', '.join(valid_columns)}")
            print("Choose the x value:") 
            x = input("> ")
            print("Choose the y -value:")
            y = input("> ")
            
            try:
                corr_choice = int(user_input)
            except ValueError:
                print("Invalid input. Please enter a number (1-4) or 'q' to quit.")
                continue

            if corr_choice == 1:
                plot("pearson", normalized_dataset, x, y)
            elif corr_choice == 2:
                plot("spearman", normalized_dataset, x, y) 
            elif corr_choice == 3:
                plot("kendall", normalized_dataset, x, y) 
            elif corr_choice == 4:
                continue
            else:
                print("Invalid input. Please enter a number (1-4) or 'q' to quit.")
                continue