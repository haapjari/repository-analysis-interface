import argparse
import logging as log
import sys

from src.dataset.dataset import *
from src.visual.visual import *
from src.config.config import *
from src.utils.utils import *


def main():
    log.basicConfig(level=log.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(
        description="examples:\n"
                    "  python -m src.main --collect 2008-01-01 2008-06-01 Go 100 10000 desc\n"
                    "  python -m src.main --normalize\n"
                    "  python -m src.main --composite --variables 'stargazers, forks' --name 'popularity'\n"
                    "  python -m src.main --dist --variables stargazers --output ./output.png\n"
                    "  python -m src.main --plot --variables stargazers forks --correlation pearson --output ./output/plot.png\n"
                    "  python -m src.main --heatmap --variables stargazers forks commits --correlation pearson --output ./output/heatmap.png\n",
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False)

    parser.add_argument('--help', action='help', help='Help Message')

    parser.add_argument('--collect', action='store_true', help='Start Dataset Collection')
    parser.add_argument('--normalize', action='store_true', help='Normalize Collected Dataset')
    parser.add_argument('--composite', action='store_true', help='Calculate Composite Variables')
    parser.add_argument('--dist', action='store_true', help='Draw Distribution Image to the Output Path')
    parser.add_argument('--plot', action='store_true', help='Draw Plot Image to the Output Path')
    parser.add_argument('--heatmap', action='store_true', help='Draw Heatmap to the Output Path')

    if parser.parse_known_args()[0].collect:
        collect_group = parser.add_argument_group('Collect Options')
        collect_group.add_argument('first_creation_date', type=str, help='First Creation Date (YYYY-MM-DD)')
        collect_group.add_argument('last_creation_date', type=str, help='Last Creation Date (YYYY-MM-DD)')
        collect_group.add_argument('language', type=str, help='Programming Language')
        collect_group.add_argument('min_stars', type=int, help='Minimum number of stars')
        collect_group.add_argument('max_stars', type=int, help='Maximum number of stars')
        collect_group.add_argument('order', type=str, help='Order: "asc" or "desc"', choices=['asc', 'desc'])

    if parser.parse_known_args()[0].dist:
        dist_group = parser.add_argument_group('Distribution Options')
        dist_group.add_argument('--variables', nargs='+', required=True, help='Variables for Distributions')
        dist_group.add_argument('--output', type=str, required=True, help='Output path for the distribution plots')

    if parser.parse_known_args()[0].plot:
        plot_group = parser.add_argument_group('Plot Options')
        plot_group.add_argument('--variables', nargs='+', required=True, help='Variables for Plot')
        plot_group.add_argument('--correlation', type=str, choices=['spearman', 'kendall', 'pearson'],
                                default='pearson', help='Correlation type: spearman, kendall or pearson')
        plot_group.add_argument('--output', type=str, required=True, help='Output path for the plot')

    if parser.parse_known_args()[0].heatmap:
        plot_group = parser.add_argument_group('Heatmap Options')
        plot_group.add_argument('--variables', nargs='+', required=True, help='Variables for Plot')
        plot_group.add_argument('--correlation', type=str, choices=['spearman', 'kendall', 'pearson'],
                                default='pearson', help='Correlation type: spearman, kendall or pearson')
        plot_group.add_argument('--output', type=str, required=True, help='Output path for the plot')

    if parser.parse_known_args()[0].composite:
        plot_group = parser.add_argument_group('Composite Score Options')
        plot_group.add_argument('--variables', nargs='+', required=True, help='Variables for Plot')
        plot_group.add_argument('--name', type=str, required=True, help='New Name')

    args = parser.parse_args()

    if args.collect:
        c = Config()
        d = Dataset(c, args.first_creation_date, args.last_creation_date, args.language, args.min_stars, args.max_stars,
                    args.order)
        try:
            d.collect()
        except Exception as e:
            log.error(e)
            sys.exit(1)

    elif args.normalize:
        c = Config()
        d = Dataset(c)

        try:
            d.normalize()
        except Exception as e:
            log.error(e)
            sys.exit(1)

    elif args.dist:
        c: Config = Config()
        v: Visual = Visual(c)

        variables: dict = args.variables
        output: str = args.output

        try:
            v.dist(variables, output)
        except Exception as e:
            log.error(e)
            sys.exit(1)

    elif args.composite:
        variables: list = args.variables
        name: str = args.name
        c = Config()
        d = Dataset(c)

        try:
            d.composite(variables, name)
        except Exception as e:
            log.error(e)
            sys.exit(1)

    elif args.plot:
        c = Config()
        v: Visual = Visual(c)

        variables: dict = args.variables
        output: str = args.output
        correlation: str = args.correlation

        try:
            v.plot(variables, correlation, output)
        except Exception as e:
            log.error(e)
            sys.exit(1)

    elif args.heatmap:
        c = Config()
        v: Visual = Visual(c)

        variables: dict = args.variables
        output: str = args.output
        correlation: str = args.correlation

        try:
            v.heatmap(variables, correlation, output)
        except Exception as e:
            log.error(e)
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()




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
