import argparse
import sys

from src.dataset.dataset import *
from src.visual.visual import *
from src.config.config import *
from src.script.script import *


def main():
    log.basicConfig(level=log.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(
        description="examples:\n"
                    "  python -m src.main --collect 2008-01-01 2008-06-01 Go 100 10000 desc\n"
                    "  python -m src.main --normalize\n"
                    "  python -m src.main --drop --table table --column column\n"
                    "  python -m src.main --weighted --variables stargazers forks --name popularity\n"
                    "  python -m src.main --dist --variables stargazers --output ./output.png\n"
                    "  python -m src.main --plot --variables stargazers forks --correlation pearson --output ./plot.png\n"
                    "  python -m src.main --heatmap --variables stargazers forks commits --correlation pearson --output ./heatmap.png\n"
                    "  python -m src.main --regression --method linear --dependent third_party_loc self_written_loc --independent forks commits\n"
                    "  python -m src.main --cluster --method hierarchical --variables forks commit_count --output ./dendogram.png"
                    "  python -m src.main --vif -variables forks commit_count",
        formatter_class=argparse.RawTextHelpFormatter,
        add_help=False)

    parser.add_argument('--help', action='help', help='Help Message')

    parser.add_argument('--collect', action='store_true', help='Collect')
    parser.add_argument('--normalize', action='store_true', help='Normalize')
    parser.add_argument('--weighted', action='store_true', help='Weighted Sum')
    parser.add_argument('--dist', action='store_true', help='Distributions')
    parser.add_argument('--plot', action='store_true', help='Plots')
    parser.add_argument('--heatmap', action='store_true', help='Heatmap')
    parser.add_argument('--regression', action='store_true', help='Regression')
    parser.add_argument('--drop', action='store_true', help="Drop a Column from a Database Table")
    parser.add_argument('--cluster', action='store_true', help="Clustering")
    parser.add_argument('--vif', action='store_true', help="Clustering")

    
    if parser.parse_known_args()[0].vif:
        cluster_group = parser.add_argument_group('VIF Options')
        cluster_group.add_argument('--variables', nargs='+', required=True, help='Variables')

    if parser.parse_known_args()[0].cluster:
        cluster_group = parser.add_argument_group('Cluster Options')
        cluster_group.add_argument('--method', type=str, required=True, choices=['hierarchical'], 
                              default='hierarchical', help='Clustering Method') 
        cluster_group.add_argument('--variables', nargs='+', required=True, help='Variables')
        cluster_group.add_argument('--output', type=str, help='Output Path')

    if parser.parse_known_args()[0].drop:
        drop_group = parser.add_argument_group('Drop Options')
        drop_group.add_argument('--table', type=str, required=True, help='Table to Drop Column From') 
        drop_group.add_argument('--column', type=str, required=True, help='Column to Drop Column From') 


    if parser.parse_known_args()[0].regression:
        regression_group = parser.add_argument_group('Regression Options')
        regression_group.add_argument('--method', type=str, required=True, choices=['linear', 'quantile'], 
                                   default='linear', help='Regression Method') 
        regression_group.add_argument('--dependent', nargs='+', required=True, help='Dependent Variables') 
        regression_group.add_argument('--independent', nargs='+', required=True, help='Independent Variables')

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

    if parser.parse_known_args()[0].weighted:
        plot_group = parser.add_argument_group('Weighted Sum Options')
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

    elif args.weighted:
        vars: list = args.variables
        name: str = args.name
        c = Config()
        d = Dataset(c)

        try:
            d.weighted(vars, name)
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

    elif args.regression:
        c = Config()
        v: Visual = Visual(c)

        method: str = args.method
        dependent: dict = args.dependent
        independent: dict = args.independent 

        try:
            v.regression(method, dependent, independent)
        except Exception as e:
            log.error(e)
            sys.exit(1)
    
    elif args.cluster:
        c = Config()
        v: Visual = Visual(c)

        method: str = args.method
        variables: list = args.variables
        output: str = args.output

        try:
            v.cluster(method, variables, output)
        except Exception as e:
            log.error(e)
            sys.exit(1)

    elif args.drop:
        c = Config()
        
        table: str = args.table
        column: str = args.column
        
        s = Script(c, table, column)

        try:
            s.drop()
        except Exception as e:
            log.error(e)
            sys.exit(1)

    elif args.vif:
        c = Config()
        
        variables: list = args.variables
        
        d = Dataset(c)

        try:
            d.collinearity(variables)
        except Exception as e:
            log.error(e)
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
