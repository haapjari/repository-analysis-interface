# Glass Draw (glsdrw)

This is a data analysis script, which I used in my thesis.

This is a Python script that reads a cleaned dataset from a CSV file, normalizes and adjusts the data, and calculates quality measures for repositories extracted from the dataset. It then prompts the user to input parameters for a plot, which can be generated using either Pearson correlation or Spearman correlation.

The purpose of this script is to provide a tool for analyzing and visualizing relationships between various metrics for software repositories. It could be used, for example, to identify correlations between the size of a repository and the number of open or closed issues, or between the age of a repository and the number of stargazers.

## Example

Here is an example usage of this script:

1. Ensure that all required dependencies are installed.
2. Save the cleaned dataset to a CSV file.

In the specific case of this script, the dataset file should contain information about software repositories, including attributes such as repository name, repository URL, open issue count, closed issue count, commit count, open-closed ratio, stargazer count, creation date, latest release, original codebase size, library codebase size, and library-to-original ratio.

The file should be cleaned and preprocessed to ensure that all data is in a consistent format and that any missing or invalid values have been handled appropriately.

3. Run the script from the command line using `python3 main.py`
4. Follow the prompts to choose a correlation type and input parameters for the plot.
5. The script will generate a plot showing the relationship between the selected parameters and saves that to the `out` folder.
