import os
import csv
from github import Github
from dotenv import load_dotenv
from datetime import datetime, timezone

class GitHub:
    """
        Wrapper for "PyGitHub" that initializes a GitHub instance using the 
        authentication token from the environment variables.
    """

    def __init__(self):
        load_dotenv()
        self.api_token = os.environ.get("GITHUB_API_TOKEN")
        self.gh = Github(self.api_token)

    
    def get_repo(self, owner, name):
        """
        Get a repository by its owner and name.

        Args:
            owner: Owner of the repository.
            name: Name of the repository.

        Returns:
            A PyGitHub Repository object.
        """

        full_name =f"{owner}/{name}"
        return self.gh.get_repo(full_name)


    def get_avg_weekly_additions(self, repo):
        stats_code_frequency = repo.get_stats_code_frequency() 
        avg_weekly_additions = 0
        avg_weekly_deletions = 0
        week_count = len(stats_code_frequency)
    
        for obj in stats_code_frequency:
            avg_weekly_additions = avg_weekly_additions + obj.additions
            avg_weekly_deletions = avg_weekly_deletions + obj.deletions

        avg_weekly_additions = avg_weekly_additions / week_count
        avg_weekly_deletions = avg_weekly_deletions / week_count

        return avg_weekly_additions, avg_weekly_deletions


# TODO: Untested
# def append_data(input_file, output_file, additional_fields):
    # # Read in the data from the input CSV file
    # with open(input_file) as f:
        # reader = csv.reader(f)
        # header = next(reader)
        # data = list(reader)

    # # Create a PyGithub object with the API token from the .env file
    # g = Github(os.environ["GITHUB_API_TOKEN"])

    # # Loop over each repository in the data and fetch the additional information from the API
    # for row in data:
        # # Extract the repository owner and name from the URL in the second column
        # owner, name = row[1].split("/")[-2:]

        # try:
            # repo = g.get_repo(f"{owner}/{name}")
        # except Exception as e:
            # print(f"Error: {e}")
            # continue

        # # Add the additional information to the row
        # row.extend([getattr(repo, field, "") for field in additional_fields])

    # # Write out the extended data to a new CSV file
    # with open(output_file, "w") as f:
        # writer = csv.writer(f)
        # writer.writerow(header + additional_fields)
        # writer.writerows(data)


# # This script tests the Github API.
# # The script uses the Github API to retrieve the kubernetes/kubernetes repository and then prints out the available fields for the repository.
# def test_github_api(field):
    # g = Github(os.environ["GITHUB_API_TOKEN"])

    # try:
        # repo = g.get_repo("kubernetes/kubernetes")
    # except Exception as e:
        # print(f"Error: {e}")

    # if field == "get_pulls":
        # attr = getattr(repo, field, "")
        # attr = attr(state="all")
        # return attr

    # attr = getattr(repo, field, "")
    # attr = attr()

    # return attr

    # # Print out the available fields for the repository
# #    print("Available fields for the repository:")
    # #for attr in dir(repo):
        # #if not attr.startswith("_"):
            # #print(attr)


# # TODO: Untested
# # This function queries the latest 10 commits to the kubernetes/kubernetes repository
# def query_latest_commits():
    # # Create a PyGithub object with the API token from the .env file
    # g = Github(os.environ["GITHUB_API_TOKEN"])

    # # Get the kubernetes/kubernetes repository
    # repo = g.get_repo("kubernetes/kubernetes")

    # # Get the latest 10 commits for the repository
    # commits = repo.get_commits()[0:10]

    # # Loop over the commits and print out their dates and total number of changes
    # for commit in commits:
        # # Get the commit object for this commit
        # full_commit = repo.get_commit(commit.sha)

        # # Calculate the total number of changes
        # total_changes = full_commit.stats.additions + full_commit.stats.deletions

        # # Print out the date and total number of changes
        # print(f"Date: {full_commit.commit.author.date}")
        # print(f"Changes: {total_changes}")

# # TODO: Untested
# def query_latest_commits():
    # # Create a PyGithub object with the API token from the .env file
    # g = Github(os.environ["GITHUB_API_TOKEN"])

    # # Get the kubernetes/kubernetes repository
    # repo = g.get_repo("kubernetes/kubernetes")

    # # Get the latest 10 commits for the repository
    # commits = repo.get_commits()[0:10]

    # # Loop over the commits and print out their dates and total number of changes
    # for commit in commits:
        # # Get the commit object for this commit
        # full_commit = repo.get_commit(commit.sha)

        # # Calculate the total number of changes
        # total_changes = full_commit.stats.additions + full_commit.stats.deletions

        # # Print out the date and total number of changes
        # print(f"Date: {full_commit.commit.author.date}")
        # print(f"Changes: {total_changes}")


# # TODO: Untested
# def get_commit_info(repo_name, owner):
    # # Create a PyGithub object with the API token from the .env file
    # g = Github(os.environ["GITHUB_API_TOKEN"])

    # # Get the repository
    # repo = g.get_repo(f"{owner}/{repo_name}")

    # # Initialize the date range for the commits to the creation date of the repository and the current date
    # since_date = repo.created_at
    # until_date = datetime.now(timezone.utc)

    # # Initialize the dictionary of commit hashes mapped to their date and total number of changes
    # commit_info = {}

    # # Loop over the pages of commits and add their date and total number of changes to the dictionary
    # while True:
        # commits = repo.get_commits(since=since_date, until=until_date)
        # for commit in commits:
            # full_commit = repo.get_commit(commit.sha)
            # total_changes = full_commit.stats.additions + full_commit.stats.deletions
            # commit_info[commit.sha] = (full_commit.commit.author.date, total_changes)
        # if len(commits) == 0:
            # break
        # since_date = commits[-1].commit.author.date

    # # Return the dictionary of commit hashes mapped to their date and total number of changes
    # return commit_info


# def get_mean_additions_per_week(owner, repo):
    # """
    # Computes the mean number of additions per week for a given repository.

    # :param owner: The repository owner.
    # :param repo: The repository name.
    # :return: The mean number of additions per week.
    # """
    # g = Github(os.environ["GITHUB_API_TOKEN"])
    # repo = g.get_repo(f"{owner}/{repo}")
    # data = repo.get_stats_code_frequency()
    # weeks = []
    # print(data)
    # #for stats in data:
        # #if hasattr(stats, 'week') and isinstance(stats.week, list) and len(stats.week) > 0 and isinstance(stats.week[0], dict): 
            # #weeks.extend(stats.week)
    # #num_weeks = len(weeks)
    # #num_additions = sum(w[1] for w in weeks)
    # #return num_additions / num_weeks


# def get_contributor_activity(repo: Repository) -> Dict[str, dict]:
    # """
    # Retrieve the contributors and their commit activity for a repository.

    # Args:
        # repo: A PyGitHub repository object.

    # Returns:
        # A dictionary with the contributor names as keys and their commit activity as values.
    # """
    # # Call the contributors endpoint
    # contributors = repo.get_stats_contributors()

    # # Create a dictionary to store the contributor activity
    # contributor_activity = {}

    # # Loop through the contributors and their commit activity
    # for contributor in contributors:
        # # Get the contributor name
        # name = contributor.author.login

        # # Initialize the contributor's activity dictionary
        # activity = {"total_commits": contributor.total, "weekly_activity": {}}

        # # Loop through the weekly activity and add it to the contributor's activity dictionary
        # for week in contributor.weeks:
            # activity["weekly_activity"][str(week["w"])] = {
                # "additions": week["a"],
                # "deletions": week["d"],
                # "commits": week["c"],
            # }

        # # Add the contributor's activity to the dictionary
        # contributor_activity[name] = activity

    # return contributor_activity