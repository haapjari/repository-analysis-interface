from typing import Dict, List, Tuple

class Repository:
    def __init__(self, repository_name="", repository_url="", open_issue_count=0, closed_issue_count=0, commit_count=0, open_closed_ratio=0, stargazer_count=0, creation_date="", latest_release="", original_codebase_size=0, library_codebase_size=0, library_to_original_ratio=0, quality_measure=0):
        self.repository_name = repository_name
        self.repository_url = repository_url
        self.open_issue_count = open_issue_count
        self.closed_issue_count = closed_issue_count
        self.commit_count = commit_count
        self.open_closed_ratio = open_closed_ratio
        self.stargazer_count = stargazer_count
        self.creation_date = creation_date
        self.latest_release = latest_release
        self.original_codebase_size = original_codebase_size
        self.library_codebase_size = library_codebase_size
        self.library_to_original_ratio = library_to_original_ratio
        self.quality_measure = quality_measure


def extract_repositories(dataset):
    """
    Extracts repository data from a dataset and returns a list of Repository objects.

    Args:
        dataset (dict): A dictionary representing the dataset, where each key represents a column
        and each value is a list of tuples containing the URL and corresponding data for that column.

    Returns:
        list: A list of Repository objects, where each object contains the data for a single repository.
    """
    repos = []

    # repository_url
    for column, values in dataset.items():
        if column == "repository_url":
            for value in values:
                repo = Repository()
                repo.repository_url = value[1]
                repos.append(repo) 

    # repository_name
    for column, values in dataset.items():
        if column == "repository_name":
            for value in values:
                repo = search_repository(repos, value[0])
                repo.repository_name = value[1]
    
    # open_issue_count
    for column, values in dataset.items():
        if column == "open_issue_count":
            for value in values:
                repo = search_repository(repos, value[0])
                repo.open_issue_count = value[1]
    
    # closed_issue_count 
    for column, values in dataset.items():
        if column == "closed_issue_count":
            for value in values:
                repo = search_repository(repos, value[0])
                repo.closed_issue_count = value[1]
    
    # commit_count
    for column, values in dataset.items():
        if column == "commit_count":
            for value in values:
                repo = search_repository(repos, value[0])
                repo.commit_count = value[1]
    
    # open_closed_ratio
    for column, values in dataset.items():
        if column == "open_closed_ratio":
            for value in values:
                repo = search_repository(repos, value[0])
                repo.open_closed_ratio = value[1]
    
    # stargazer_count
    for column, values in dataset.items():
        if column == "stargazer_count":
            for value in values:
                repo = search_repository(repos, value[0])
                repo.stargazer_count = value[1]
    
    # creation_date
    for column, values in dataset.items():
        if column == "creation_date":
            for value in values:
                repo = search_repository(repos, value[0])
                repo.creation_date = value[1]
    
    # latest_release
    for column, values in dataset.items():
        if column == "latest_release":
            for value in values:
                repo = search_repository(repos, value[0])
                #print(value)
                repo.latest_release = value[1]
    
    # original_codebase_size
    for column, values in dataset.items():
        if column == "original_codebase_size":
            for value in values:
                repo = search_repository(repos, value[0])
                repo.original_codebase_size = value[1]
    
    # library_codebase_size
    for column, values in dataset.items():
        if column == "library_codebase_size":
            for value in values:
                repo = search_repository(repos, value[0])
                repo.library_codebase_size = value[1]
    
    # library_to_original_ratio
    for column, values in dataset.items():
        if column == "library_to_original_ratio":
            for value in values:
                repo = search_repository(repos, value[0])
                repo.library_to_original_ratio = value[1]
    
    return repos

def search_repository(repos, url):
    """
    Searches a list of Repository objects for a repository with the specified URL and returns the corresponding object.

    Args:
        repos (list): A list of Repository objects to search through.
        url (str): The URL of the repository to search for.

    Returns:
        Repository or None: The Repository object with the specified URL, or None if no matching repository is found.
    """
    for repo in repos:
        if repo.repository_url == url:
            return repo
