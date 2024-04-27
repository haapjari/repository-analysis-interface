import json
import requests

from datetime import datetime, timedelta
from src.config.config import get
from src.utils.utils import normalize_numeric, normalize_dates


class Dataset:
    def __init__(self, config, first_date=None, last_date=None, language=None, min_stars=None, max_stars=None,
                 order: str = None):
        self.config = config
        self.first_date = first_date
        self.last_date = last_date
        self.language = language
        self.min_stars = min_stars
        self.max_stars = max_stars
        self.order = order

    def collect(self):
        """
        Collect data from the Search API and send it to the Database API.
        """
        start_date = datetime.strptime(self.first_date, "%Y-%m-%d")
        end_date = datetime.strptime(self.last_date, "%Y-%m-%d")
        delta = end_date - start_date

        search_api_host = get("SEARCH_API_HOST")
        database_api_host = get("DATABASE_API_HOST")
        token = get("GITHUB_TOKEN")

        if delta.days > 7:
            current_date = start_date
            while current_date < end_date:
                week_end = min(current_date + timedelta(days=7), end_date)

                s = f"{search_api_host}/api/v1/repos/search?firstCreationDate={current_date.date()}&lastCreationDate={week_end.date()}&language={self.language}&minStars={self.min_stars}&maxStars={self.max_stars}&order={self.order}"

                response = requests.get(s, headers={"Authorization": f"Bearer {token}"})
                response.raise_for_status()

                data = response.json()

                if data["total_count"] > 0:
                    for repo in data["items"]:
                        if repo is not None:
                            url = f"{database_api_host}/api/v1/repos"

                            payload = {
                                "name": repo["name"],
                                "full_name": repo["full_name"],
                                "created_at": repo["created_at"],
                                "stargazer_count": repo["stargazer_count"],
                                "language": repo["language"],
                                "open_issues": repo["open_issues"],
                                "closed_issues": repo["closed_issues"],
                                "open_pull_request_count": repo["open_pull_request_count"],
                                "closed_pull_request_count": repo["closed_pull_request_count"],
                                "forks": repo["forks"],
                                "watcher_count": repo["watcher_count"],
                                "subscriber_count": repo["subscriber_count"],
                                "commit_count": repo["commit_count"],
                                "network_count": repo["network_count"],
                                "latest_release": repo["latest_release"],
                                "total_releases_count": repo["total_releases_count"],
                                "contributor_count": repo["contributor_count"],
                                "third_party_loc": repo["third_party_loc"],
                                "self_written_loc": repo["self_written_loc"]
                            }

                            headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}

                            response = requests.post(url, headers=headers, data=json.dumps(payload))

                            response.raise_for_status()

                current_date += timedelta(days=7 + 1)
        else:
            s = f"{search_api_host}/api/v1/repos/search?firstCreationDate={start_date.date()}&lastCreationDate={end_date.date()}&language={self.language}&minStars={self.min_stars}&maxStars={self.max_stars}&order={self.order}"

            response = requests.get(s, headers={"Authorization": f"Bearer {token}"})
            response.raise_for_status()

            data = response.json()

            if data["total_count"] > 0:
                for repo in data["items"]:
                    if repo is not None:
                        url = f"{database_api_host}/api/v1/repos"

                        payload = {
                            "name": repo["name"],
                            "full_name": repo["full_name"],
                            "created_at": repo["created_at"],
                            "stargazer_count": repo["stargazer_count"],
                            "language": repo["language"],
                            "open_issues": repo["open_issues"],
                            "closed_issues": repo["closed_issues"],
                            "open_pull_request_count": repo["open_pull_request_count"],
                            "closed_pull_request_count": repo["closed_pull_request_count"],
                            "forks": repo["forks"],
                            "watcher_count": repo["watcher_count"],
                            "subscriber_count": repo["subscriber_count"],
                            "commit_count": repo["commit_count"],
                            "network_count": repo["network_count"],
                            "latest_release": repo["latest_release"],
                            "total_releases_count": repo["total_releases_count"],
                            "contributor_count": repo["contributor_count"],
                            "third_party_loc": repo["third_party_loc"],
                            "self_written_loc": repo["self_written_loc"]
                        }

                        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
                        response = requests.post(url, headers=headers, data=json.dumps(payload))
                        response.raise_for_status()

    @staticmethod
    def normalize():
        """
        Normalize the collected data and send it to the Database API.
        """
        database_api_host = get("DATABASE_API_HOST")
        s = f"{database_api_host}/api/v1/repos"

        response = requests.get(s)
        response.raise_for_status()

        repos = response.json()

        normalized_repos = {}
        numeric_fields = [
            "stargazer_count", "open_issues", "closed_issues",
            "open_pull_request_count", "closed_pull_request_count",
            "forks", "watcher_count", "subscriber_count", "commit_count",
            "network_count", "total_releases_count", "contributor_count",
            "third_party_loc", "self_written_loc"
        ]

        # This loop processes each repository in the 'repos' list by normalizing dates and numeric data.
        # For each repository, it performs the following steps:
        #
        # 1. Creates a copy of the repository data.
        # 2. Extracts and normalizes the date fields ('created_at' and 'latest_release').
        # 3. Updates the repository copy with the normalized date values.
        # 4. Collects numeric fields to be normalized, if they exist in the repository.
        # 5. Normalizes these numeric fields and updates the repository copy with these values.
        # 6. Stores each updated repository in the 'normalized_repos' dictionary using the repository's name as the key.
        for r in repos:
            repo_name = r['name']
            normalized_repo = r.copy()

            dates = {"created_at": r["created_at"], "latest_release": r["latest_release"]}
            normalized_dates = normalize_dates(dates)

            normalized_repo.update(normalized_dates)

            numeric_values = {field: r[field] for field in numeric_fields if field in r}
            if numeric_values:
                normalized_numerics = normalize_numeric(numeric_values)
                normalized_repo.update(normalized_numerics)

            normalized_repos[repo_name] = normalized_repo

        # This loop sends the normalized repository data to the database API.
        # For each repository in the 'normalized_repos' dictionary, it performs the following steps:
        # 1. Constructs the URL for the normalized repository data.
        # 2. Converts the repository data to JSON format.
        # 3. Sends a POST request to the database API with the JSON data.
        for repo_name, repo_data in normalized_repos.items():
            s = f"{database_api_host}/api/v1/repos/normalized"
            headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}

            data_json = json.dumps(repo_data)

            response = requests.post(s, headers=headers, data=data_json)

            response.raise_for_status()

    @staticmethod
    def composite(variables: dict, name: str) -> int:
        """
        Composite Function.
        """
        # TODO

        # Read Normalized Data from the Database API
        # Match the "Variables" from the Data
        # Create Weighted Sum of the "Variables"
        # Store that to New Variable: "Name"

        pass


