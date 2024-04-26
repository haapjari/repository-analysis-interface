import json
import requests

from datetime import datetime, timedelta
from src.config.config import get


class Dataset:
    def __init__(self, config, first_date, last_date, language, min_stars, max_stars, order: str):
        self.config = config
        self.first_date = first_date
        self.last_date = last_date
        self.language = language
        self.min_stars = min_stars
        self.max_stars = max_stars
        self.order = order

    def collect(self):
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

    def normalize(self):
        # TODO
        print("we are in normalize")

        print("read n entries from database")
        print("normalize data")
        print("save normalized data to normalize database")

        pass

    def analyze(self):
        # TODO
        pass
