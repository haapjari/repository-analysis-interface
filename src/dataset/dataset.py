import json
import requests
import pandas as pd
import numpy as np

from datetime import datetime, timedelta
from src.config.config import get
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

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
            """
            If the date range is greater than 7 days, the function will split the range into weekly intervals and
            collect data for each week.
            """
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

                            total_loc = repo['self_written_loc'] + repo['third_party_loc']

                            self_written_loc_proportion = repo['self_written_loc'] / total_loc if total_loc > 0 else 0
                            third_party_loc_proportion = repo['third_party_loc'] / total_loc if total_loc > 0 else 0

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
                                "self_written_loc": repo["self_written_loc"],
                                "self_written_loc_proportion": self_written_loc_proportion,
                                "third_party_loc_proportion": third_party_loc_proportion,
                            }

                            headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}

                            response = requests.post(url, headers=headers, data=json.dumps(payload))

                            response.raise_for_status()

                current_date += timedelta(days=7 + 1)
        else:
            """
            If the date range is less than or equal to 7 days, the function will collect data for the entire range.
            """
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

        date_fields = [ "created_at", "latest_release" ]

        for r in repos:
            for field in date_fields:
                if field in r:
                    try:
                        r[field] = pd.to_datetime(r[field]).timestamp()
                    except (ValueError, TypeError, pd.errors.OutOfBoundsDatetime):
                        r[field] = pd.to_datetime("1970-01-01").timestamp()


        numeric_fields = [
            "stargazer_count", "open_issues", "closed_issues",
            "open_pull_request_count", "closed_pull_request_count",
            "forks", "watcher_count", "commit_count", "subscriber_count",
            "total_releases_count", "contributor_count", "network_count",
            "third_party_loc", "self_written_loc", "created_at", "latest_release",
            "self_written_loc_proportion", "third_party_loc_proportion"
        ]

        raw = {field: [] for field in numeric_fields}

        for r in repos:
            for field in numeric_fields:
                if field in r:
                    raw[field].append(r[field])

        normalized = {field: [] for field in numeric_fields}

        for field, data  in raw.items():
            scaler = MinMaxScaler()
            np_data = np.array(data).reshape(-1 ,1)
            normalized_data = scaler.fit_transform(np_data)
            normalized[field] = normalized_data

        normalized_repos = []

        for i in range(len(repos)):
            new_repo = {}
            for field in numeric_fields:
                if field in normalized and i < len(normalized[field]):
                    new_repo[field] = normalized[field][i][0]  
            new_repo['full_name'] = repos[i]['full_name']
            normalized_repos.append(new_repo)

        
        for r in normalized_repos:
            s = f"{database_api_host}/api/v1/repos/normalized"
            headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}

            data_json = json.dumps(r)
            response = requests.post(s, headers=headers, data=data_json)
            response.raise_for_status()


    @staticmethod
    def weighted(variables: list, name: str):
        """
        Create required composite variables on-demand, and update them in the Database Entries.
        """
        database_api_host = get("DATABASE_API_HOST")
        s = f"{database_api_host}/api/v1/repos/normalized"
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}

        response = requests.get(s, headers=headers)
        response.raise_for_status()

        repos = response.json()

        df = pd.DataFrame(repos)

        variables = [var.strip(',') for var in variables]

        num_variables = len(variables)
        even_weights = {var: 1.0 / num_variables for var in variables}

        for index, repo in df.iterrows():
            if all(var in repo for var in variables):
                weighted_average = sum(repo[var] * even_weights[var] for var in variables)
                df.at[index, name] = weighted_average

        updated_repos = df.to_dict('records')

        for updated_repo in updated_repos:
            url = f"{database_api_host}/api/v1/repos/normalized/{updated_repo['id']}"
            headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
            data = json.dumps(updated_repo)
            response = requests.put(url, headers=headers, data=data)
            response.raise_for_status()


    @staticmethod 
    def collinearity(variables: list):
        """
        Calculate and print the Variance Inflation Factor (VIF) for each variable.
        """

        database_api_host = get("DATABASE_API_HOST")
        s = f"{database_api_host}/api/v1/repos/normalized"
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}

        response = requests.get(s, headers=headers)
        response.raise_for_status()

        repos = response.json()
        df = pd.DataFrame(repos)

        # Ensure all variables are in the dataframe
        variables = [var.strip(',') for var in variables if var in df.columns]

        if not variables:
            return

        # Add a constant column for intercept
        df['intercept'] = 1

        vif_data = pd.DataFrame()
        vif_data["feature"] = variables
        vif_data["VIF"] = [variance_inflation_factor(df[variables + ['intercept']].values, i) for i in range(len(variables))]

        print(vif_data)
