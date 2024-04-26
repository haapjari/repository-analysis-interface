import logging as log
from datetime import datetime, timedelta
from src.config.config import Config, get


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
        # TODO
        start_date = datetime.strptime(self.first_date, "%Y-%m-%d")
        end_date = datetime.strptime(self.last_date, "%Y-%m-%d")
        delta = end_date - start_date
        search_api_host = get("SEARCH_API_HOST")
        database_api_host = get("DATABASE_API_HOST")

        query = 0

        if delta.days > 7:
            current_date = start_date
            while current_date < end_date:
                week_end = min(current_date + timedelta(days=7), end_date)

                print(query)
                print(current_date.date())
                print(week_end.date())
                print(self.language)
                print(self.min_stars)
                print(self.max_stars)
                print(self.order)
                print("")

                print(f"Request {search_api_host} :: {current_date.date()} to {week_end.date()}")
                print(f"Storing to {database_api_host}")

                current_date += timedelta(days=7 + 1)
                query += 1
        else:
            # TODO Single Query

            print(start_date.date())
            print(end_date.date())
            print(self.language)
            print(self.min_stars)
            print(self.max_stars)
            print(self.order)


    def normalize(self):
        # TODO
        pass

    def analyze(self):
        # TODO
        pass
