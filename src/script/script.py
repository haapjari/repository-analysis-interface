import requests

from src.config.config import get 

class Script:
    def __init__(self, config, table: str, column: str):
        self.config = config
        self.table = table
        self.column = column

    def drop(self):
        """
        Drop a Column from a Repository.
        """

        database_api_host = get("DATABASE_API_HOST")
        s = f"{database_api_host}/api/v1/{self.table}?column={self.column}"

        response = requests.delete(s)
        response.raise_for_status()
