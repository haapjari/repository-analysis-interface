import os
from dotenv import load_dotenv


def get(key):
    """
    Get the value of the environment variable with the given key.
    """
    env_key = key.upper()

    return os.getenv(env_key)


class Config:
    def __init__(self):
        load_dotenv()
