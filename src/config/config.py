import os
from dotenv import load_dotenv


def get(key):
    env_key = key.upper()

    return os.getenv(env_key)


class Config:
    def __init__(self):
        load_dotenv()
