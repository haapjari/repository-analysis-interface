import numpy as np
import pandas as pd
import logging as log
from sklearn.preprocessing import MinMaxScaler


def normalize_numeric(data_dict):
    """
    Normalize Numeric Values in a Dictionary.
    """

    # Keys.
    keys = list(data_dict.keys())

    # Convert to Integers (Including Strings)
    values = [int(value) if isinstance(value, str) else value for value in data_dict.values()]

    # Transform Values into Numpy Array.
    values_array = np.array(values).reshape(-1, 1)

    scaler = MinMaxScaler()
    normalized_values = scaler.fit_transform(values_array)

    # Convert to Dictionary.
    normalized_dict = dict(zip(keys, normalized_values.flatten()))

    return normalized_dict


def normalize_dates(data_dict):
    """
    Normalize Dates in a Dictionary. Handles Out of Bounds Datetime errors and defaults to 0.0 if necessary.
    """

    keys = list(data_dict.keys())
    timestamps = []

    # Convert to UNIX Timestamps.
    for i, date in enumerate(data_dict.values()):
        try:
            timestamp = pd.to_datetime(date).timestamp()
            timestamps.append(timestamp)
        except Exception as e:
            log.error(f"Error: {e} for date '{date}' at position {i}")
            timestamps.append(None)

    valid_indices = [i for i, ts in enumerate(timestamps) if ts is not None]
    valid_keys = [keys[i] for i in valid_indices]
    valid_timestamps = [timestamps[i] for i in valid_indices]

    if valid_timestamps:
        timestamps_array = np.array(valid_timestamps).reshape(-1, 1)
        scaler = MinMaxScaler()
        normalized_timestamps = scaler.fit_transform(timestamps_array).flatten()

        normalized_dict = {key: norm_ts for key, norm_ts in zip(valid_keys, normalized_timestamps)}
    else:
        normalized_dict = {}

    for key in keys:
        if key not in normalized_dict:
            normalized_dict[key] = 0.0

    return normalized_dict
