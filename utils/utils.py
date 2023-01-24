import csv
import chardet
import pandas as pd
import sys
from datetime import datetime
import io
import numpy as np
import github

# Read a .csv file and return a string.
def read_csv_file(file_path):
    """
    Read a CSV file and return its contents as a string.

    Args:
        file_path (str): The path to the CSV file to read.

    Returns:
        str: The contents of the CSV file as a string, where each row is separated
            by a newline character and each value within a row is separated by a comma.
    """
    data = ""
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read())
        file.seek(0)
        csv_reader = csv.reader(file.read().decode(result['encoding']).splitlines())
        for row in csv_reader:
            data += ",".join(row) + "\n"

    return data

def extract_columns(dataset_file: str, columns: list, result_type):
    if result_type == "tuple":
        df = pd.read_csv(io.StringIO(dataset_file))

        extracted_columns = {}
    
        for column in columns:
            extracted_columns[column] = list(zip(df['repository_url'], df[column]))

        return extracted_columns

    if result_type == "normal":
        df = pd.read_csv(io.StringIO(dataset_file))

        extracted_columns = {}
    
        for column in columns:
            extracted_columns[column] = df[column]

        return extracted_columns

def print_columns(dataset_columns):
    """
    Print the names of the columns in a dataset.

    Args:
        dataset_columns (dict): A dictionary with column names as keys and lists of tuples as values.
            Each tuple contains a unique identifier for the row and the corresponding value for that
            column in that row.
    """
    for column, values in dataset_columns.items():
        print(column)


def print_values(dataset_columns, column_arg):
    """
    Print the values in the specified column of a dataset.

    Args:
        dataset_columns (dict): A dictionary with column names as keys and lists of tuples as values.
            Each tuple contains a unique identifier for the row and the corresponding value for that
            column in that row.
        column_arg (str): The name of the column to print the values of.
    """
    for column, values in dataset_columns.items():
        if column == column_arg:
            for value in values:
                print(f'{value[0]} : {value[1]}')

def convert_to_unix_timestamp(date_string):
    """
    Convert a date string in the format "%Y-%m-%dT%H:%M:%SZ" to a Unix timestamp.

    Args:
        date_string (str): A string representing a date in the format "%Y-%m-%dT%H:%M:%SZ".

    Returns:
        float: A Unix timestamp representing the same date and time as the input string.
    """
    date_format = "%Y-%m-%dT%H:%M:%SZ"

    datetime_object = datetime.strptime(date_string, date_format) 
    unix_timestamp = datetime_object.timestamp()
    
    return unix_timestamp


def convert_dict_to_dataframe(dataset):   
    for key in dataset.keys():
        dataset[key] = np.reshape(dataset[key], -1)

    df = pd.DataFrame.from_dict(dataset)

    return df 


def convert_dataframe_to_dict(df):
    dataset = df.to_dict()

    for key in dataset.keys():
        if isinstance(dataset[key], np.ndarray):
            dataset[key] = np.reshape(dataset[key], -1)

    return dataset


def create_dictionary(dataset, columns):
    dictionary = {}
    for column in columns:
        dictionary[column] = []
        if column in dataset:
            for value in dataset[column].values():
                dictionary[column].append(value)
    return dictionary


def process_csv_file(input_file_path, output_file_path, gh):
    try:
        with open(input_file_path, mode='r', newline='', encoding='utf-8') as file:
            # Create a csv reader object
            csv_reader = csv.reader(file)
            
            header = next(csv_reader)
            header.append("avg_weekly_additions")

            # List to hold the updated rows
            updated_rows = []

            for row in csv_reader:
                try:
                    parts = row[1].split("/")
                    name = parts[-1]
                    owner = parts[-2]

                    owner = gh.get_repo(owner, name)

                    avg_weekly_additions, avg_weekly_deletions = gh.get_avg_weekly_additions(owner)

                    # add a column value to the row
                    row.append(avg_weekly_additions)

                    updated_rows.append(row)

                    print(f"Processing {owner.full_name}...")

                except (github.GithubException, IndexError) as e:
                    print(f"Error processing row {row}: {e}")

            # Open the output CSV file and create a writer object
            with open(output_file_path, 'w', newline='') as outfile:
                writer = csv.writer(outfile)
            
                # Write the header row to the output file
                writer.writerow(header)
            
                # Write the updated rows to the output file
                writer.writerows(updated_rows)

    except IOError as e:
        print(f"Error reading file: {e}")