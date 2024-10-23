"""Utility Functions"""
import pandas as pd

def read_csv_data(path, **kwargs):
    """Read data from a CSV file

    Args:
        path (str): path to file

    Returns:
        pd.core.frame.DataFrame: data loaded from csv
    """
    return pd.read_csv(path, **kwargs)
