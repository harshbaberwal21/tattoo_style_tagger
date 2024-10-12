"""Utility Functions"""
import pandas as pd
import torch

def read_csv_data(path, **kwargs):
    """Read data from a CSV file

    Args:
        path (str): path to file

    Returns:
        pd.core.frame.DataFrame: data loaded from csv
    """
    return pd.read_csv(path, **kwargs)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"