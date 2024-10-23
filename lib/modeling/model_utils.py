"""Module containing utility functions for training and evaluation."""

import torch
from torch import nn


DATA_DIR = "/Users/harshbaberwal/Desktop/Projects/git_repos/tattoo_style_tagger/data/"

IMAGE_DIR = DATA_DIR + "tattoo_images/raw_tattoo_images/"

MODEL_DIR = DATA_DIR + "model_artifacts/"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_label_index_map(tattoos_meta_data):
    """Get the label string to index mapping for model training.

    Args:
        tattoos_meta_data (pandas.core.DataFrame): Tattoos metadata processed and augmented.

    Returns:
        dict: {style_name : style_index)
    """
    temp = (
        tattoos_meta_data[["styles"]]
        .drop_duplicates()
        .sort_values("styles")
        .reset_index(drop=True)
        .to_dict()["styles"]
    )
    label_index_map = {v: k for k, v in temp.items()}
    return label_index_map


def get_tattoo_ids(tattoos_meta_data, example_type):
    """Get all the tattoo_ids.

    Args:
        tattoos_meta_data (pandas.core.DataFrame): Tattoos metadata processed and augmented.
        example_type (str): indicates the IDs needed, train, test or val.

    Returns:
        pandas.core.DataFrame: tattoo_id dataframe
    """
    tattoo_ids = (
        tattoos_meta_data.loc[
            tattoos_meta_data["example_type"] == example_type, ["tattoo_id"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return tattoo_ids


def get_tattoo_labels(tattoos_meta_data, example_type):
    """Get the labels by tattoo_id in long format.

    Args:
        tattoos_meta_data (pandas.core.DataFrame): Tattoos metadata processed and augmented.
        example_type (str): indicates the IDs needed, train, test or val.

    Returns:
        pandas.core.DataFrame: dataframe at tattoo_id x styles (labels) level.
    """
    labels_df = (
        tattoos_meta_data.loc[
            tattoos_meta_data["example_type"] == example_type, ["tattoo_id", "styles"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return labels_df


def get_predicted_labels(predicted_logits, threshold=0.5):
    """Get predicted labels index from logits.

    Args:
        predicted_logits (torch.tensor): Predicted logits
        threshold (float, optional): Threshold for positive classification. Defaults to 0.5.

    Returns:
        torch.tensor: predicted labels indices
    """
    sigmoid = nn.Sigmoid()
    predicted_prob = sigmoid(predicted_logits)
    predicted_labels_indices = torch.where(predicted_prob > threshold, 1, 0)
    return predicted_labels_indices


def save_model(model, filepath):
    """Save the model parameters.

    Args:
        model (nn.module): the model to save
        filepath (str): the file path and name to save it with. Should have the extension .pt.
    """
    torch.save(model.state_dict(), f=filepath)
