"""Module containing utility functions for training and evaluation."""
import torch
import torch.nn as nn
from .model_definition import get_model


def get_label_index_map(tattoos_meta_data):
    temp = tattoos_meta_data[['styles']].drop_duplicates().sort_values('styles').reset_index(drop=True).to_dict()['styles']
    label_index_map = {v:k for k,v in temp.items()}
    return label_index_map


def get_tattoo_ids(tattoos_meta_data, example_type):
    tattoo_ids = tattoos_meta_data.loc[
    tattoos_meta_data['example_type'] == example_type,
    ['tattoo_id']
    ].drop_duplicates().reset_index(drop=True)
    return tattoo_ids


def get_labels(tattoos_meta_data, example_type):
    labels_df = tattoos_meta_data.loc[
    tattoos_meta_data['example_type'] == example_type,
    ['tattoo_id', 'styles']
    ].drop_duplicates().reset_index(drop=True)
    return labels_df


IMAGE_DIR = '/Users/harshbaberwal/Desktop/Projects/git_repos/tattoo_style_tagger/data/tattoo_images/raw_tattoo_images'


MODEL_DIR = '/Users/harshbaberwal/Desktop/Projects/git_repos/tattoo_style_tagger/data/tattoo_images/model_artifacts'


def get_predicted_labels(predicted_logits, threshold = 0.5):
    sigmoid = nn.Sigmoid()
    predicted_prob = sigmoid(predicted_logits)
    predicted_labels_indices = torch.where(predicted_prob > threshold, 1, 0)
    return predicted_labels_indices


def save_model(model, filepath):
    torch.save(model.state_dict(), f=filepath)


def load_model(out_labels_count, filepath):
    model = get_model(out_labels_count)
    model.load_state_dict(torch.load(f=filepath, weights_only=True))
    model.eval()
