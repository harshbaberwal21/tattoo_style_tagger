"""Module containing functions for defining dataset and loading data."""

# pylint: disable=too-many-arguments, too-many-positional-arguments

import os
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .model_utils import (
    get_tattoo_ids,
    get_tattoo_labels,
    IMAGE_DIR,
)


class TattooDataset(Dataset):
    """Tattoo images dataset."""

    def __init__(self, id_df, labels_df, label_index_map, root_dir, transform=None):
        """
        Arguments:
            labels_df (string): Path to the csv with style labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_df = labels_df.copy()
        self.root_dir = root_dir
        self.transform = transform
        self.ids_df = id_df.copy()
        self.label_index_map = label_index_map
        self.len = len(self.ids_df)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        tattoo_id = self.ids_df.loc[idx, "tattoo_id"]

        img_name = os.path.join(self.root_dir, f"{tattoo_id}.jpg")
        image = Image.open(img_name)

        tattoo_style_positions = (
            self.labels_df[self.labels_df["tattoo_id"] == tattoo_id]["styles"]
            .map(lambda x: self.label_index_map[x])
            .values
        )

        labels = torch.zeros(1, len(self.label_index_map), dtype=torch.float32)
        labels[0, tattoo_style_positions] = 1

        if self.transform:
            transformed_image_tansor = self.transform(image)
            return transformed_image_tansor, labels

        return image, labels


class PrecomputedActivationsDataset(Dataset):
    """Dataset class with precomuted activations.

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, precomputed_activations, labels):
        self.activations = precomputed_activations
        self.labels = labels
        self.len = len(self.activations)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.activations[idx], self.labels[idx]


def pad_to_square_transform(image_tensor):
    """Pad an image to make it a square image. This is done to avoid
    image distortion when the image is resized to fixed lower dimension.

    Args:
        image_tensor (tensor): Tensor of the image to be padded.

    Returns:
        tensor: padded image tensor.
    """

    # Calculate padding
    original_height, original_width = image_tensor.shape[1], image_tensor.shape[2]
    target_height = target_width = max(original_height, original_width)

    padding_left = (target_width - original_width) // 2
    padding_top = (target_height - original_height) // 2
    padding_right = target_width - original_width - padding_left
    padding_bottom = target_height - original_height - padding_top

    # Apply padding
    padded_image_tensor = F.pad(
        image_tensor,
        (padding_left, padding_right, padding_top, padding_bottom),
        "constant",
        0,
    )

    return padded_image_tensor


def get_image_tranforms(target_image_dims=(256, 256)):
    """Get data transforms to be applied to an image.

    Args:
        target_image_dims (tuple, optional): Target image dimensions. Defaults to (256, 256).

    Returns:
        data_transoform: transformer for image tensor.
    """
    image_tranforms = transforms.Compose(
        [
            transforms.ToTensor(),
            pad_to_square_transform,
            transforms.Resize(target_image_dims),
        ]
    )

    return image_tranforms


def get_precomputed_data_loader(
    model, tattoos_meta_data, label_index_map, batch_size, device, example_type="train"
):
    """Get data loader.

    Args:
        dataset (TattooDataset): An instance of TattooDataset
        batch_size (int): batch size for model training.

    Returns:
        _type_: _description_
    """

    image_transforms = get_image_tranforms()

    tattoo_ids = get_tattoo_ids(tattoos_meta_data, example_type)
    tattoo_labels = get_tattoo_labels(tattoos_meta_data, example_type)
    dataset = TattooDataset(
        tattoo_ids,
        tattoo_labels,
        label_index_map,
        IMAGE_DIR,
        transform=image_transforms,
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    precomputed_activations, labels = precompute_frozen_network_activations(
        model.frozen, data_loader, device
    )

    precomputed_dataset = PrecomputedActivationsDataset(precomputed_activations, labels)
    precomputed_data_loader = DataLoader(
        precomputed_dataset, batch_size=batch_size, shuffle=True
    )

    return precomputed_data_loader


def precompute_frozen_network_activations(frozen_layers, tattoo_data_loader, device):
    """Pass data through fixed/frozen network and cache activations for faster training.

    Args:
        frozen_layers (pytorch.nn.module): the model portion that is frozen i.e.
        has the requires_grad attribute of the parameters set to False.
        tattoo_data_loader (torch.utils.data.DataLoader): DataLoader object for
        the data to pass through
        device (str): The device available for processing (cpu, gpu or mps).

    Returns:
       dict of torch.Tensor: cached activations
    """
    frozen_layers.eval()
    precomputed_activations = []
    labels_list = []
    with torch.no_grad():
        for train_features, labels in tattoo_data_loader:
            train_features, labels = train_features.to(device), labels.to(device)
            activations = frozen_layers(train_features)
            precomputed_activations.append(torch.flatten(activations, 1))
            labels_list.append(torch.flatten(labels, 0, 1))

    labels = torch.cat(labels_list)
    precomputed_activations = torch.cat(precomputed_activations)
    # torch.save(precomputed_activations, 'precomputed_train_activations.pt')

    return precomputed_activations, labels
