"""Module containing functions for defining dataset and loading data."""

import os
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


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
        self.labels_df = labels_df
        self.root_dir = root_dir
        self.transform = transform
        self.ids_df = id_df
        self.label_index_map = label_index_map
        self.len = len(self.ids_df)
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        
        tattoo_id = self.ids_df.loc[idx,'tattoo_id']
        
        img_name = os.path.join(self.root_dir, f"{tattoo_id}.jpg")
        image = Image.open(img_name)

        tattoo_style_positions = self.labels_df[self.labels_df['tattoo_id']==tattoo_id]['styles'].map(
            lambda x: self.label_index_map[x]
        ).values

        labels = torch.zeros(len(self.label_index_map), dtype=torch.float32)
        labels[tattoo_style_positions] = 1

        
        if self.transform:
            transformed_image_tansor = self.transform(image)
            return transformed_image_tansor, labels

        return image, labels


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
    padded_image_tensor = F.pad(image_tensor, (padding_left, padding_right, padding_top, padding_bottom), "constant", 0)
    
    return padded_image_tensor


def get_image_tranforms(target_image_dims = (256, 256)):
    """Get data transforms to be applied to an image.

    Args:
        target_image_dims (tuple, optional): Target image dimensions. Defaults to (256, 256).

    Returns:
        data_transoform: transformer for image tensor.
    """
    image_tranforms = transforms.Compose([
        transforms.ToTensor(),
        pad_to_square_transform,
        transforms.Resize(target_image_dims),
    ])

    return image_tranforms

def get_data_loader(dataset, batch_size):
    """Get data loader.

    Args:
        dataset (TattooDataset): An instance of TattooDataset
        batch_size (int): batch size for model training.

    Returns:
        _type_: _description_
    """

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return  data_loader
