"""Module with functions to augment the data."""
from time import time

import numpy as np
import pandas as pd
from PIL import Image


def prep_metadata_w_flipped_image_ids(tattoo_meta_data_processed):
    """Prepare metadata for augmentation by creating a new id for each present
    image which is the negative counterpart of the original id and then concatenating
    the results together to build complete augmented metadata.

    Args:
        tattoo_meta_data_processed (pd.core.frame.DataFrame): Processed tattoo metadata.

    Returns:
        pd.core.frame.DataFrame: Metadata with new IDs and relevant styles.
    """
    tattoo_meta_data_processed_to_augment = tattoo_meta_data_processed[
        tattoo_meta_data_processed['to_process']
        ]

    tattoo_meta_data_processed_to_augment['tattoo_id_flipped'] = \
        -1 * tattoo_meta_data_processed_to_augment['tattoo_id']

    tattoo_meta_data_processed_augmented = pd.concat([
        tattoo_meta_data_processed_to_augment[['tattoo_id', 'styles']],
        tattoo_meta_data_processed_to_augment[
            ['tattoo_id_flipped', 'styles']
            ].rename(columns = {'tattoo_id_flipped':'tattoo_id'})
            ])
    return tattoo_meta_data_processed_augmented


def flip_images_to_augment_data(tattoo_meta_data_processed_augmented, tattoo_images_path):
    """Augment data by flipping the original images left to right and saving them
    with the new image id (-ve counterpart of original id) as filename.

    Args:
        tattoo_meta_data_processed_augmented (pd.core.frame.DataFrame): Processed
        tattoo metadata with new augmented IDs available

        downloaded_image_path (str): Path to the download raw tattoo images.
    """
    positive_tattoo_ids = tattoo_meta_data_processed_augmented[
        tattoo_meta_data_processed_augmented.tattoo_id>0
        ]['tattoo_id'].unique()

    i=0
    for tattoo_id in positive_tattoo_ids:

        img = Image.open(f"{tattoo_images_path}/{tattoo_id}.jpg")
        flipped_img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        if len(np.array(img).shape)==3:
            flipped_img.save(f"{tattoo_images_path}/-{tattoo_id}.jpg")
        elif len(np.array(img).shape)==2:
            img.convert('RGB').save(f"{tattoo_images_path}/{tattoo_id}.jpg")
            flipped_img.convert('RGB').save(f"{tattoo_images_path}/-{tattoo_id}.jpg")
        else:
            flipped_img.save(f"{tattoo_images_path}/-{tattoo_id}.jpg")
        i+=1
        if i%1000 == 0:
            print(f"done with {i} images")
            time.sleep(3)


if __name__ == "__main__":
    pass
