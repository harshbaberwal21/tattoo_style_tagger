"""Driver module containing functions for python callables in DAG"""

import os
import pandas as pd

from ..data_pull.meta_data_pull import read_tattoos_meta_data
from ..data_pull.meta_data_processing import (
    process_tattoo_meta_data_to_long_form,
    filter_out_tattoos,
    handle_duplicates,
)
from ..data_pull.download_tattoo_images import download_tattoo_images
from ..data_preprocessing.data_pre_processing import (
    get_dims_for_all_images,
    flag_images_with_dims_outliers,
    filter_out_less_represented_styles,
)
from ..data_preprocessing.data_augmentation import (
    prep_metadata_w_flipped_image_ids,
    flip_images_to_augment_data,
)
from ..temp_constants import(
TATTOO_DATA_PATH,
TATTOO_META_DATA_PATH,
TATTOO_IMAGES_PATH,
)

def read_tattoos_meta_data_driver():
    """Driver function for reading tattoo meta data.
    """
    style_querying_info = pd.read_csv(f"{TATTOO_DATA_PATH}/style_queries_w_description.csv")

    tattoos_meta_data = read_tattoos_meta_data(style_querying_info)

    tattoos_meta_data.to_csv(f'{TATTOO_META_DATA_PATH}/tattoos_meta_data.csv', index=False)


def process_tattoos_meta_data_driver():
    """Driver function for processing tattoo meta data.
    """
    tattoos_meta_data = pd.read_csv(f'{TATTOO_META_DATA_PATH}/tattoos_meta_data.csv')

    tattoos_meta_data_long = process_tattoo_meta_data_to_long_form(tattoos_meta_data)
    tattoos_meta_data_long_filtered = filter_out_tattoos(tattoos_meta_data_long)
    tattoos_meta_data_processed = handle_duplicates(tattoos_meta_data_long_filtered)

    tattoos_meta_data_processed.to_csv(
        f'{TATTOO_META_DATA_PATH}/tattoos_meta_data_processed.csv',
        index=False
        )


def download_tattoos_images_driver():
    """Driver function for downloading images.
    """
    tattoos_meta_data_processed = pd.read_csv(
        f'{TATTOO_META_DATA_PATH}/tattoos_meta_data_processed.csv'
        )

    download_tattoo_images(tattoos_meta_data_processed, TATTOO_IMAGES_PATH)


def pre_process_tattoos_images_driver():
    """Driver function for pre-processing tattoo iimages and
    update meta data accordingly.
    """
    tattoos_meta_data_processed = pd.read_csv(
        f'{TATTOO_META_DATA_PATH}/tattoos_meta_data_processed.csv'
        )

    tattoo_images_path = os.path.join(TATTOO_IMAGES_PATH, "raw_tattoo_images")
    tattoos_dims_df, _ = get_dims_for_all_images(tattoos_meta_data_processed, tattoo_images_path)

    tattoos_meta_data_processed_w_filter = flag_images_with_dims_outliers(
        tattoos_meta_data_processed,
        tattoos_dims_df
        )
    tattoos_meta_data_processed_final = filter_out_less_represented_styles(
        tattoos_meta_data_processed_w_filter
        )
    tattoos_meta_data_processed_final.to_csv(
        f'{TATTOO_META_DATA_PATH}/tattoos_meta_data_processed_final.csv',
        index=False
        )


def augment_tattoos_images_data_driver():
    """Driver function for augmenting the images.
    """
    tattoos_meta_data_processed_final = pd.read_csv(
        f'{TATTOO_META_DATA_PATH}/tattoos_meta_data_processed_final.csv'
        )

    tattoos_meta_data_processed_final_augmented = prep_metadata_w_flipped_image_ids(
        tattoos_meta_data_processed_final
        )

    tattoo_images_path = os.path.join(TATTOO_IMAGES_PATH, "raw_tattoo_images")

    flip_images_to_augment_data(tattoos_meta_data_processed_final_augmented, tattoo_images_path)

    tattoos_meta_data_processed_final_augmented.to_csv(
        f'{TATTOO_META_DATA_PATH}/tattoos_meta_data_processed_final_augmented.csv',
        index=False
        )
