import pandas as pd
import os

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
DATA_PATH = ""

def tattoos_meta_data_reading_driver(**args):
    
    style_querying_info = pd.read_csv(f"{DATA_PATH}/style_queries_w_description.csv")
    
    tattoos_meta_data = read_tattoos_meta_data(style_querying_info)

    tattoos_meta_data.to_csv(f'{DATA_PATH}/tattoos_meta_data.csv', index=False)


def tattoos_meta_data_processing_driver(**args):

    tattoos_meta_data = pd.read_csv(f'{DATA_PATH}/tattoos_meta_data.csv')

    tattoos_meta_data_long = process_tattoo_meta_data_to_long_form(tattoos_meta_data)
    tattoos_meta_data_long_filtered = filter_out_tattoos(tattoos_meta_data_long)
    tattoos_meta_data_processed = handle_duplicates(tattoos_meta_data_long_filtered)

    tattoos_meta_data_processed.to_csv(f'{DATA_PATH}/tattoos_meta_data_processed.csv', index=False)


def tattoos_image_downloading_driver(**args):

    tattoos_meta_data_processed = pd.read_csv(f'{DATA_PATH}/tattoos_meta_data_processed.csv')

    download_tattoo_images(tattoos_meta_data_processed, DATA_PATH)


def tattoos_image_pre_processing_driver(**args):

    tattoos_meta_data_processed = pd.read_csv(f'{DATA_PATH}/tattoos_meta_data_processed.csv')
    
    tattoo_images_path = os.path.join(DATA_PATH, "raw_tattoo_images")
    tattoos_dims_df, _ = get_dims_for_all_images(tattoos_meta_data_processed, tattoo_images_path)
    
    tattoo_meta_data_processed_w_filter = flag_images_with_dims_outliers(
        tattoos_meta_data_processed,
        tattoos_dims_df
        )
    tattoo_meta_data_processed_final = filter_out_less_represented_styles(
        tattoo_meta_data_processed_w_filter
        )
    tattoo_meta_data_processed_final.to_csv(f'{DATA_PATH}/tattoo_meta_data_processed_final.csv', index=False)


def tattoos_image_augmenting_driver(**args):
    tattoo_meta_data_processed_final = pd.read_csv(f'{DATA_PATH}/tattoo_meta_data_processed_final.csv')

    tattoo_meta_data_processed_final_augmented = prep_metadata_w_flipped_image_ids(tattoo_meta_data_processed_final)

    tattoo_images_path = os.path.join(DATA_PATH, "raw_tattoo_images")

    flip_images_to_augment_data(tattoo_meta_data_processed_final_augmented, tattoo_images_path)

    tattoo_meta_data_processed_final_augmented.to_csv(
        f'{DATA_PATH}/tattoo_meta_data_processed_final_augmented.csv',
        index=False
        )
