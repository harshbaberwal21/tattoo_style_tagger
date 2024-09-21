"""Functions to download the tattoo images."""
import os
import time
import requests
import pandas as pd

def download_tattoo_images(tattoo_meta_data_processed, download_path):
    """Download the tattoo images for tattoos in processed metadata using
    the associated image_url.

    Args:
        tattoo_meta_data_processed (pd.core.frame.DataFrame): Processed tattoos metadata
        download_path (str): Path of the directory to download images in.

    Raises:
        ValueError: If the provided download_path does not exist.
        e: _description_
    """

    if os.path.exists(download_path):
        download_folder_path = os.path.join(download_path, "raw_tattoo_images")
    else:
        raise ValueError("Given download path either does not exist or is invalid.\n\
            Please enter a valid path that exists.")

    if not os.path.exists(download_folder_path):
        os.mkdir(download_folder_path)

    download_counter = 0
    tattoo_urls_w_ids  = tattoo_meta_data_processed[['tattoo_id', 'image_url']].drop_duplicates()

    for tattoo_url_row in tattoo_urls_w_ids.sort_values('tattoo_id').iterrows():
        image_url = tattoo_url_row[1]['image_url']
        tattoo_id = tattoo_url_row[1]['tattoo_id']

        try:
            download_image(image_url, f'{download_folder_path}/{tattoo_id}')
            download_counter += 1
        except Exception as e:
            print('Last tattoo_id downloaded: ', tattoo_id)
            raise e

        if download_counter%10 == 0:
            time.sleep(2)

        if download_counter%100 == 0:
            time.sleep(5)

        if download_counter%1000 == 0:
            print(f'Downloaded {download_counter} images')
            time.sleep(10)

def download_image(image_url, download_path_w_name):
    """Download the image from the given URL.

    Args:
        image_url (str): Image URL
        download_path_w_name (str): Path of the directory to download
        in with filename included, except the extension.
    """
    img_data = requests.get(image_url).content
    with open(f'{download_path_w_name}.jpg', 'wb') as handler:
        handler.write(img_data)


if __name__ == "__main__":
    pass
