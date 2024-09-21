"""Module with functions for tattoo image pre-processing."""

from PIL import Image
import numpy as np
import pandas as pd


def flag_images_with_dims_outliers(tattoo_meta_data_processed, tattoos_dims_df):
    """Tag tattoos from processed metadata with outlier dimensions viz.
    height and width, as not to process by creating a new flag column, 'to_process'.

    Args:
        tattoo_images_meta_data (pd.core.frame.DataFrame): Processed tattoo metadata
        tattoos_size_df (pd.core.frame.DataFrame): Data with height and width of each
        image in the metadata.

    Returns:
        pd.core.frame.DataFrame: Tattoo metadata with a to_process flag set to True
        for viable images.
    """

    tattoos_dims_df_filtererd = tattoos_dims_df[
        (tattoos_dims_df["height"] <= tattoos_dims_df["height"].quantile(0.99))
        & (tattoos_dims_df["height"] > tattoos_dims_df["height"].quantile(0.005))
        & (tattoos_dims_df["width"] <= tattoos_dims_df["height"].quantile(0.99))
        & (tattoos_dims_df["width"] > tattoos_dims_df["height"].quantile(0.005))
    ]

    tattoo_meta_data_processed_w_filter = tattoo_meta_data_processed.merge(
        tattoos_dims_df_filtererd[["tattoo_id"]],
        on="tattoo_id",
        how="left",
        indicator=True,
    )

    tattoo_meta_data_processed_w_filter["to_process"] = np.where(
        tattoo_meta_data_processed_w_filter["_merge"] == "left_only", False, True
    )

    tattoo_meta_data_processed_w_filter.drop(columns="_merge", inplace=True)

    return tattoo_meta_data_processed_w_filter


def filter_out_less_represented_styles(tattoo_meta_data, tattoo_img_count_thresh=50):
    """Filter out styles that do not have enough images associated with them.

    Args:
        tattoo_meta_data (pd.core.frame.DataFrame): Processed tattoo metadata.
        tattoo_img_count_thresh (int, optional): Minimum image count required per style.
        Defaults to 50.

    Returns:
        pd.core.frame.DataFrame: tattoo metadata with less represented styles removed.
    """
    tattoo_image_counts_by_style = tattoo_meta_data.groupby("styles", as_index=False)[
        "tattoo_id"
    ].nunique()

    styles_to_drop = tattoo_image_counts_by_style[
        tattoo_image_counts_by_style["tattoo_id"] < tattoo_img_count_thresh
    ]["styles"].values

    tattoo_meta_data = tattoo_meta_data[
        ~tattoo_meta_data["styles"].isin(styles_to_drop)
    ].reset_index(drop=True)

    return tattoo_meta_data


# pylint: disable=bare-except
def get_dims_for_all_images(tattoo_meta_data, raw_tattoo_images_path):
    """Read in and fetch the dimensions for all the downloaded images
    specified in tattoo metadata.

    Args:
        tattoo_meta_data (pd.core.frame.DataFrame): Processed tattoo metadata.
        raw_tattoo_images_path (str): Path of the dir with raw tattoo images.

    Returns:
        pd.core.frame.DataFrame: Data with tattoo_id and the respective dimensions
        viz. height and width.
    """
    img_dim = []
    ids_not_read = []
    tattoo_urls_w_ids = tattoo_meta_data[["tattoo_id", "image_url"]].drop_duplicates()
    for i in tattoo_urls_w_ids["tattoo_id"].values:
        try:
            image_path = f"{raw_tattoo_images_path}/{i}.jpg"
            width, height = get_image_dims(image_path)
            temp = {}
            temp["tattoo_id"], temp["height"], temp["width"] = i, height, width
            img_dim.append(temp)
        except:
            ids_not_read.append(i)
    tattoos_dim_df = pd.DataFrame(img_dim)
    return tattoos_dim_df, ids_not_read


# pylint: enable=bare-except


def get_image_dims(image_path):
    """Read image and get its height and width.

    Args:
        image_path (str): Complete path to image.

    Returns:
        tuple: height and width of the image at input image_path
    """
    with Image.open(image_path) as img:
        return img.size


if __name__ == "__main__":
    pass
