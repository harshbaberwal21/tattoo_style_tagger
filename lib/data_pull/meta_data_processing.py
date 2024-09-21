"""Functions to process the tattoos' meta data."""


def process_tattoo_meta_data_to_long_form(tattoo_images_meta_data):
    """Process tattoos' meta data and return a dataframe.

    Args:
        tattoo_images_meta_data (pd.core.frame.DataFrame): Tattoos' meta data pulled
        from web.

    Returns:
        pd.core.frame.DataFrame: Tattoos' meta data with associated styles
        exploded to a long form.
    """
    tattoo_images_meta_data = tattoo_images_meta_data.copy()

    if "searched_style_query" in tattoo_images_meta_data:
        tattoo_images_meta_data.drop(columns="searched_style_query", inplace=True)

    tattoo_images_meta_data.drop_duplicates(inplace=True)

    tattoo_images_meta_data["styles"] = (
        tattoo_images_meta_data["styles"]
        .str.replace("[", "")
        .str.replace("]", "")
        .str.split(",")
    )

    tattoo_meta_data_exploded = tattoo_images_meta_data.explode(column="styles")
    tattoo_meta_data_exploded["styles"] = tattoo_meta_data_exploded[
        "styles"
    ].str.strip()
    tattoo_meta_data_exploded["styles"] = tattoo_meta_data_exploded[
        "styles"
    ].str.replace("'", "")

    tattoo_meta_data_exploded = tattoo_meta_data_exploded.loc[
        ~tattoo_meta_data_exploded["styles"].isin([""]), :
    ].copy()

    return tattoo_meta_data_exploded


def filter_out_tattoos(tattoo_meta_data_exploded, style_labels_count_threshold=7):
    """Filter out tattoos with style label counts
    greater than threshold. The default threshold is 7,
    meaning all images are tagged with at most 7 distinct styles.

    Args:
        tattoo_images_meta_data_exploded (pd.core.frame.DataFrame): Tattoos meta data
        with associated styles exploded to long form.
        style_count_threshold (int): Maximum allowed count of distinct style labels
        associated with an image.

    Returns:
        pd.core.frame.DataFrame: Tattoos' meta data with associated styles
        exploded to a long form.
    """
    tattoo_meta_data_exploded = tattoo_meta_data_exploded.copy()

    tattoo_images_style_counts = tattoo_meta_data_exploded.groupby(
        "image_url", as_index=False
    )["styles"].nunique()

    tattoo_urls_to_consider = tattoo_images_style_counts[
        tattoo_images_style_counts["styles"] <= style_labels_count_threshold
    ]["image_url"].values

    tattoo_meta_data_exploded_filtered = (
        tattoo_meta_data_exploded[
            tattoo_meta_data_exploded["image_url"].isin(tattoo_urls_to_consider)
        ]
        .reset_index(drop=True)
        .copy()
    )

    return tattoo_meta_data_exploded_filtered


def handle_duplicates(tattoo_meta_data_exploded_filtered):
    """Handle duplicates viz. tattoo ids with same image URL by merging
    their style labels and resetting the tattoo_id, combining them into one.

    Args:
        tattoo_meta_data_exploded_filtered (_type_): _description_

    Returns:
        pd.core.frame.DataFrame: Tattoos' meta data without any duplicates.
    """
    tattoo_meta_data_exploded_filtered = tattoo_meta_data_exploded_filtered.copy()

    # tattoo_meta_data_exploded_filtered.drop(columns='tattoo_id', inplace=True)

    new_tattoo_ids = (
        tattoo_meta_data_exploded_filtered[["image_url"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .reset_index()
    )
    new_tattoo_ids.rename(columns={"index": "tattoo_id"}, inplace=True)
    new_tattoo_ids["tattoo_id"] = new_tattoo_ids["tattoo_id"] + 1

    tattoo_meta_data_exploded_filtered.rename(
        columns={"tattoo_id": "orig_tattoo_id"}, inplace=True
    )

    tattoo_meta_data_exploded_filtered_final = tattoo_meta_data_exploded_filtered.merge(
        new_tattoo_ids, on="image_url", how="left"
    )

    # Null Check
    print(tattoo_meta_data_exploded_filtered_final.isna().sum())

    return tattoo_meta_data_exploded_filtered_final


if __name__ == "__main__":
    pass
