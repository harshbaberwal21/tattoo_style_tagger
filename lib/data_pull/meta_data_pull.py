"""Functions to read the meta data of Tattoos"""

import time
import json
import requests
import pandas as pd


def read_tattoos_meta_data(styles: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """Hit the API and read the meta data of all the tattoos.

    Args:
        styles (pd.core.frame.DataFrame): The data with the style name and
        relevant queryies for the API

    Returns:
        pd.core.frame.DataFrame: Tattoos meta data with associated styles
        and image URL.
    """

    styles_meta_data: list = []

    for style_row in styles[["style", "style_query"]].iterrows():
        style_name: str = style_row[1]["style"]
        style_query: str = style_row[1]["style_query"]

        print(f"Reading style {style_name}")

        style_meta_data = read_meta_data_for_single_style(style_query)
        styles_meta_data.append(style_meta_data)

        print(f"Done reading meta data for style {style_name} ")

    all_styles_meta_data = pd.concat(styles_meta_data)
    time.sleep(5)

    return all_styles_meta_data


def read_meta_data_for_single_style(style_query, max_pull_count=10000):
    """Hit the API and read the meta data of all the tattoos
    for a particular style.

    Args:
        style_query (str): corresponding style query string for the API.
        max_pull_count (int): Maximum count of relevant images to read
        meta data for.

    Returns:
        pd.core.frame.DataFrame: Tattoos meta data with associated styles
        and image URL for input style query.
    """
    page_limit: int = 100
    max_pages: int = int(max_pull_count / page_limit)
    base_url: str = "https://backend-api.tattoodo.com/api/v2/search/posts?"
    tattoo_meta_data_dfs: list = []

    for page in range(1, max_pages + 1):

        api_url: str = base_url + f"style={style_query}&page={page}&limit={page_limit}"
        try:
            response = requests.get(api_url, timeout=5)
        except requests.exceptions.Timeout:
            print(f"{page} of {style_query} Timed out. Moving to next page.")
            continue
        response_dict: dict = json.loads(response.text)

        if "error" in response_dict.keys() or not response_dict["data"]:
            break

        tattoos_meta_data_from_page = read_meta_data_from_one_page(
            response_dict["data"]
        )
        tattoo_meta_data_dfs.append(tattoos_meta_data_from_page)

        time.sleep(3)

    style_tattoo_meta_data: pd.core.frame.DataFrame = pd.concat(tattoo_meta_data_dfs)
    style_tattoo_meta_data["searched_style_query"] = style_query

    return style_tattoo_meta_data


def read_meta_data_from_one_page(response_dict_data: dict):
    """Read tattoo's meta data from a single page.

    Args:
        response_dict_data (dict): API response as loaded into dictionary

    Returns:
        pd.core.frame.DataFrame: _description_
    """
    tattoos_records: list = []
    for tattoo_data in response_dict_data:
        record: dict = {}
        record["tattoo_id"] = tattoo_data["id"]
        record["image_url"] = tattoo_data["image"]["url"]
        record["styles"] = tattoo_data["classification"]["styles"]

        tattoos_records.append(record)

    tattoo_meta_data = pd.DataFrame(tattoos_records)
    return tattoo_meta_data


if __name__ == "__main__":
    pass
