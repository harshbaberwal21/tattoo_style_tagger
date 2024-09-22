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

    meta_data_for_styles: list = []

    for style_row in styles[["style", "style_query"]].iterrows():
        style_name: str = style_row[1]["style"]
        style_query: str = style_row[1]["style_query"]

        print(f"Reading style {style_name}")

        meta_data_for_style = read_meta_data_for_single_style(style_query)
        meta_data_for_styles.append(meta_data_for_style)

        print(f"Done reading meta data for style {style_name} ")

        time.sleep(5)

    meta_data_for_all_styles = pd.concat(meta_data_for_styles)

    return meta_data_for_all_styles.reset_index(drop=True)


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
    meta_data_for_style: list = []

    for page in range(1, max_pages + 1):

        api_url: str = base_url + f"style={style_query}&page={page}&limit={page_limit}"
        try:
            with requests.get(api_url, timeout=5) as response:
                response_text = response.text
        except requests.exceptions.Timeout:
            print(f"{page} of {style_query} Timed out. Moving to next page.")
            continue
        response_dict: dict = json.loads(response_text)

        if "error" in response_dict.keys() or not response_dict["data"]:
            break

        meta_data_from_page = read_meta_data_from_page(
            response_dict["data"]
        )
        meta_data_for_style.append(meta_data_from_page)

        time.sleep(3)

    all_meta_data_for_style = pd.concat(meta_data_for_style)
    all_meta_data_for_style["searched_style_query"] = style_query

    return all_meta_data_for_style


def read_meta_data_from_page(response_dict_data: dict):
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
        record["styles"] = str(tattoo_data["classification"]["styles"])

        tattoos_records.append(record)

    meta_data_from_page = pd.DataFrame(tattoos_records)
    return meta_data_from_page


if __name__ == "__main__":
    pass
