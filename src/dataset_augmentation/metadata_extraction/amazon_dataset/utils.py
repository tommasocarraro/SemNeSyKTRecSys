import json

import pandas as pd
from loguru import logger


def object_recursive_delete_fields(json_object, fields_to_remove) -> None:
    """
    Recursively delete fields from a JSON object. Nested fields should adhere to structure:
    "field.subfield.subfield.subfield..."
    Args:
        json_object: object read from JSON file from which to remove the fields.
        fields_to_remove: list of fields to remove.
    """
    if isinstance(json_object, list):
        for item in json_object:
            object_recursive_delete_fields(item, fields_to_remove)
    elif isinstance(json_object, dict):
        for field in fields_to_remove:
            parts = field.split(".")
            if len(parts) == 1:
                json_object.pop(parts[0], None)
            else:
                if parts[0] in json_object:
                    object_recursive_delete_fields(
                        json_object[parts[0]], [".".join(parts[1:])]
                    )
        # Recursively apply to nested dictionaries
        for key in list(json_object.keys()):
            object_recursive_delete_fields(json_object[key], fields_to_remove)


def correct_missing_types(metadata: dict[str, dict[str, str]]) -> None:
    """
    This function adds the type (i.e., movie, music, or book) to every item in the given metadata for which this
    field is None.

    Note it directly updates the given metadata file.

    :param metadata: json file on which this operation has to be performed.
    """
    # get music ASINs
    music = pd.read_csv("./data/processed/legacy/reviews_CDs_and_Vinyl_5.csv")
    music_asins = {k: 0 for k in list(music["itemId"].unique())}
    # get movie ASINs
    movie = pd.read_csv("./data/processed/legacy/reviews_Movies_and_TV_5.csv")
    movie_asins = {k: 0 for k in list(movie["itemId"].unique())}
    # modify metadata
    for asin, data in metadata.items():
        if data["type"] is None:
            if asin in music_asins:
                data["type"] = "music"
            elif asin in movie_asins:
                data["type"] = "movies"
            else:
                data["type"] = "books"


def get_metadata_stats(metadata):
    """
    This function computes some metadata statistics on the given file.
    It groups data into categories (books, music, movie) and counts the number of items without title, person, or
    year.

    :param metadata: metadata JSON file on which the statistics have to be computed
    """
    # open metadata
    with open(metadata) as json_file:
        m_data = json.load(json_file)
    # create statistics dict
    stats = {
        "movies_and_tv": {"title": 0, "person": 0, "year": 0},
        "cds_and_vinyl": {"title": 0, "person": 0, "year": 0},
        "books": {"title": 0, "person": 0, "year": 0},
    }
    for asin, data in m_data.items():
        if data["title"] is None:
            stats[data["type"]]["title"] += 1
        if data["person"] is None:
            stats[data["type"]]["person"] += 1
        if data["year"] is None:
            stats[data["type"]]["year"] += 1
    logger.info(stats)
