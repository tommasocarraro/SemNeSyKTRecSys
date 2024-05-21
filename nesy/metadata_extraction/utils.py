import json
import pandas as pd


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


def correct_missing_types(metadata):
    """
    This function adds the type (i.e., movie, music, or book) to every item in the given metadata for which this
    field is None.

    Note it directly updates the given metadata file.

    :param metadata: json file on which this operation has to be performed.
    """
    # open metadata
    with open(metadata) as json_file:
        m_data = json.load(json_file)
    # get music ASINs
    music = pd.read_csv("./data/processed/legacy/reviews_CDs_and_Vinyl_5.csv")
    music_asins = list(music["itemId"].unique())
    # get movie ASINs
    movie = pd.read_csv("./data/processed/legacy/reviews_Movies_and_TV_5.csv")
    movie_asins = list(movie["itemId"].unique())
    # get book ASINs
    book = pd.read_csv("./data/processed/reviews_Books_5.csv")
    book_asins = list(book["itemId"].unique())
    # modify metadata
    for asin, data in m_data.items():
        if data["type"] is None:
            if asin in music_asins:
                data["type"] = "cds_and_vinyl"
            elif asin in movie_asins:
                data["type"] = "movies_and_tv"
            else:
                data["type"] = "books"
    # update file
    with open(metadata, 'w', encoding='utf-8') as f:
        json.dump(m_data, f, ensure_ascii=False, indent=4)
