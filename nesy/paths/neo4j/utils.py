import pandas as pd
import json


def compute_cold_start_pairs(mapping_1: dict, mapping_2: dict) -> dict:
    """
    This function takes as input two mapping dictionaries, one from source domain and the other for target domain.
    Based on statistics about the ratings in the two domains, it creates source-target pairs for which semantic
    paths have to be generated. The logic is simple. The cold-start items in the target domain are selected. These items
    are the ones for which it is crucial to get some information from the source domain.

    :param mapping_1: dictionary containing the source domain mapping
    :param mapping_2: dictionary containing the target domain mapping
    :return: list of pairs on which the paths have to be computed
    """


def get_rating_stats(rating_file: str, entity: str) -> dict:
    """
    This function takes as input a rating file and counts the number of ratings for each user or item.
    The stats are returned in a dict.

    :param rating_file: path to rating file
    :param entity: whether to count the number of ratings of users or items
    :return: dictionary reporting the stats
    """
    if entity == 'item':
        df_col = 'itemId'
    elif entity == 'user':
        df_col = 'userId'
    else:
        raise ValueError("Entity must be 'item' or 'user'")
    df = pd.read_csv(rating_file)
    ratings_count = df.groupby(df_col).size().to_dict()
    # sort dictionary by the rating count
    return dict(sorted(ratings_count.items(), key=lambda entity_: entity_[1]))


def get_cold_start(stats: dict, threshold: int) -> list:
    """
    This function takes statistics about ratings in the dataset and returns the list of items (or users)
    that have less than or equal threshold ratings.

    :param stats: dictionary containing stats about ratings
    :param threshold: the threshold to select items (or users)
    :return: list of items (or users) with less than or equal threshold ratings
    """
    return [id_ for id_, count in stats.items() if count <= threshold]


def refine_cold_start_items(cold_start_list: list, target_mapping_file: str) -> list:
    """
    This function takes a cold-start list of items and refines it by removing items that have not been matched with
    Wikidata.

    :param cold_start_list: list of cold-start items
    :param target_mapping_file: mapping file containing the matches in the target domain
    :return: refined cold-start list
    """
    with open(target_mapping_file, 'r') as json_file:
        target_mapping = json.load(json_file)
    return [id_ for id_ in cold_start_list if isinstance(target_mapping[id_], dict)]
