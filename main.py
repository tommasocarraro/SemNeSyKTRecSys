import os

from nesy.data import create_asin_metadata_json, create_pandas_dataset,\
    entity_linker_api, entity_linker_api_query, metadata_scraping, filter_metadata, metadata_stats
import json
import pandas as pd

if __name__ == "__main__":
    # filter_metadata("./data/processed/metadata.json", ["reviews_Movies_and_TV_5",
    #                                                    "reviews_Books_5",
    #                                                    "reviews_CDs_and_Vinyl_5"])
    # create_asin_metadata_json("./data/raw/metadata.json")
    # metadata_scraping("./data/processed/metadata.json")
    # create_pandas_dataset("./data/raw/reviews_Movies_and_TV_5_large.json")
    # entity_linker("./data/processed/reviews_Movies_and_TV_5.csv", "./data/processed/metadata.json")
    # get_wid_labels("./data/raw/movies.json")
    # get_wid_labels("./data/raw/music.json")
    # get_wid_labels("./data/raw/books.json")
    # entity_linker_api_query("./data/processed/reviews_Movies_and_TV_5.csv", use_dump=True)
    # metadata_scraping("./data/processed/filtered-metadata.json", os.cpu_count())
    metadata_stats("./data/processed/complete-filtered-metadata.json")
    # # todo fare il contrario da wikidata ad amazon, come check che i match sono corretti
    # with open("./data/processed/mapping-reviews_Movies_and_TV_5.json") as json_file:
    #     mapping = json.load(json_file)
    # p = pd.read_csv("./data/processed/reviews_Movies_and_TV_5.csv")
    # print(p["itemId"].nunique())
    # print(len(mapping))
    # print((len([m for m in mapping if mapping[m] != ""])))


"""
SELECT DISTINCT ?item
WHERE {
  {
    ?item wdt:P31/wdt:P279* wd:Q11424.
  }
  UNION
  {
    ?item wdt:P31/wdt:P279* wd:Q15416.
  }
}
"""
