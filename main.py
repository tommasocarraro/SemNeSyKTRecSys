# STATISTICS METADATA
# start: 363876 matched and 116159 without title
# first scraping: 456341 matched and 441 are captcha-or-DOM and 23253 are 404 error
# end: 467833 matched and 11723 are 404 error and 479 are captcha in wayback -> captcha are more because some of the
# 404 error before became a captcha after Wayback Machine. Note also that about 70 items with a DOM problem have to be
# manually inserted in the file because Soup was not finding an ID that was actually there

import os

from nesy.data import create_asin_metadata_json, create_pandas_dataset,\
    entity_linker_api, entity_linker_api_query, metadata_scraping, filter_metadata, metadata_stats, metadata_cleaning, get_wid_per_cat
import json
import pandas as pd

if __name__ == "__main__":
    # filter_metadata("./data/processed/metadata.json", ["reviews_Movies_and_TV_5",
    #                                                    "reviews_Books_5",
    #                                                    "reviews_CDs_and_Vinyl_5"])
    # create_asin_metadata_json("./data/raw/metadata.json")
    # metadata_scraping("./data/processed/metadata.json")
    create_pandas_dataset("./data/raw/reviews_CDs_and_Vinyl_5.json")
    # entity_linker_api_query("./data/processed/reviews_Movies_and_TV_5.csv", True)
    # get_wid_labels("./data/raw/movies.json")
    # get_wid_labels("./data/raw/music.json")
    # get_wid_labels("./data/raw/books.json")
    # entity_linker_api_query("./data/processed/reviews_Movies_and_TV_5.csv", use_dump=True, retry=True)
    # todo completare e verificare che sia tutto ok e perche' non mi carica i problemi di DOM correttamente
    # metadata_scraping("./data/processed/final-metadata.json", 1,
    #                   motivation="DOM", save_tmp=True, batch_size=20, wayback=True)
    # metadata_stats("./data/processed/filtered-metadata.json", errors=["no-title"], save_asins=False)
    # metadata_stats("./data/processed/complete-filtered-metadata.json", errors=["captcha-or-DOM", "404-error"], save_asins=False)
    # metadata_stats("./data/processed/final-metadata.json", errors=["404-error", "DOM", "captcha"], save_asins=False)
    # metadata_stats("./data/processed/mapping-reviews_Movies_and_TV_5.json", errors=["not-found", "not-in-dump", "exception"], save_asins=False)
    # get_wid_per_cat("movies")
    # metadata_cleaning("./data/processed/final-metadata.json")
    # # todo fare il contrario da wikidata ad amazon, come check che i match sono corretti
    # with open("./data/processed/mapping-reviews_Movies_and_TV_5.json") as json_file:
    #     mapping = json.load(json_file)
    # p = pd.read_csv("./data/processed/reviews_Movies_and_TV_5.csv")
    # print(p["itemId"].nunique())
    # print(len(mapping))
    # print((len([m for m in mapping if mapping[m] != ""])))
    # with open("./data/processed/complete-filtered-metadata.json") as json_file:
    #     m_data = json.load(json_file)
    # # take the ASINs for the products that have a missing title in the metadata file
    # no_titles = [k for k, v in m_data.items() if v == "404-error"]
    # # collect_wayback_links(no_titles[:100], os.cpu_count(), 100)
    # scrape_title_wayback_api_2(no_titles[:100], 1, 1, False)

    # with open("./data/processed/complete-filtered-metadata.json") as json_file:
    #     m_data = json.load(json_file)
    # for filename in os.listdir("./data/processed"):
    #     f = os.path.join("./data/processed", filename)
    #     # checking if it is a file
    #     if os.path.isfile(f) and f[-1].isdigit():
    #         with open(f) as json_file:
    #             content = json.load(json_file)
    #             m_data.update(content)
    # with open('./data/processed/final-metadata.json', 'w', encoding='utf-8') as f:
    #     json.dump(m_data, f, ensure_ascii=False, indent=4)
