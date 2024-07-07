from nesy.paths.entity_linking.wiki_special_search_linker import entity_linker_api_query
from nesy.paths.entity_linking.utils import linking_stats, get_ids

if __name__ == "__main__":
    # entity_linker_api_query("./data/processed/complete-movies.json", "./data/processed/mapping-movies.json", n_cores=1, retry_reason="exception")
    # entity_linker_api_query("./data/processed/complete-books.json", "./mapping-books.json", n_cores=10)
    # linking_stats("./mapping-movies.json", errors=["not-title", "not-found-query",
    #                                                "exception", "not-in-correct-category"])
    linking_stats("./data/processed/mapping-movies.json", errors=["not-title", "not-found-query",
                                                   "exception", "not-in-correct-category"])
    linking_stats("./data/processed/mapping-music.json", errors=["not-title", "not-found-query",
                                                  "exception", "not-in-correct-category"])
    linking_stats("./data/processed/mapping-books.json", errors=["not-title", "not-found-query",
                                                  "exception", "not-in-correct-category"])
