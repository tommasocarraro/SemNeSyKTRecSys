from nesy.paths.entity_linking.wiki_special_search_linker import entity_linker_api_query
from nesy.paths.entity_linking.utils import linking_stats, get_ids

if __name__ == "__main__":
    # todo per eccezioni uso quello originale se la regex elimina tutto
    # entity_linker_api_query("./complete-movies.json", "./mapping-movies.json",
    #                         n_cores=1, retry_reason="not-in-correct-category")
    # entity_linker_api_query("./data/processed/complete-books.json", "./data/processed/mapping-books.json",
    #                         n_cores=1, retry_reason="long-title-exception")
    linking_stats("./data/processed/mapping-movies.json", errors=["not-title", "not-found-query",
                                                                  "exception", "not-in-correct-category", "long-title-exception"])
    linking_stats("./data/processed/mapping-music.json", errors=["not-title", "not-found-query",
                                                                 "exception", "not-in-correct-category", "long-title-exception"])
    linking_stats("./data/processed/mapping-books.json", errors=["not-title", "not-found-query",
                                                                 "exception", "not-in-correct-category", "long-title-exception"])
