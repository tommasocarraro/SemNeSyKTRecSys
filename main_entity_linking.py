from nesy.paths.entity_linking.wiki_special_search_linker import entity_linker_api_query
from nesy.paths.entity_linking.utils import linking_stats, get_ids

if __name__ == "__main__":
    # todo per eccezioni uso quello originale se la regex elimina tutto
    entity_linker_api_query("./data/processed/augmented-movies.json", "./data/processed/mapping-aug-movies.json",
                            n_cores=10)
    entity_linker_api_query("./data/processed/augmented-music.json", "./data/processed/mapping-aug-music.json",
                            n_cores=10)
    entity_linker_api_query("./data/processed/augmented-books.json", "./data/processed/mapping-aug-books.json",
                            n_cores=10)
    linking_stats("./data/processed/mapping-aug-movies.json", errors=["not-title", "not-found-query",
                                                                      "exception", "not-in-correct-category",
                                                                      "long-title-exception"])
    linking_stats("./data/processed/mapping-aug-music.json", errors=["not-title", "not-found-query",
                                                                     "exception", "not-in-correct-category",
                                                                     "long-title-exception"])
    linking_stats("./data/processed/mapping-aug-books.json", errors=["not-title", "not-found-query",
                                                                     "exception", "not-in-correct-category",
                                                                     "long-title-exception"])
