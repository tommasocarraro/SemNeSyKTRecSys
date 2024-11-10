from src.entity_linking import linking_stats

if __name__ == "__main__":
    # entity_linker_api_query("./data/processed/augmented-movies.json", "./data/processed/mapping-aug-movies.json",
    #                         n_cores=10)
    # entity_linker_api_query("./data/processed/augmented-music.json", "./data/processed/mapping-aug-music.json",
    #                         n_cores=10)
    # entity_linker_api_query("./data/processed/augmented-books.json", "./data/processed/mapping-aug-books.json",
    #                         n_cores=10)
    linking_stats(
        "./data/processed/mappings/mapping-movies.json",
        errors=[
            "not-title",
            "not-found-query",
            "exception",
            "not-in-correct-category",
            "long-title-exception",
        ],
    )
    linking_stats(
        "./data/processed/mappings/mapping-music.json",
        errors=[
            "not-title",
            "not-found-query",
            "exception",
            "not-in-correct-category",
            "long-title-exception",
        ],
    )
    linking_stats(
        "./data/processed/mappings/mapping-books.json",
        errors=[
            "not-title",
            "not-found-query",
            "exception",
            "not-in-correct-category",
            "long-title-exception",
        ],
    )
