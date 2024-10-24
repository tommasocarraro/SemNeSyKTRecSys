from src.data import process_source_target

process = process_source_target(0, "./data/ratings/reviews_Books_5.csv",
                                "./data/ratings/reviews_Movies_and_TV_5.csv",
                                "./data/kg_paths/books(pop:300)-movies(cs:5).json")
