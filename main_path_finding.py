from nesy.paths.neo4j.path_finder import neo4j_path_finder

if __name__ == '__main__':
    neo4j_path_finder("data/processed/mappings/mapping-movies.json",
                      "data/processed/mappings/mapping-music.json",
                      max_hops=10, n_cores=120, cold_start=True, popular=True, cs_threshold=5, pop_threshold=200)
