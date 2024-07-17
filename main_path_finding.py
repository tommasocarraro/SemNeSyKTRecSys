from nesy.paths.neo4j.path_finder import neo4j_path_finder

if __name__ == '__main__':
    neo4j_path_finder("data/processed/mappings/mapping-movies.json",
                      "data/processed/mappings/mapping-music.json",
                      "./data/processed/path-movies-music.json",
                      max_hops=2, n_cores=10)
