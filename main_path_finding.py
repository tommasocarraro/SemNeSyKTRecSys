from nesy.paths.neo4j.path_finder import neo4j_path_finder

if __name__ == '__main__':
    neo4j_path_finder("data/processed/mappings/mapping-movies.json",
                      "data/processed/mappings/mapping-music.json",
                      "./data/processed/paths/path-movies-music.json",
                      max_hops=10, n_cores=120)
    # todo vedere se il discorso di pre-caricare il grafo puo' aiutare
    # todo bisogna per forza rendere piu' efficienti le cose gestendo la dimensione del batch
