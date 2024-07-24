from nesy.paths.neo4j.path_finder import neo4j_path_finder

if __name__ == '__main__':
    neo4j_path_finder("data/processed/mappings/mapping-movies.json",
                      "data/processed/mappings/mapping-music.json",
                      "./data/processed/paths/path-movies-music.json",
                      max_hops=10, n_cores=10)
    # todo adesso siamo rimasti che bisogna capire perche' si ferma e non e' lineare l'andamento
    # todo vedere discorso memoria con consigli su sito Neo4j -> fatto
    # todo eliminare piu' relazioni e nodi possibili -> fatto
    # todo caricare in cache con una query stupida -> fatto
    # todo piano B -> distribuzioni dataset oppure eliminare cio' che abbiamo matchato con solo titolo