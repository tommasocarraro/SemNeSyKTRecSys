from nesy.paths.neo4j.path_finder import neo4j_path_finder
from nesy.paths.neo4j.utils import get_cold_start, get_rating_stats, refine_cold_start_items

if __name__ == '__main__':
    neo4j_path_finder("data/processed/mappings/mapping-movies.json",
                      "data/processed/mappings/mapping-music.json",
                      max_hops=10, n_cores=10, cold_start=True, popular=True, cs_threshold=5, pop_threshold=300)
    # todo piano B -> distribuzioni dataset oppure eliminare cio' che abbiamo matchato con solo titolo
    # todo -> vedo i cold-start user. per ogni user, vedo gli item -> questa e' una buona opzione per il cold-start user
