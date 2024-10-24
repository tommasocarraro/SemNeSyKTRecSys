from nesy.paths.paths import get_multiple_paths
from nesy.paths.labels import create_wikidata_labels_sqlite, generate_all_labels
from nesy.preprocess_kg import preprocess_kg
import pandas as pd


def main():
    preprocess_flag = True
    paths_flag = False
    labels_cache_flag = False
    labels_flag = False

    # this dataset was provided by the KGTK developers, but it's been since deleted from their servers
    # sorry for the inconvenience
    kg = "./data/wikidata/claims.wikibase-item.tsv.gz"
    cache = "./data/wikidata/graph-cache.sqlite3.db"
    kg_preprocessed = "./data/wikidata/claims.wikibase-item_preprocessed.tsv.gz"

    # first we preprocess the graph and create the cache
    if preprocess_flag:
        selected_relations = pd.read_csv("./data/wikidata/selected-relations.csv")[
            "ID"
        ].tolist()
        kg_preprocessed = preprocess_kg(
            input_graph=kg,
            cache_path=cache,
            compress_inter_steps=False,
            debug=True,
            selected_properties=selected_relations,
        )

    # then we define the list of (source, target) pairs and compute the paths
    if paths_flag:
        pairs = [
            # 2001: A Space Odyssey -> The Blue Danube
            ("Q103474", "Q482621"),
            # Waldmeister -> 2001: A Space Odyssey
            ("Q7961534", "Q103474"),
            # The Rains of Castamere -> Game of Thrones
            ("Q18463992", "Q23572"),
            # Do Androids Dream of Electric Sheep? -> Blade Runner 2049
            ("Q605249", "Q21500755"),
            # Ready Player One (book) -> Ready Player One (film)
            ("Q3906523", "Q22000542"),
            # The Lord of the Rings: The Two Towers -> The Hobbit (book)
            ("Q164963", "Q74287"),
            # American Pie Presents: Band Camp -> The Anthem
            ("Q261044", "Q3501212"),
            # New Divide -> Transformers
            ("Q19985", "Q171453"),
            # Halloween -> Dragula
            ("Q909063", "Q734624"),
            # Timeline -> Jurassic Park
            ("Q732060", "Q167726"),
            # My Heart Will Go On -> Inception
            ("Q155577", "Q25188"),
            # The Godfather -> The Sicilian
            ("Q47703", "Q960155"),
            # The Girl with the Dragon Tattoo (podcast episode) - > The Girl Who Played with Fire (book)
            ("Q116783360", "Q1137369"),
        ]
        get_multiple_paths(
            input_graph=kg_preprocessed,
            graph_cache=cache,
            output_dir="data/paths",
            pairs=pairs,
            max_hops=3,
            debug=False,
            n_jobs=6,
        )

    # lastly we generate the paths files with labels instead of IDs
    if labels_cache_flag:
        create_wikidata_labels_sqlite("./data/wikidata/labels.en.tsv")
    if labels_flag:
        generate_all_labels("./data/paths")


if __name__ == "__main__":
    main()
