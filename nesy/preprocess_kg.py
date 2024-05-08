import shutil
from .kgtk_wrappers import (
    kgtk_query,
    kgtk_add_id,
    kgtk_cat,
    kgtk_build_cache,
    kgtk_filter,
)
from os.path import dirname, join, basename
from os import makedirs
from .utils import remove_ext, compute_graph_extension


def preprocess_kg(
    input_graph: str,
    cache_path: str,
    debug: bool,
    index_mode: str = "mode:graph",
    compress_inter_steps: bool = False,
):
    """
    Preprocesses the knowledge graph by performing the following steps:
    1. Imports the graph into a temporary cache.
    2. Extracts all properties and computes their inverse.
    3. Generates IDs for the inversed properties.
    4. Concatenates the original and the inversed properties.
    5. Builds the graph cache.
    6. Cleans up temporary files.

    Args:
        input_graph (str): Path to the input graph file.
        cache_path (str): Path to the graph cache file.
        debug (bool): Flag indicating whether to enable debug mode.
        index_mode (str, optional): Index mode for building the graph cache. Defaults to "mode:graph".
        compress_inter_steps (bool, optional): Flag indicating whether to compress temporary files during intermediate. Defaults to False.
    """
    graph_name = remove_ext(basename(input_graph))
    base_temp_dir = join(dirname(input_graph), "tmp")
    makedirs(base_temp_dir, exist_ok=True)

    print("Importing graph into temporary cache")
    temp_graph_cache = join(base_temp_dir, "temp_cache.sqlite3.db")
    kgtk_build_cache(input_graph=input_graph, graph_cache=temp_graph_cache, debug=debug)

    # optionally use filter to remove unwanted properties
    # print("Filtering out unwanted properties")
    # filtered_graph = join(
    #     base_temp_dir,
    #     graph_name + "_filtered" + compute_extension(compress_inter_steps),
    # )
    # filter(
    #     input_graph=input_graph,
    #     output_path=filtered_graph,
    #     word_separator="|",
    #     invert=True,
    #     pattern=" ; P364|P21|P407|P1889|P103|P27|P21|P495; ",
    # )

    print("Extracting all properties and computing their inverse")
    output_inverse_graph = join(
        base_temp_dir,
        graph_name + "_inverse" + compute_graph_extension(compress_inter_steps),
    )
    kgtk_query(
        input_graph=input_graph,
        match_clause="(n1)-[r1]->(n2)",
        return_clause="n2 as node1, CONCAT(r1.label, '_') as label, n1 as node2",
        output_path=output_inverse_graph,
        debug=debug,
        graph_cache=temp_graph_cache,
    )

    print("Generating IDs for the inversed properties")
    output_inverse_ids_graph = join(
        base_temp_dir,
        graph_name
        + "_inverse_with_ids"
        + compute_graph_extension(compress_inter_steps),
    )
    kgtk_add_id(input_graph=output_inverse_graph, output_path=output_inverse_ids_graph)

    print("Concatenating the original and the inversed properties")
    concatenated_graph = join(dirname(input_graph), graph_name + "_preprocessed.tsv.gz")
    kgtk_cat(
        input_graphs=[input_graph, output_inverse_ids_graph],
        output_path=concatenated_graph,
    )

    print("Building the graph cache")
    kgtk_build_cache(
        input_graph=concatenated_graph,
        graph_cache=cache_path,
        index_mode=index_mode,
        debug=debug,
    )

    print("Cleaning up temporary files")
    shutil.rmtree(base_temp_dir)

    print(f"Final knowledge graph saved at {concatenated_graph}")


# Example usage
# kg = "data/wikidata/claims.wikibase-item.tsv.gz"
# cache = "data/wikidata/graph-cache.sqlite3.db"
# preprocess_kg(input_graph=kg, cache_path=cache, compress_inter_steps=True, debug=True)
